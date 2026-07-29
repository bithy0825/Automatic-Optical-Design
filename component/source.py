from typing import override, Self, Any
from collections.abc import Iterator, Mapping

import torch
import torch.nn.functional as F
from jaxtyping import Float

from core import (
    OpticalModule,
    RayFloat2D,
    RayFloat3D,
    RayFloatScalar,
    RayBundle,
    SystemBoolScalar,
    TraceFlow,
    Transformer,
    Verdict,
    fmt_param,
    term,
)
from component.protocol import Component
from core.repr import styled
from materials import Material
from sampling import SampleOptions, sample

# 视场角物理边界：|θ| ≤ 90°，超过则光线朝 -z 倒退，对前向追迹无意义。
_MAX_FIELD_DEG: float = 90.0


def _direction_from_angles(
    field_rad: Float[torch.Tensor, "F 2"],
) -> Float[torch.Tensor, "F 3"]:
    """视场角 (rad) → 单位方向矢量（z 为光轴）。

    方向 ∝ (tanθx, tanθy, 1)，为避免 tan 在 ±90° 附近的爆炸与符号翻转，
    改用等价的 (sinθx·cosθy, cosθx·sinθy, cosθx·cosθy) 计算，并 clamp
    保证光轴分量 Vz ≥ 0。
    """
    tx, ty = field_rad.unbind(dim=-1)
    cx, cy = tx.cos(), ty.cos()
    vz = cx.mul(cy).clamp_min(0.0)  # 光轴分量，保证非负
    d = torch.stack(
        (
            tx.sin().mul(cy),  # Vx
            cx.mul(ty.sin()),  # Vy
            vz,
        ),  # Vz
        dim=-1,
    )
    return F.normalize(d, dim=-1)


def _denorm(
    s: Float[torch.Tensor, "F"], rng: tuple[float, float]
) -> Float[torch.Tensor, "F"]:
    """[-1, 1] 归一化采样 → [lo, hi] 仿射映射。"""
    lo, hi = rng
    return s.add(1.0).mul(0.5 * (hi - lo)).add(lo)


def _interp_wavelength(
    samples: Float[torch.Tensor, "W"],
    configured: tuple[float, ...],
) -> Float[torch.Tensor, "W"]:
    """[-1, 1] 采样位置 → 物理波长 (nm)。

    *configured* 视为 [-1, 1] 上均布的控制点线性插值；采样数与控制点数
    相同时直接返回配置值，与设计波长精确对齐。
    """
    m = len(configured)
    if m == 1:
        return samples.new_full(samples.shape, configured[0])
    if samples.shape[0] == m:
        return samples.new_tensor(configured)

    pos = samples.add(1.0).mul(0.5 * (m - 1)).clamp(0.0, m - 1.0)  # → [0, m-1]
    i0 = pos.floor().long().clamp_max(m - 2)
    ctrl = samples.new_tensor(configured)
    return torch.lerp(ctrl[i0], ctrl[i0 + 1], pos - i0)


def _normalize_angle_range(
    val: float | tuple[float, float], *, name: str
) -> tuple[float, float]:
    """单值或 (min, max) → (min, max)，校验顺序与 ±90° 物理边界。"""
    match val:
        case int() | float():
            lo = hi = float(val)
        case ((int() | float()) as a, (int() | float()) as b):
            lo, hi = float(a), float(b)
        case _:
            raise TypeError(f"{name} must be a number or (min, max) pair, got {val!r}")

    if lo > hi:
        raise ValueError(f"{name} range inverted: ({lo}, {hi})")
    if max(abs(lo), abs(hi)) > _MAX_FIELD_DEG:
        raise ValueError(
            f"{name} must lie within ±{_MAX_FIELD_DEG}° (rays travel +z), "
            f"got ({lo}, {hi})"
        )
    return (lo, hi)


def _normalize_wavelengths(val: float | tuple[float, ...]) -> tuple[float, ...]:
    """单值或元组 → 升序正波长元组 (nm)。"""
    match val:
        case int() | float():
            ws = (float(val),)
        case str() | bytes():
            raise TypeError(f"wavelength must be number(s) in nm, got {val!r}")
        case _:
            ws = tuple(sorted(float(w) for w in val))

    if not ws:
        raise ValueError("wavelength must contain at least one value")
    if min(ws) <= 0.0:
        raise ValueError(f"wavelength must be positive (nm), got {ws}")
    return ws


def _validate_configs(
    pupil_cfg: SampleOptions, field_cfg: SampleOptions, wavel_cfg: SampleOptions
) -> None:
    """校验采样配置与光源语义的兼容性。"""
    for name, cfg in (("pupil_cfg", pupil_cfg), ("field_cfg", field_cfg)):
        if cfg.region == "line":
            raise ValueError(
                f"{name}.region must be 'rect' or 'disk', got {cfg.region!r}"
            )
    if wavel_cfg.region != "line":
        raise ValueError(f"wavel_cfg.region must be 'line', got {wavel_cfg.region!r}")


class InfiniteSource(Component):
    """无限远物方光源：平行光入射 z=0 入瞳平面。

    光线起点分布于直径 *epd* 的入瞳上，方向仅由视场角决定（物在无限远，
    同一视场的所有光线互相平行）。种群个体共享同一照明条件，光源无
    可变异参数（``mutable`` 为空）。

    Args:
        epd:        入瞳直径 (mm)，必须为正。
        field_x:    x 视场角 (deg)，单值或 (min, max)，|θ| ≤ 90°。
        field_y:    y 视场角 (deg)，同上。
        wavelength: 设计波长 (nm)，单值或元组。
        population: 种群大小 P（共享照明的独立光学系统数）。
        pupil_cfg:  光瞳采样配置，region 须为 ``"rect"`` / ``"disk"``。
        field_cfg:  视场采样配置，region 须为 ``"rect"`` / ``"disk"``。
        wavel_cfg:  波长采样配置，region 须为 ``"line"``。
        transmitted: 出射侧材料，缺省为空气。
    """

    kind = term.SOURCE

    def __init__(
        self,
        epd: float,
        field_x: float | tuple[float, float],
        field_y: float | tuple[float, float],
        wavelength: float | tuple[float, ...],
        *,
        population: int = 1,
        pupil_cfg: SampleOptions | None = None,
        field_cfg: SampleOptions | None = None,
        wavel_cfg: SampleOptions | None = None,
        transmitted: Material | None = None,
    ) -> None:
        super().__init__()
        if population < 1:
            raise ValueError(f"population must be >= 1, got {population}")
        if epd <= 0:
            raise ValueError(f"epd must be positive, got {epd}")

        if transmitted is None:
            transmitted = Material.from_name(population, "air")
        self.transmitted = transmitted

        if pupil_cfg is None:
            pupil_cfg = SampleOptions(method="uniform", region="rect", count=(128, 1))
        if field_cfg is None:
            field_cfg = SampleOptions(method="uniform", region="rect", count=(3, 1))
        if wavel_cfg is None:
            wavel_cfg = SampleOptions(method="uniform", region="line", count=3)
        _validate_configs(pupil_cfg, field_cfg, wavel_cfg)

        self.pupil_cfg = pupil_cfg
        self.field_cfg = field_cfg
        self.wavel_cfg = wavel_cfg

        self.epd = float(epd)
        self.field_x = _normalize_angle_range(field_x, name="field_x")
        self.field_y = _normalize_angle_range(field_y, name="field_y")
        self.wavelengths = _normalize_wavelengths(wavelength)

    @override
    def _label(self) -> str:
        return styled(
            "InfiniteSource",
            f"epd={self.epd}, fov_x={self.field_x}, fov_y={self.field_y}, "
            f"λ={self.wavelengths}, P={self.population}",
        )

    @override
    def _params(self) -> Iterator[str]:
        yield f"transmitted: {fmt_param(self.transmitted.names())}"

    @override
    def forward(self, flow: TraceFlow | None = None) -> TraceFlow:
        """发射初始光线：*flow* 缺省时新建单位变换的 TraceFlow，否则替换其光线。"""
        tf = (
            flow.transformer
            if flow is not None
            else Transformer.identity(
                self.population, device=self.device, dtype=self.dtype
            )
        )

        P, V, pupil, field, wavelength = self._emit_rays(
            tf,
            sample(self.pupil_cfg).to(device=tf.device, dtype=tf.dtype),
            sample(self.field_cfg).to(device=tf.device, dtype=tf.dtype),
            sample(self.wavel_cfg).to(device=tf.device, dtype=tf.dtype),
        )

        rays = RayBundle(
            points=P, directions=V, pupil=pupil, field=field, wavelength=wavelength
        )
        verdict = Verdict.alive_like(P[..., 0])

        if flow is None:
            return TraceFlow(rays=rays, transformer=tf, verdict=verdict)
        return flow.with_rays(rays).with_transformer(tf).at_verdict(verdict)

    @override
    def clone(self) -> Self:
        """深拷贝：照明配置原样沿用，**出射材料克隆为新 Material**（独立 Indices）。

        光源无可训练参数，``epd`` / 视场角 / 波长 / 采样配置皆为只读配置，重用
        即可；唯一可变状态是 ``transmitted``（下游链由此 MaterialRef 视图读取），
        必须 clone，否则两个系统的材料会共享 Indices、变异互相泄漏。
        """
        return type(self)(
            epd=self.epd,
            field_x=self.field_x,
            field_y=self.field_y,
            wavelength=self.wavelengths,
            population=self.population,
            pupil_cfg=self.pupil_cfg,
            field_cfg=self.field_cfg,
            wavel_cfg=self.wavel_cfg,
            transmitted=self.transmitted.clone(),
        )

    def _emit_rays(
        self,
        tf: Transformer,
        pupil_pts: Float[torch.Tensor, "N 2"],
        field_pts: Float[torch.Tensor, "F 2"],
        wavel_pts: Float[torch.Tensor, "W"],
    ) -> tuple[RayFloat3D, RayFloat3D, RayFloat2D, RayFloat2D, RayFloatScalar]:
        """归一化采样点 → 物理光线 (P, V) 与逐光线标签 (pupil, field, wavelength)。"""
        B, f, w, n = (
            tf.population,
            field_pts.shape[0],
            wavel_pts.shape[0],
            pupil_pts.shape[0],
        )

        pupil = pupil_pts.mul(0.5 * self.epd)  # (N, 2) 物理光瞳坐标
        field = torch.deg2rad(
            torch.stack(
                (
                    _denorm(field_pts[:, 0], self.field_x),
                    _denorm(field_pts[:, 1], self.field_y),
                ),
                dim=-1,
            )
        )  # (F, 2) rad
        wavelength = _interp_wavelength(wavel_pts, self.wavelengths)  # (W,) nm

        P = torch.cat((pupil, pupil.new_zeros(n, 1)), dim=-1)  # (N, 3) z=0 入瞳平面
        V = _direction_from_angles(field)  # (F, 3)

        P = tf.transform_points(P.view(1, 1, 1, n, 3).expand(B, f, w, n, 3))
        V = tf.transform_vectors(V.view(1, f, 1, 1, 3).expand(B, f, w, n, 3))

        pupil_out = pupil.view(1, 1, 1, n, 2).expand(B, f, w, n, 2)
        field_out = field.view(1, f, 1, 1, 2).expand(B, f, w, n, 2)
        wavelength_out = wavelength.view(1, 1, w, 1).expand(B, f, w, n)

        return P, V, pupil_out, field_out, wavelength_out

    @classmethod
    @override
    def where(cls, mask: SystemBoolScalar, new: Self, old: Self) -> Self:
        """照明配置（epd/视场/波长/采样）为种群共享配置，从 *new* 继承；
        唯一逐个体状态 ``transmitted`` 经 ``Material.where`` 合并。"""
        OpticalModule._check_operands(mask, new, old)
        return cls(
            epd=new.epd,
            field_x=new.field_x,
            field_y=new.field_y,
            wavelength=new.wavelengths,
            population=new.population,
            pupil_cfg=new.pupil_cfg,
            field_cfg=new.field_cfg,
            wavel_cfg=new.wavel_cfg,
            transmitted=Material.where(mask, new.transmitted, old.transmitted),
        )

    @classmethod
    @override
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        fx, fy = _field_ranges(term.FOV.resolve(options))
        return cls(
            epd=term.EPD.resolve(options),
            field_x=fx,
            field_y=fy,
            wavelength=term.WAVELENGTH.resolve(options),
            population=population,
            pupil_cfg=SampleOptions.from_options(term.PUPIL.resolve(options)),
            field_cfg=SampleOptions.from_options(term.FIELD.resolve(options)),
            wavel_cfg=SampleOptions.from_options(term.WAVEL.resolve(options)),
        )


def _field_ranges(fov: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    if isinstance(fov, (int, float)):
        half = float(fov) / 2.0
        return (-half, half), (0.0, 0.0)

    if isinstance(fov, (tuple, list)):
        if len(fov) != 2:
            raise TypeError(f"fov must have 2 elements, got {len(fov)}: {fov!r}")
        x, y = fov
        if isinstance(x, (int, float)):
            return (-float(x) / 2.0, float(x) / 2.0), (-float(y) / 2.0, float(y) / 2.0)
        (xlo, xhi), (ylo, yhi) = fov
        return (float(xlo), float(xhi)), (float(ylo), float(yhi))
    raise TypeError(f"fov must be a number or 2-tuple, got {fov!r}")
