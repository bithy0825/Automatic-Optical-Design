"""优化目标规格：一个设计任务的标称参数（视场 / F 数 / 焦距 / 波长）。"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from core import term
from core.repr import render_line, styled

type FOV = float | tuple[float, float] | tuple[tuple[float, float], tuple[float, float]]


def _parse_fov(raw: Any) -> FOV:
    """``fov`` 配置值 → 规范化 FOV（全视场角 / 逐轴角 / 逐轴范围）。"""
    match raw:
        case int() | float():
            return float(raw)
        case [a, b] | (a, b) if isinstance(a, (int, float)):
            return (float(a), float(b))
        case [[a, b], [c, d]] | ((a, b), (c, d)):
            return ((float(a), float(b)), (float(c), float(d)))
        case _:
            raise TypeError(f"invalid fov: {raw!r}")


def _serialize_fov(fov: FOV) -> float | list[float] | list[list[float]]:
    """规范化 FOV → TOML 可写形式（``to_dict`` 用）。"""
    match fov:
        case float():
            return fov
        case (float() as a, float() as b):
            return [a, b]
        case ((float() as a, float() as b), (float() as c, float() as d)):
            return [[a, b], [c, d]]
        case _:
            raise RuntimeError(f"unreachable: {fov!r}")


def _fov_name(fov: FOV) -> str:
    """FOV → 紧凑文本（自动命名用）。"""
    match fov:
        case float() as f:
            return f"{f:g}"
        case (float() as a, float() as b):
            return f"{a:g}_{b:g}"
        case (
            (float() as h_min, float() as h_max),
            (float() as v_min, float() as v_max),
        ):
            return f"{h_min:g}_{h_max:g}_{v_min:g}_{v_max:g}"
        case _:
            raise RuntimeError(f"unreachable: {fov!r}")


@dataclass(frozen=True, slots=True)
class Target:
    """设计目标：一个优化任务瞄准的标称参数组合。"""

    id: str
    fov: FOV
    F: float
    effl: float
    wavelengths: list[float]  # nm

    @classmethod
    def from_options(cls, options: Mapping[str, float]) -> Self:
        """从 ``[target]`` 配置节构造；``id`` 缺省时按参数自动命名。"""
        fov = _parse_fov(term.FOV.resolve(options))
        F = float(term.F_NUMBER.resolve(options))
        effl = float(term.EFFL.resolve(options))

        wavelengths = [float(w) for w in term.WAVELENGTH.resolve(options, default=[550.0])]

        ident = term.ID.resolve(
            options,
            default=f"fov_{_fov_name(fov)}_F_{F:g}_effl_{effl:g}",
        )

        return cls(
            id=ident,
            fov=fov,
            F=F,
            effl=effl,
            wavelengths=wavelengths,
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为配置映射（词表规范键；并入 source 元件块用）。"""
        return {
            term.ID.canonical: self.id,
            term.FOV.canonical: _serialize_fov(self.fov),
            term.F_NUMBER.canonical: self.F,
            term.EFFL.canonical: self.effl,
            term.EPD.canonical: self.epd,
            term.WAVELENGTH.canonical: self.wavelengths,
        }

    @property
    def epd(self) -> float:
        """入瞳直径 (mm)：`EPD = EFFL / F`。"""
        return self.effl / self.F

    def __repr__(self) -> str:
        return render_line(
            styled(
                "Target",
                f"id={self.id!r}, fov={self.fov!r}, F={self.F:g}, "
                f"effl={self.effl:g}, epd={self.epd:g}, λ={self.wavelengths!r}",
            )
        )
