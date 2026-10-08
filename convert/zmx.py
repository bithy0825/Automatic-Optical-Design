"""Zemax .zmx 处方 → 训练配置 TOML。

用法：

    python -m convert.zmx <file.zmx> [out.toml]

成功：不给 out.toml 则打印到 stdout；给出则将 TOML 写入该路径（父目录不存在
自动创建）并打印保存路径。被过滤：打印 ``SKIP <文件名>: <原因>``，退出码 1，
不落盘、不建目录。

仅支持：STANDARD（球面/圆锥面）、EVENASPH（偶次非球面）、平面 STOP、像面；
EFFL / F 数 / 视场角按推断链解析。介质约定与过滤规则见
:mod:`convert._common`。
"""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from convert import _common as cm
from convert._common import Skip, Surface

_UNIT_SCALE = {"MM": 1.0, "CM": 10.0, "IN": 25.4, "METER": 1000.0}  # → mm


@dataclass(slots=True)
class Zmx:
    """zmx 文件的解析结果（头部 + 全部面，含 OBJ(0) 与 IMA(末)）。"""

    fnum: float | None = None
    enpd: float | None = None
    obna: float | None = None
    ftyp: int = 0
    fields: list[float] = field(default_factory=list)  # XFLN/XFLD/YFLN/YFLD 汇总
    merit_effl: float | None = None
    unit: str = "MM"
    mnum: int = 1  # MNUM：多重结构的配置数（>1 = 变焦/多位置，过滤）
    surfaces: list[Surface] = field(default_factory=list)


# ── 解析 ──


def _parse_surface_field(s: Surface, key: str, tokens: list[str]) -> None:
    try:
        match key:
            case "TYPE":
                s.type = tokens[1].upper()
            case "CURV":
                s.curv = float(tokens[1])
            case "DISZ":
                s.disz = None if tokens[1].upper().startswith("INF") else float(tokens[1])
            case "DIAM":
                s.diam = float(tokens[1])
            case "CONI":
                s.coni = float(tokens[1])
            case "PARM":
                s.parms[int(tokens[1])] = float(tokens[2])
            case "STOP":
                s.is_stop = True
            case "GLAS":
                s.glass = tokens[1]
                if len(tokens) > 4:
                    s.nd = float(tokens[4])
                if len(tokens) > 5:
                    s.vd = float(tokens[5])
    except (ValueError, IndexError):
        pass  # 未知/残缺字段：忽略，由过滤与推断阶段把关


def _parse_header(zmx: Zmx, key: str, tokens: list[str]) -> None:
    try:
        match key:
            case "FNUM":
                zmx.fnum = float(tokens[1])
            case "ENPD":
                zmx.enpd = float(tokens[1])
            case "OBNA":
                zmx.obna = float(tokens[1])
            case "FTYP":
                zmx.ftyp = int(float(tokens[1]))
            case "UNIT":
                zmx.unit = tokens[1].upper()
            case "MNUM":
                zmx.mnum = int(float(tokens[1]))
            case "XFLN" | "XFLD" | "YFLN" | "YFLD":
                zmx.fields.extend(float(t) for t in tokens[1:])
            case "EFFL":  # 评价函数 EFFL 操作数：第 7 个数值为目标值
                vals = [float(t) for t in tokens[1:]]
                if len(vals) >= 7:
                    zmx.merit_effl = vals[6]
    except (ValueError, IndexError):
        pass


def parse_zmx(path: Path) -> Zmx:
    """解析 .zmx 全文：头部关键字 + SURF 面块（缩进行归属当前面）。"""
    zmx = Zmx()
    cur: Surface | None = None
    for raw in cm.read_text(path).splitlines():
        if not raw.strip():
            continue
        indented = raw[0] in " \t"
        tokens = raw.split()
        key = tokens[0].upper()
        if not indented:
            cur = None
            if key == "SURF":
                cur = Surface(index=int(tokens[1]))
                zmx.surfaces.append(cur)
                continue
            _parse_header(zmx, key, tokens)
        elif cur is not None:
            _parse_surface_field(cur, key, tokens)
    return zmx


# ── 推断链 ──


def _infer_effl(zmx: Zmx, optical: list[Surface]) -> float:
    effl = cm.ynu_effl(optical)
    if effl is None:
        effl = zmx.merit_effl
    if effl is None or not 0.1 < effl < 1e5:
        raise Skip("cannot infer EFFL")
    return effl


def _infer_fnumber(zmx: Zmx, optical: list[Surface], effl: float) -> float:
    if zmx.fnum and zmx.fnum > 0:
        f = zmx.fnum
    elif zmx.enpd and zmx.enpd > 0:
        f = effl / zmx.enpd
    elif zmx.obna and zmx.obna > 0:
        f = 1.0 / (2.0 * zmx.obna)
    else:  # 光阑尺寸浮动：EPD ≈ 光阑全直径
        stop = next((s for s in optical if s.is_stop and s.diam > 0), None)
        if stop is None:
            raise Skip("cannot infer F-number")
        f = effl / (2.0 * stop.diam)
    if not 0.1 < f < 1e3:
        raise Skip("cannot infer F-number")
    return f


def _infer_fov(zmx: Zmx, effl: float) -> float:
    """最大半视场角（度）；文件无视场信息或推断越界时按焦距自动推断。"""
    hmax = max((abs(v) for v in zmx.fields), default=0.0)
    theta = 0.0
    if hmax > 0:
        if zmx.ftyp == 0:  # 角度（度）
            theta = hmax
        elif zmx.ftyp == 1:  # 物高 → 按物距换算
            obj = zmx.surfaces[0].disz
            if obj and obj > 0:
                theta = math.degrees(math.atan(hmax / obj))
        else:  # 2/3：像高 → 按焦距换算
            theta = math.degrees(math.atan(hmax / effl))
    if theta <= 0:  # 无视场信息：按焦距自动推断
        theta = cm.auto_fov(effl)
    # θ≥90°（鱼眼/垃圾视场 spec）不再被 auto 兜底掩盖，交给 convert 的 >45° 过滤
    return theta


# ── 转换主流程 ──


def convert(path: Path) -> str:
    """将一个 .zmx 处方转换为训练配置 TOML 文本；不可转换抛 :class:`Skip`。"""
    zmx = parse_zmx(path)
    scale = _UNIT_SCALE.get(zmx.unit)
    if scale is None:
        raise Skip(f"unknown unit: {zmx.unit}")
    if scale != 1.0:  # 统一折算成 mm（曲率除，长度乘；视场值按 FTYP 语义）
        for s in zmx.surfaces:
            s.curv /= scale
            if s.disz is not None:
                s.disz *= scale
            s.diam *= scale
        if zmx.merit_effl is not None:
            zmx.merit_effl *= scale
        if zmx.ftyp != 0:
            zmx.fields = [v * scale for v in zmx.fields]
    if zmx.mnum > 1:
        raise Skip(f"multi-configuration (MNUM {zmx.mnum})")
    if len(zmx.surfaces) < 3:
        raise Skip("no optical surfaces")

    optical = zmx.surfaces[1:-1]  # OBJ 与 IMA 之间
    for s in optical:
        if s.type not in ("STANDARD", "EVENASPH"):
            raise Skip(f"unsupported surface type: {s.type} (surf {s.index})")
        if s.glass and s.glass.upper() == "MIRROR":
            raise Skip(f"mirror surface (surf {s.index})")
        if s.glass and s.glass.upper() in cm.NON_GLASS:
            raise Skip(f"non-glass medium: {s.glass} (surf {s.index})")
        if s.disz is None:
            raise Skip(f"infinite thickness (surf {s.index})")
        if s.diam <= 0:  # 自动口径：以 2×EPD 兜底全直径，且不越半球域
            if zmx.enpd is None or zmx.enpd <= 0:
                raise Skip(f"zero diameter (surf {s.index})")
            s.diam = zmx.enpd
            if s.curv != 0.0:
                cap = math.floor(1.8 / abs(s.curv))
                if math.ceil(2.0 * s.diam) > cap:
                    s.diam = max(cap, 1) / 2.0
        a1 = s.parms.pop(1, 0.0)
        if a1 != 0.0:  # r² 项折入曲率（近轴 z ≈ (c + 2·a1)·r²/2），保留更高次项
            s.curv += 2.0 * a1

    optical = cm.drop_noop_surfaces(optical)
    if not optical:
        raise Skip("no optical surfaces")

    effl = cm.snap_effl(_infer_effl(zmx, optical))
    fnum = _infer_fnumber(zmx, optical, effl)
    theta = _infer_fov(zmx, effl)
    if theta > 45:
        raise Skip(f"extreme FOV ({theta:.1f}°)")

    std_scale = cm.fov_std_scale(theta)
    parts = [cm.header(path.stem, effl, fnum, theta)]
    # allow_negative 只给链上第一个面（与光源同面，t≈0 的数值噪声/首面负曲率
    # 的合法负根）；中间面负距离=X 型打架，必须保持判死
    for i, s in enumerate(optical):
        prev_air = i == 0 or optical[i - 1].glass is None
        # 光阑判定：正常光阑（STOP 标记；空气中的光阑任意曲率均无光焦度，一
        # 并按平面光阑转换）+ 空-空平面（判为光阑，多光阑，保留其口径裁切）。
        # 携带玻璃或前承玻璃的曲面光阑则退化为折射面，以免丢失光焦度。
        stop_plane = s.glass is None and (
            (s.is_stop and (cm.is_flat(s) or prev_air)) or (prev_air and cm.is_flat(s))
        )
        if stop_plane:
            parts.append(cm.stop_block(s, front=i == 0, std_scale=std_scale))
        else:
            parts.append(cm.refractor_block(s, allow_negative=i == 0, std_scale=std_scale))
        parts.append(cm.gap_block(s, last=i == len(optical) - 1, effl=effl, std_scale=std_scale))
    parts.append(cm.sensor_block(effl, theta))
    return "\n\n".join(parts) + "\n"


def main(argv: list[str]) -> int:
    return cm.cli(convert, argv, doc=__doc__ or "")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
