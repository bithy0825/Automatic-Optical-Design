"""len2toml — 将 OSLO .len 处方转换为本项目的训练配置 TOML。

用法:
    python scripts/len2toml.py <file.len> [out.toml]

接口与 zmx2toml 一致:成功打印或保存 TOML,过滤则打印 ``SKIP <文件名>: <原因>``、
退出码 1。设计见 docs/superpowers/specs/2026-08-06-zmx2toml-design.md。

.len 要点(实测 ZEBASE-OSLO 两库):NXT 分块;RD 为半径(曲率=1/RD,缺省=平面);
TH 厚度;AP 半口径;AST 行为光阑;头行 ``LEN NEW "名" <effl> <面数>`` 直取焦距;
EBR 入射光束半径定 F 数;UNI 1.0 = cm(×10 转 mm);GLA 可为名字或数值折射率;
材料一律 sellmeier 随机(只借结构)。
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zmx2toml as z2t

Skip = z2t.Skip
Surface = z2t.Surface

_UNIT_SCALE = 10.0  # UNI 1.0 = cm → mm

# 面块内可识别的关键字;其余一律视为不可表达而过滤
_KNOWN_SURFACE_KEYS = {
    "AIR", "GLA", "RD", "TH", "AP", "AST", "CC",
    "WV", "WV2", "WV3", "WW", "END", "PY",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 解析
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_block(s: Surface, key: str, tokens: list[str]) -> None:
    try:
        match key:
            case "AIR":
                s.glass = None
            case "GLA":
                name = tokens[1]
                s.glass = name
                try:
                    s.nd = float(name)  # 数值形式:GLA 1.573 1.573 1.573
                except ValueError:
                    s.nd = 0.0  # 目录名:折射率待库查(仅 ynu 后备用)
            case "RD":
                rd = float(tokens[1])
                if abs(rd) < 1e-9:
                    raise Skip("zero radius of curvature")
                s.curv = 1.0 / (rd * _UNIT_SCALE)
            case "TH":
                s.disz = float(tokens[1]) * _UNIT_SCALE
            case "AP":
                s.diam = float(tokens[1]) * _UNIT_SCALE
            case "CC":
                s.coni = float(tokens[1])  # OSLO 圆锥常数,约定与 zmx CONI 相同
            case "AST":
                s.is_stop = True
    except (IndexError, ValueError):
        pass


def parse_len(path: Path) -> tuple[list[Surface], float | None, float | None, float | None, float | None]:
    """返回 (光学面列表, 头行 EFFL[mm], EBR[mm], ANG[度], 像面 AP[mm])。"""
    text = z2t._read_text(path)
    m = re.search(r'^UNI\s+([\d.eE+-]+)', text, re.M)
    if not m or abs(float(m.group(1)) - 1.0) > 1e-9:
        raise Skip("unknown unit (UNI != 1.0)")
    m = re.search(r'^LEN\s+NEW\s+"[^"]*"\s+([\d.eE+-]+)', text, re.M)
    hdr_effl = float(m.group(1)) * _UNIT_SCALE if m else None
    m = re.search(r'^EBR\s+([\d.eE+-]+)', text, re.M)
    ebr = float(m.group(1)) * _UNIT_SCALE if m else None
    m = re.search(r'^ANG\s+([\d.eE+-]+)', text, re.M)
    ang = float(m.group(1)) if m else None

    blocks = re.split(r"^NXT\s*$", text, flags=re.M)
    if len(blocks) < 3:
        raise Skip("no optical surfaces")
    image_ap: float | None = None

    surfaces: list[Surface] = []
    for i, block in enumerate(blocks[1:], start=1):
        s = Surface(index=i)
        for line in block.splitlines():
            if not line.strip():
                continue
            tokens = line.split()
            key = tokens[0].upper()
            if key == "END":
                continue
            if key not in _KNOWN_SURFACE_KEYS:
                raise Skip(f"unsupported surface data: {key} (block {i})")
            _parse_block(s, key, tokens)
        surfaces.append(s)

    # 末块为像面:取其 AP 作 FOV 后备,不进入光学面列表
    if surfaces:
        image_ap = surfaces[-1].diam if surfaces[-1].diam > 0 else None
        optical = surfaces[:-1]
    else:
        optical = []
    if not optical:
        raise Skip("no optical surfaces")
    return optical, hdr_effl, ebr, ang, image_ap


# ═══════════════════════════════════════════════════════════════════════════════
# 推断
# ═══════════════════════════════════════════════════════════════════════════════


def _infer_effl(optical: list[Surface], hdr_effl: float | None) -> float:
    if hdr_effl is not None and 0.1 < hdr_effl < 1e5:
        return hdr_effl
    effl = z2t._ynu_effl(optical)  # 头行缺失/异常时的后备
    if effl is None or not 0.1 < effl < 1e5:
        raise Skip("cannot infer EFFL")
    return effl


def _infer_fnumber(effl: float, ebr: float | None) -> float:
    if ebr is None or ebr <= 0:
        raise Skip("cannot infer F-number (no EBR)")
    f = effl / (2.0 * ebr)
    if not 0.1 < f < 1e3:
        raise Skip("cannot infer F-number")
    return f


def _infer_fov(effl: float, ang: float | None, image_ap: float | None) -> float:
    if ang is not None and 0 < ang < 90:
        return ang
    if image_ap is not None:  # 像面 AP = 像圈半径
        return math.degrees(math.atan(image_ap / effl))
    return z2t._auto_fov(effl)  # 无视场信息:按焦距自动推断


# ═══════════════════════════════════════════════════════════════════════════════
# 转换主流程
# ═══════════════════════════════════════════════════════════════════════════════


def convert(path: Path) -> str:
    optical, hdr_effl, ebr, ang, image_ap = parse_len(path)

    for s in optical:
        if s.glass and s.glass.upper() == "MIRROR":
            raise Skip(f"mirror surface (surf {s.index})")
        if s.disz is None:
            s.disz = 0.0  # .len 中 TH 缺省 = 零厚度(哑面)
        elif s.disz > 1e6:
            raise Skip(f"infinite thickness (surf {s.index})")
        if s.diam <= 0:  # AP 缺失:以 2×入瞳直径兜底全直径,且不越半球域
            if ebr is None or ebr <= 0:
                raise Skip(f"missing aperture (surf {s.index})")
            s.diam = 2.0 * ebr
            if s.curv != 0.0:
                cap = math.floor(1.8 / abs(s.curv))  # 0.9×半球域极限(全直径)
                if math.ceil(2.0 * s.diam) > cap:
                    s.diam = max(cap, 1) / 2.0

    effl = z2t._snap_effl(_infer_effl(optical, hdr_effl))
    fnum = _infer_fnumber(effl, ebr)
    theta = _infer_fov(effl, ang, image_ap)

    parts = [z2t._header(path.stem, effl, fnum, theta)]
    seen_refractor = False
    for i, s in enumerate(optical):
        is_last = i == len(optical) - 1
        if s.is_stop and s.curv == 0.0:
            parts.append(z2t._stop(s))
        elif s.glass is not None or s.curv != 0.0:
            parts.append(z2t._refractor(s, allow_negative=not seen_refractor))
            seen_refractor = True
        # 平面空气哑面(无 AST):删面,gap 照出(厚度为 0 时整块跳过)
        if is_last or (s.disz or 0.0) > 0 or s.glass is not None or s.curv != 0.0 or s.is_stop:
            parts.append(z2t._gap(s, last=is_last, effl=effl))
    parts.append(z2t._sensor(effl, theta))
    return "\n\n".join(parts) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    path = Path(argv[1])
    try:
        toml = convert(path)
    except Skip as e:
        print(f"SKIP {path.name}: {e}")
        return 1
    if len(argv) == 3:
        out = Path(argv[2])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(toml, encoding="utf-8")
        print(out)
    else:
        print(toml)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
