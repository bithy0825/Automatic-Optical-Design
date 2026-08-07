"""zmx2toml — 将 Zemax .zmx 处方转换为本项目的训练配置 TOML。

用法:
    python scripts/zmx2toml.py <file.zmx> [out.toml]

成功:不给 out.toml 则打印到 stdout;给出则将 TOML 写入该路径(父目录不存在
自动创建)并打印保存路径。被过滤:打印 ``SKIP <文件名>: <原因>``,退出码 1,
不落盘、不建目录。设计见 docs/superpowers/specs/2026-08-06-zmx2toml-design.md。

仅支持:STANDARD(球面/圆锥面)、EVENASPH(偶次非球面)、平面 STOP、像面;
EFFL / F 数 / 视场角按推断链解析,材料一律 sellmeier 随机(只借 zmx 的结构)。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Skip(Exception):
    """过滤:携带可读原因。"""


# ═══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Surface:
    index: int
    type: str = "STANDARD"
    curv: float = 0.0
    disz: float | None = None  # None = INFINITY
    diam: float = 0.0  # 半直径(zmx DIAM 原值)
    glass: str | None = None
    nd: float = 0.0
    vd: float = 0.0
    coni: float = 0.0
    parms: dict[int, float] = field(default_factory=dict)
    is_stop: bool = False


@dataclass
class Zmx:
    fnum: float | None = None
    enpd: float | None = None
    obna: float | None = None
    ftyp: int = 0
    fields: list[float] = field(default_factory=list)  # XFLN/XFLD/YFLN/YFLD 汇总
    merit_effl: float | None = None
    unit: str = "MM"
    mnum: int = 1  # MNUM:多重结构的配置数(>1 = 变焦/多位置,过滤)
    surfaces: list[Surface] = field(default_factory=list)  # 含 OBJ(0) 与 IMA(末)


# ═══════════════════════════════════════════════════════════════════════════════
# 解析
# ═══════════════════════════════════════════════════════════════════════════════


def _float(tok: str) -> float:
    return float(tok)


def _parse_surface_field(s: Surface, key: str, tokens: list[str]) -> None:
    try:
        match key:
            case "TYPE":
                s.type = tokens[1].upper()
            case "CURV":
                s.curv = _float(tokens[1])
            case "DISZ":
                s.disz = (
                    None if tokens[1].upper().startswith("INF") else _float(tokens[1])
                )
            case "DIAM":
                s.diam = _float(tokens[1])
            case "CONI":
                s.coni = _float(tokens[1])
            case "PARM":
                s.parms[int(tokens[1])] = _float(tokens[2])
            case "STOP":
                s.is_stop = True
            case "GLAS":
                s.glass = tokens[1]
                if len(tokens) > 4:
                    s.nd = _float(tokens[4])
                if len(tokens) > 5:
                    s.vd = _float(tokens[5])
    except (ValueError, IndexError):
        pass  # 未知/残缺字段:忽略,由过滤与推断阶段把关


def _parse_header(zmx: Zmx, key: str, tokens: list[str]) -> None:
    try:
        match key:
            case "FNUM":
                zmx.fnum = _float(tokens[1])
            case "ENPD":
                zmx.enpd = _float(tokens[1])
            case "OBNA":
                zmx.obna = _float(tokens[1])
            case "FTYP":
                zmx.ftyp = int(_float(tokens[1]))
            case "UNIT":
                zmx.unit = tokens[1].upper()
            case "MNUM":
                zmx.mnum = int(_float(tokens[1]))
            case "XFLN" | "XFLD" | "YFLN" | "YFLD":
                zmx.fields.extend(_float(t) for t in tokens[1:])
            case "EFFL":  # 评价函数 EFFL 操作数:第 7 个数值为目标值
                vals = [_float(t) for t in tokens[1:]]
                if len(vals) >= 7:
                    zmx.merit_effl = vals[6]
    except (ValueError, IndexError):
        pass


def _read_text(path: Path) -> str:
    """读取 zmx/len 文本:UTF-16(带 BOM)与 UTF-8/ASCII 自适应。"""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def parse_zmx(path: Path) -> Zmx:
    zmx = Zmx()
    cur: Surface | None = None
    for raw in _read_text(path).splitlines():
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


# ═══════════════════════════════════════════════════════════════════════════════
# 推断链
# ═══════════════════════════════════════════════════════════════════════════════

_DB = None  # sellmeier 单例(懒加载,仅用于 EFFL 计算,不做材料初始化)


def _db_nd(name: str) -> float | None:
    """库中同名(或 N- 前缀)玻璃的 d 线折射率。"""
    global _DB
    if _DB is None:
        from materials.sellmeier import SellmeierMaterialDatabase

        _DB = SellmeierMaterialDatabase.create()
    for cand in (name, f"N-{name}"):
        if cand in _DB:
            return float(_DB.nd[_DB.index_of(cand)])
    return None


def _ynu_effl(optical: list[Surface]) -> float | None:
    """一阶 ynu 近轴追迹 EFFL;任一面折射率不明或系统不会聚则返回 None。"""
    y, n_u, n_prev = 1.0, 0.0, 1.0
    for i, s in enumerate(optical):
        if s.glass is None:
            n_next = 1.0
        else:
            n_next = s.nd if s.nd > 1.0 else _db_nd(s.glass)
            if n_next is None:
                return None
        n_u -= y * (n_next - n_prev) * s.curv
        if i + 1 < len(optical):
            y += (s.disz or 0.0) * n_u / n_next
        n_prev = n_next
    u_final = n_u / n_prev
    if u_final >= 0:  # 末光线必须向轴会聚
        return None
    return -1.0 / u_final


def infer_effl(zmx: Zmx, optical: list[Surface]) -> float:
    effl = _ynu_effl(optical)
    if effl is None:
        effl = zmx.merit_effl
    if effl is None or not 0.1 < effl < 1e5:
        raise Skip("cannot infer EFFL")
    return effl


def _snap_effl(effl: float) -> float:
    """设计目标吸附:与最近整数相差 <1% 时取整。

    计算值是处方的实际近轴焦距(含制造/换算噪声),优化目标应是设计意图
    ——59.98 → 60、100.04 → 100;12.5、57.3 这类本真非整值保持不变。
    """
    nearest = round(effl)
    if nearest > 0 and abs(effl - nearest) / effl < 0.01:
        return float(nearest)
    return effl


def infer_fnumber(zmx: Zmx, optical: list[Surface], effl: float) -> float:
    if zmx.fnum and zmx.fnum > 0:
        f = zmx.fnum
    elif zmx.enpd and zmx.enpd > 0:
        f = effl / zmx.enpd
    elif zmx.obna and zmx.obna > 0:
        f = 1.0 / (2.0 * zmx.obna)
    else:  # 光阑尺寸浮动:EPD ≈ 光阑全直径
        stop = next((s for s in optical if s.is_stop and s.diam > 0), None)
        if stop is None:
            raise Skip("cannot infer F-number")
        f = effl / (2.0 * stop.diam)
    if not 0.1 < f < 1e3:
        raise Skip("cannot infer F-number")
    return f


def _auto_fov(effl: float) -> float:
    """视场缺失/失效时自动推断:全画幅半对角线 21.6mm 按焦距缩放,钳 [3°, 45°]。"""
    return min(45.0, max(3.0, math.degrees(math.atan(21.6 / effl))))


def infer_fov(zmx: Zmx, effl: float) -> float:
    """最大半视场角(度);文件无视场信息或推断越界时按焦距自动推断。"""
    hmax = max((abs(v) for v in zmx.fields), default=0.0)
    theta = 0.0
    if hmax > 0:
        if zmx.ftyp == 0:  # 角度(度)
            theta = hmax
        elif zmx.ftyp == 1:  # 物高 → 按物距换算
            obj = zmx.surfaces[0].disz
            if obj and obj > 0:
                theta = math.degrees(math.atan(hmax / obj))
        else:  # 2/3:像高 → 按焦距换算
            theta = math.degrees(math.atan(hmax / effl))
    if not 0 < theta < 90:
        theta = _auto_fov(effl)
    return theta


# ═══════════════════════════════════════════════════════════════════════════════
# TOML 生成
# ═══════════════════════════════════════════════════════════════════════════════

_SENSOR_DIAGONALS = (7.7, 9.5, 16.0, 21.6, 28.3, 34.5, 43.3, 55.0, 79.2, 87.3)

_UNIT_SCALE = {"MM": 1.0, "CM": 10.0, "IN": 25.4, "METER": 1000.0}  # → mm


def _f(x: float) -> str:
    """TOML 浮点字面量(保证带小数点或指数)。"""
    s = f"{x:.6g}"
    return s if any(c in s for c in ".eE") else s + ".0"


def _bounds(m: float, lo: float, hi: float) -> str:
    """固定走廊 [lo, hi];若 mean 落在走廊外,按 50% 抖动扩张以包含它。"""
    lo = min(lo, m - 0.5 * abs(m))
    hi = max(hi, m + 0.5 * abs(m))
    return f"[{_f(round(lo, 3))}, {_f(round(hi, 3))}]"


def _alpha_bounds(s: Surface, D: int, jmax: int) -> str:
    """单区间覆盖全部系数(边界损失对 Alpha 逐元素施加同一区间)。"""
    rho = D / 2
    worst = max(
        (abs(s.parms.get(j, 0.0) * rho ** (2 * j)) for j in range(2, jmax + 1)),
        default=0.0,
    )
    b = round(max(0.2, 1.5 * worst), 3)
    return f"[{_f(-b)}, {_f(b)}]"


def _init_std(mean: float, floor: float) -> float:
    """初始化 std:3σ ≈ ±100%(= |mean|/3);mean≈0 时取绝对下限 floor。"""
    return max(abs(mean) / 3.0, floor)


def _domain_cmax(s: Surface, D: int) -> float | None:
    """半球域曲率上限 1/(√(1+κ)·半口径);1+κ≤0(双曲)无域限返回 None。"""
    if 1.0 + s.coni <= 0.0:
        return None
    return 1.0 / (math.sqrt(1.0 + s.coni) * (D / 2))


def _header(stem: str, effl: float, fnum: float, theta: float) -> str:
    return f'''[target]
fov = [[0, {_f(round(theta, 3))}], [0, 0]]
F = {_f(round(fnum, 3))}
effl = {_f(round(effl, 3))}
wavelength = [486, 589, 656]

[ga]
population = 256
topk = 64
generation = 120

[[optimizer]]
type = "adam"
step = 200
scheduler = "cosine"
grad_norm = 10.0
lr = {{ curvature = 2e-4, thickness = 5e-2, diameter = 2e-4, kappa = 1e-3, alpha = 1e-3 }}

[loss]
effl = 0.01
blur = 1.0
distortion = 1.0
thickness = 10.0
curvature = 10.0

[train]
device = "auto"
output = "{stem}.pth"
save_every = 30
history = "{stem}.json"

[[component]]
type = "source"
pupil = {{ method = "fibonacci", region = "disk", count = 256 }}
field = {{ method = "uniform", region = "rect", count = [3, 1] }}
wavel = {{ method = "uniform", region = "line", count = 3 }}

[[component]]
type = "gap"
thickness = {{ method = "raw", value = 0.0 }}'''


def _diameter_of(s: Surface) -> int:
    """机械全直径:ceil(2×半直径)。"""
    return math.ceil(2 * s.diam)


def _diameter_bounds(D: int) -> str:
    return f"[{_f(round(D * 0.5, 1))}, {_f(round(D * 1.5, 1))}]"


def _material_of(s: Surface) -> str:
    if s.glass is None:
        return 'material = { method = "raw", value = "air", db = "constant" }'
    return 'material = { method = "random", db = "sellmeier" }'


def _shape_of(s: Surface) -> tuple[str, int]:
    """(面型, 最高非零 PARM 阶)。PARM j = r^{2j} 系数 → alpha{2j}。"""
    jmax = max((j for j, v in s.parms.items() if j >= 2 and v != 0.0), default=0)
    if jmax:
        return "asphere", jmax
    return ("conic", 0) if s.coni != 0.0 else ("sphere", 0)


def _refractor(s: Surface, *, allow_negative: bool) -> str:
    D = _diameter_of(s)
    shape, jmax = _shape_of(s)
    c = round(s.curv, 3)

    d_std = _init_std(D, 0.1)
    c_std = _init_std(c, 0.01)
    cmax = _domain_cmax(s, D)
    if cmax is not None:
        c_std = min(c_std, max(0.005, (0.9 * cmax - abs(c)) / 3.0))
    k_std = _init_std(s.coni, 0.05) if shape != "sphere" else 0.0

    lines = ["[[component]]", 'type = "refractor"', f'shape = "{shape}"']
    if allow_negative:
        lines.append("solver = { allow_negative = true }")
    lines.append(
        f'diameter = {{ method = "normal", mean = {_f(D)}, std = {_f(d_std)} }}'
    )
    lines.append(
        f'curvature = {{ method = "normal", mean = {_f(c)}, std = {_f(c_std)} }}'
    )
    if shape != "sphere":
        lines.append(
            f'kappa = {{ method = "normal", mean = {_f(s.coni)}, std = {_f(k_std)} }}'
        )
    a_std_worst = 0.0
    if shape == "asphere":  # α = A_j × (D/2)^{2j}(归一化到机械口径)
        rho = D / 2
        for j in range(2, jmax + 1):
            a = s.parms.get(j, 0.0) * rho ** (2 * j)
            a_std = _init_std(a, 0.001)
            a_std_worst = max(a_std_worst, a_std)
            lines.append(
                f'alpha{2 * j} = {{ method = "normal", mean = {_f(a)}, std = {_f(a_std)} }}'
            )
    lines.append(_material_of(s))

    train = ["curvature"]
    mutate = [f"curvature = {_f(c_std / 2)}"]
    bounds = [f"curvature = {_bounds(c, -0.1, 0.1)}"]
    if shape != "sphere":
        train.append("kappa")
        mutate.append(f"kappa = {_f(k_std / 2)}")
        bounds.append(f"kappa = {_bounds(s.coni, -5.0, 5.0)}")
    if shape == "asphere":
        train.append("alpha")
        mutate.append(f"alpha = {_f(a_std_worst / 2)}")
        bounds.append(f"alpha = {_alpha_bounds(s, D, jmax)}")
    train.append("diameter")
    mutate.append(f"diameter = {_f(min(d_std / 2, 1.0))}")
    if s.glass is not None:
        mutate.append("material = 2.0")
    bounds.append(f"diameter = {_diameter_bounds(D)}")
    lines.append(f"train = {{ {', '.join(f'{k} = true' for k in train)} }}")
    lines.append(f"mutate = {{ {', '.join(mutate)} }}")
    lines.append(f"bounds = {{ {', '.join(bounds)} }}")
    return "\n".join(lines)


def _stop(s: Surface, *, front: bool = False) -> str:
    D = _diameter_of(s)
    std = _init_std(D, 0.1)
    lines = ["[[component]]", 'type = "stop"']
    if front:  # 前置光阑与光源同面,t≈0 的数值噪声需允许负距离解
        lines.append("solver = { allow_negative = true }")
    lines += [
        f'diameter = {{ method = "normal", mean = {_f(D)}, std = {_f(std)} }}',
        "train = { diameter = true }",
        f"mutate = {{ diameter = {_f(min(std / 2, 1.0))} }}",
        f"bounds = {{ diameter = {_diameter_bounds(D)} }}",
    ]
    return "\n".join(lines)


def _gap(s: Surface, *, last: bool, effl: float) -> str:
    t = round(s.disz or 0.0, 3)
    if last:
        hi = float(math.ceil(1.7 * effl))
    elif s.glass is not None:
        hi = 15.0
    else:
        hi = 20.0
    lo = 0.5 if t >= 0.5 else round(0.5 * t, 3)  # 超薄层按 50% 抖动压低下限
    if t > hi:
        hi = round(1.5 * t, 3)  # 超厚层按 50% 抖动顶出上限
    std = _init_std(t, 0.1)
    return "\n".join(
        [
            "[[component]]",
            'type = "gap"',
            f'thickness = {{ method = "normal", mean = {_f(t)}, std = {_f(std)} }}',
            "train = { thickness = true }",
            f"mutate = {{ thickness = {_f(std / 2)} }}",
            f"bounds = {{ thickness = [{_f(lo)}, {_f(hi)}] }}",
        ]
    )


def _sensor(effl: float, theta: float) -> str:
    need = 2 * effl * math.tan(math.radians(theta))
    diameter = next(
        (d for d in _SENSOR_DIAGONALS if d >= need),
        float(2 * math.ceil(need / 2)),  # 无匹配:向上取整到偶数
    )
    return "\n".join(
        [
            "[[component]]",
            'type = "sensor"',
            f'diameter = {{ method = "raw", value = {_f(diameter)} }}',
        ]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 转换主流程
# ═══════════════════════════════════════════════════════════════════════════════


def convert(path: Path) -> str:
    zmx = parse_zmx(path)
    scale = _UNIT_SCALE.get(zmx.unit)
    if scale is None:
        raise Skip(f"unknown unit: {zmx.unit}")
    if scale != 1.0:  # 统一折算成 mm(曲率除,长度乘;视场值按 FTYP 语义)
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
        if s.disz is None:
            raise Skip(f"infinite thickness (surf {s.index})")
        if s.diam <= 0:  # 自动口径:以 2×EPD 兜底全直径,且不越半球域
            if zmx.enpd is None or zmx.enpd <= 0:
                raise Skip(f"zero diameter (surf {s.index})")
            s.diam = zmx.enpd
            if s.curv != 0.0:
                cap = math.floor(1.8 / abs(s.curv))
                if math.ceil(2.0 * s.diam) > cap:
                    s.diam = max(cap, 1) / 2.0
        if s.parms.get(1, 0.0) != 0.0:
            s.parms = {}  # r² 项无处安放:全部偶次项置 0,退化为 conic/sphere

    effl = _snap_effl(infer_effl(zmx, optical))
    fnum = infer_fnumber(zmx, optical, effl)
    theta = infer_fov(zmx, effl)

    parts = [_header(path.stem, effl, fnum, theta)]
    # allow_negative 只给链上第一个面(与光源同面,t≈0 的数值噪声/首面负曲率
    # 的合法负根);中间面负距离=X 型打架,必须保持判死
    for i, s in enumerate(optical):
        if s.is_stop and s.curv == 0.0:
            parts.append(_stop(s, front=i == 0))
        else:
            parts.append(_refractor(s, allow_negative=i == 0))
        parts.append(_gap(s, last=i == len(optical) - 1, effl=effl))
    parts.append(_sensor(effl, theta))
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
