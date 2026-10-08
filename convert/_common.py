"""处方转换的共享机械：面数据结构、过滤异常、推断链与 TOML 发射器。

介质约定：材料只允许玻璃与空气——玻璃一律 sellmeier 随机（只借处方的
结构），含浸液等非玻璃介质（水/油）的处方直接过滤。两侧同为空气的面
不参与折射：平面判为光阑（多光阑，保留口径裁切），曲面判无效删除并
折并厚度；空气中的光阑不论曲率一律转平面光阑，携带玻璃或前承玻璃的
曲面光阑则退化为折射面，以免丢失光焦度。
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


class Skip(Exception):
    """过滤：携带可读原因。"""


@dataclass(slots=True)
class Surface:
    """处方中的一个光学面（解析器逐字段填充，故可变）。"""

    index: int
    type: str = "STANDARD"
    curv: float = 0.0
    disz: float | None = None  # None = INFINITY
    diam: float = 0.0  # 半直径（zmx DIAM 原值）
    glass: str | None = None
    nd: float = 0.0
    vd: float = 0.0
    coni: float = 0.0
    parms: dict[int, float] = field(default_factory=dict)
    is_stop: bool = False


def read_text(path: Path) -> str:
    """读取 zmx/len 文本：UTF-16（带 BOM）与 UTF-8/ASCII 自适应。"""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


# ── 推断链 ──

_DB = None  # sellmeier 单例（懒加载，仅用于 EFFL 计算，不做材料初始化）

# 非玻璃介质（浸液等）：本框架材料只支持玻璃与空气，含之即过滤。
NON_GLASS = {"WATER"}

# Zemax 模型玻璃占位 (nd, vd)：处方转换器（如 Cambridge/Imaging Optics 系列）
# 给目录玻璃填的默认值；命中时库值优先于内联值。
_PLACEHOLDER_NV = (1.5, 40.0)


def db_nd(name: str) -> float | None:
    """库中同名（或 N- 前缀）玻璃的 d 线折射率。"""
    global _DB
    if _DB is None:
        from materials.sellmeier import SellmeierMaterialDatabase

        _DB = SellmeierMaterialDatabase.create()
    for cand in (name, f"N-{name}"):
        if cand in _DB:
            return float(_DB.nd[_DB.index_of(cand)])
    return None


def _medium_nd(s: Surface, *, db_first: bool) -> float | None:
    """表面后介质的 d 线折射率；空气 1.0，未知返回 None。

    内联 nd 是专利处方的真实数据，优先；*db_first* 时占位 (1.5, 40) 的
    模型玻璃以库值为准（转换器填的默认值不如目录可信）。
    """
    if s.glass is None:
        return 1.0
    inline = s.nd if s.nd > 1.0 else None
    if db_first and (s.nd, s.vd) == _PLACEHOLDER_NV:
        return db_nd(s.glass) or inline
    return inline or db_nd(s.glass)


def ynu_effl(optical: list[Surface]) -> float | None:
    """一阶 ynu 近轴追迹 EFFL；任一面折射率不明或系统不会聚则返回 None。

    占位符玻璃全或无地换用库值：只换可解析的子集会造成折射率体系不一致
    （部分 1.5 部分真值），追迹反而失真——任一占位玻璃不可解析则全部用内联值。
    """
    ph: list[str] = []
    for s in optical:
        if s.glass and (s.nd, s.vd) == _PLACEHOLDER_NV:
            ph.append(s.glass)
    db_first = bool(ph) and all(db_nd(g) is not None for g in ph)
    y, n_u, n_prev = 1.0, 0.0, 1.0
    for i, s in enumerate(optical):
        n_next = _medium_nd(s, db_first=db_first)
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


def snap_effl(effl: float) -> float:
    """设计目标吸附：与最近整数相差 <1% 时取整。

    计算值是处方的实际近轴焦距（含制造/换算噪声），优化目标应是设计意图
    ——59.98 → 60、100.04 → 100；12.5、57.3 这类本真非整值保持不变。
    """
    nearest = round(effl)
    if nearest > 0 and abs(effl - nearest) / effl < 0.01:
        return float(nearest)
    return effl


def auto_fov(effl: float) -> float:
    """视场缺失/失效时自动推断：半对角线 9.5mm 按焦距缩放，钳 [3°, 45°]。"""
    return min(45.0, max(3.0, math.degrees(math.atan(9.5 / effl))))


# ── TOML 发射器 ──

_SENSOR_DIAGONALS = (7.7, 9.5, 16.0, 21.6, 28.3, 34.5, 43.3, 55.0, 79.2, 87.3)


def fmt_float(x: float) -> str:
    """TOML 浮点字面量（保证带小数点或指数）。"""
    s = f"{x:.6g}"
    return s if any(c in s for c in ".eE") else s + ".0"


def _bounds(m: float, lo: float, hi: float) -> str:
    """固定走廊 [lo, hi]；若 mean 落在走廊外，按 50% 抖动扩张以包含它。"""
    lo = min(lo, m - 0.5 * abs(m))
    hi = max(hi, m + 0.5 * abs(m))
    return f"[{fmt_float(round(lo, 3))}, {fmt_float(round(hi, 3))}]"


def _alpha_bounds(s: Surface, D: int, jmax: int) -> str:
    """单区间覆盖全部系数（边界损失对 Alpha 逐元素施加同一区间）。"""
    rho = D / 2
    worst = max(
        (abs(s.parms.get(j, 0.0) * rho ** (2 * j)) for j in range(2, jmax + 1)),
        default=0.0,
    )
    b = round(max(0.2, 1.5 * worst), 3)
    return f"[{fmt_float(-b)}, {fmt_float(b)}]"


def fov_std_scale(theta: float) -> float:
    """广角设计初始化 std 倍率：角度越大越收紧（越接近原始处方结构）。

    倍率 3→2→1 对应 <25°/25°–35°/35°–45°，scale = 倍率/3。
    """
    if theta >= 35.0:
        return 1.0 / 3.0
    if theta >= 25.0:
        return 2.0 / 3.0
    return 1.0


def _init_std(mean: float, floor: float, scale: float = 1.0) -> float:
    """初始化 std：3σ ≈ ±100%（= |mean|/3）；mean≈0 时取绝对下限 floor。

    scale > 1 加宽分布（广角设计探索增强）。
    """
    return max(abs(mean) / 3.0 * scale, floor)


def _domain_cmax(s: Surface, D: int) -> float | None:
    """半球域曲率上限 1/(√(1+κ)·半口径)；1+κ≤0（双曲）无域限返回 None。"""
    if 1.0 + s.coni <= 0.0:
        return None
    return 1.0 / (math.sqrt(1.0 + s.coni) * (D / 2))


def header(stem: str, effl: float, fnum: float, theta: float) -> str:
    """训练配置头部（target / ga / optimizer / loss / train / source / 首 gap）。"""
    return f'''[target]
fov = [[0, {fmt_float(round(theta, 3))}], [0, 0]]
F = {fmt_float(round(fnum, 3))}
effl = {fmt_float(round(effl, 3))}
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
    """机械全直径：ceil(2×半直径)。"""
    return math.ceil(2 * s.diam)


def _diameter_bounds(D: int) -> str:
    return f"[{fmt_float(round(D * 0.5, 1))}, {fmt_float(round(D * 1.5, 1))}]"


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


def refractor_block(s: Surface, *, allow_negative: bool, std_scale: float = 1.0) -> str:
    """折射器元件块：面形参数初始化 + train/mutate/bounds 三件套。"""
    D = _diameter_of(s)
    shape, jmax = _shape_of(s)
    c = round(s.curv, 3)

    d_std = _init_std(D, 0.1, std_scale)
    c_std = _init_std(c, 0.01, std_scale)
    cmax = _domain_cmax(s, D)
    if cmax is not None:
        # 半球域上限：大口径面（长焦/航拍）若放任 0.005 下限与 ±0.1 走廊，
        # 初始化与变异会采样到 c·ρ>1 的几何不可能区（矢高域全灭）。
        # std 下限改为相对域限；走廊收进 0.95·cmax。
        c_std = min(c_std, max(0.05 * cmax, (0.9 * cmax - abs(c)) / 3.0))
    k_std = _init_std(s.coni, 0.05, std_scale) if shape != "sphere" else 0.0

    lines = ["[[component]]", 'type = "refractor"', f'shape = "{shape}"']
    if allow_negative:
        lines.append("solver = { allow_negative = true }")
    lines.append(
        f'diameter = {{ method = "normal", mean = {fmt_float(D)}, std = {fmt_float(d_std)} }}'
    )
    lines.append(
        f'curvature = {{ method = "normal", mean = {fmt_float(c)}, std = {fmt_float(c_std)} }}'
    )
    if shape != "sphere":
        lines.append(
            f'kappa = {{ method = "normal", mean = {fmt_float(s.coni)}, std = {fmt_float(k_std)} }}'
        )
    a_std_worst = 0.0
    if shape == "asphere":  # α = A_j × (D/2)^{2j}（归一化到机械口径）
        rho = D / 2
        for j in range(2, jmax + 1):
            a = s.parms.get(j, 0.0) * rho ** (2 * j)
            a_std = _init_std(a, 0.001, std_scale)
            a_std_worst = max(a_std_worst, a_std)
            lines.append(
                f'alpha{2 * j} = {{ method = "normal", mean = {fmt_float(a)}, std = {fmt_float(a_std)} }}'
            )
    lines.append(_material_of(s))

    train = ["curvature"]
    mutate = [f"curvature = {fmt_float(c_std / 2)}"]
    clo, chi = -0.1, 0.1
    if cmax is not None:
        clo, chi = max(clo, -0.95 * cmax), min(chi, 0.95 * cmax)
    bounds = [f"curvature = {_bounds(c, clo, chi)}"]
    if shape != "sphere":
        train.append("kappa")
        mutate.append(f"kappa = {fmt_float(k_std / 2)}")
        bounds.append(f"kappa = {_bounds(s.coni, -5.0, 5.0)}")
    if shape == "asphere":
        train.append("alpha")
        mutate.append(f"alpha = {fmt_float(a_std_worst / 2)}")
        bounds.append(f"alpha = {_alpha_bounds(s, D, jmax)}")
    train.append("diameter")
    mutate.append(f"diameter = {fmt_float(min(d_std / 2, 1.0))}")
    if s.glass is not None:
        mutate.append("material = 2.0")
    bounds.append(f"diameter = {_diameter_bounds(D)}")
    lines.append(f"train = {{ {', '.join(f'{k} = true' for k in train)} }}")
    lines.append(f"mutate = {{ {', '.join(mutate)} }}")
    lines.append(f"bounds = {{ {', '.join(bounds)} }}")
    return "\n".join(lines)


def stop_block(s: Surface, *, front: bool = False, std_scale: float = 1.0) -> str:
    """光阑元件块。"""
    D = _diameter_of(s)
    std = _init_std(D, 0.1, std_scale)
    lines = ["[[component]]", 'type = "stop"']
    if front:  # 前置光阑与光源同面，t≈0 的数值噪声需允许负距离解
        lines.append("solver = { allow_negative = true }")
    lines += [
        f'diameter = {{ method = "normal", mean = {fmt_float(D)}, std = {fmt_float(std)} }}',
        "train = { diameter = true }",
        f"mutate = {{ diameter = {fmt_float(min(std / 2, 1.0))} }}",
        f"bounds = {{ diameter = {_diameter_bounds(D)} }}",
    ]
    return "\n".join(lines)


def gap_block(s: Surface, *, last: bool, effl: float, std_scale: float = 1.0) -> str:
    """间隔元件块。"""
    # 制造下限 0.1mm：过薄间隙的初始值与下界同时抬升——否则优化器会把
    # 薄间隙推向无意义的亚 0.1mm 结构（处方里这种值多为接触/胶合记法）
    t = max(round(s.disz or 0.0, 3), 0.1)
    if last:
        hi = float(math.ceil(1.7 * effl))
    elif s.glass is not None:
        hi = 15.0
    else:
        hi = 20.0
    lo = 0.5 if t >= 0.5 else 0.1
    if t > hi:
        hi = round(1.5 * t, 3)  # 超厚层按 50% 抖动顶出上限
    std = _init_std(t, 0.1, std_scale)
    return "\n".join(
        [
            "[[component]]",
            'type = "gap"',
            f'thickness = {{ method = "normal", mean = {fmt_float(t)}, std = {fmt_float(std)} }}',
            "train = { thickness = true }",
            f"mutate = {{ thickness = {fmt_float(std / 2)} }}",
            f"bounds = {{ thickness = [{fmt_float(lo)}, {fmt_float(hi)}] }}",
        ]
    )


def sensor_block(effl: float, theta: float) -> str:
    """传感器元件块：按视场像高匹配标准靶面直径。"""
    need = 2 * effl * math.tan(math.radians(theta))
    diameter = next(
        (d for d in _SENSOR_DIAGONALS if d >= need),
        float(2 * math.ceil(need / 2)),  # 无匹配：向上取整到偶数
    )
    return "\n".join(
        [
            "[[component]]",
            'type = "sensor"',
            f'diameter = {{ method = "raw", value = {fmt_float(diameter)} }}',
        ]
    )


# ── 空-空面三分法 ──


def is_flat(s: Surface) -> bool:
    """几何平面：无曲率且无非球面项（PARM 1 已折入 curv，余项 ≥2 即有矢高）。"""
    return s.curv == 0.0 and not s.parms


def drop_noop_surfaces(optical: list[Surface]) -> list[Surface]:
    """删除空-空无效面并折并厚度；返回保留面。

    前后介质同为空气的面不参与折射（同介质，任意曲率均无光焦度）：曲面判
    无效删除，其 DISZ 累进前一保留面的 DISZ；平面保留，由 convert 判为光阑
    （多光阑，裁切职责不丢）。首段无效面（光源与首面之间）的厚度**丢弃**：
    框架光源是无穷远共轭模型（视场角 + F 数定义光束），物方间距不属于镜头
    结构——折叠进去只会让轴外光束按 间距×tanθ 平移而错过首面（离轴全灭）。
    光阑面永不删除。
    """
    kept: list[Surface] = []
    i = 0
    while i < len(optical):
        s = optical[i]
        if s.is_stop or s.glass is not None or is_flat(s) or (kept and kept[-1].glass is not None):
            kept.append(s)
            i += 1
            continue
        j = i  # 连续无效面段 [i, j)（空-空曲面；平面与光阑在段外保留）
        while (
            j < len(optical)
            and not optical[j].is_stop
            and optical[j].glass is None
            and not is_flat(optical[j])
        ):
            j += 1
        if kept:
            kept[-1].disz = (kept[-1].disz or 0.0) + sum(s.disz or 0.0 for s in optical[i:j])
        # 首段：丢弃间距（见 docstring）
        i = j
    return kept


# ── CLI 驱动 ──


def cli(convert: Callable[[Path], str], argv: list[str], *, doc: str) -> int:
    """CLI 驱动：``<输入处方> [out.toml]``；过滤打印 ``SKIP`` 返回 1，
    参数个数错误打印用法返回 2；给出 out 时写文件（父目录自动创建）。"""
    if len(argv) not in (2, 3):
        print(doc)
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
