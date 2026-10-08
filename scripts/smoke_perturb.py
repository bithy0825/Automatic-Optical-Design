"""perturb 模块冒烟测试：位姿不变量 / 陈旧指纹 / 散射冻结 / 裁剪裁决。

用法：python scripts/smoke_perturb.py
"""

import torch

from component import Sequential
from perturb import Clip, Pose, Scatter, build_callback
from perturb.error import from_options

torch.set_default_dtype(torch.float64)

P = 4

SOURCE_BLOCK = {
    "type": "source",
    "epd": 8.0,
    "fov": 4.0,
    "wavelength": 555.0,
    "pupil": {"method": "uniform", "region": "disk", "count": [4, 8]},
    "field": {"method": "uniform", "region": "rect", "count": [3, 1]},
    "wavel": {"method": "uniform", "region": "line", "count": 1},
}
LENS_FRONT = {
    "type": "refractor",
    "shape": "sphere",
    "curvature": {"method": "raw", "value": 0.02},
    "diameter": {"method": "raw", "value": 10.0},
    "material": {"method": "raw", "value": "N-BK7"},
}
LENS_BACK = {
    "type": "refractor",
    "shape": "disk",
    "diameter": {"method": "raw", "value": 10.0},
    "material": {"method": "raw", "value": "air"},
}
GAP_5 = {"type": "gap", "thickness": {"method": "raw", "value": 5.0}}
GAP_92 = {"type": "gap", "thickness": {"method": "raw", "value": 92.0}}
SENSOR = {"type": "sensor", "diameter": {"method": "raw", "value": 5.0}}

# 元件索引: 0=source, 1=S1(球面), 2=gap5, 3=S2(平面), 4=gap92, 5=S3(sensor)
BLOCKS = [SOURCE_BLOCK, LENS_FRONT, GAP_5, LENS_BACK, GAP_92, SENSOR]


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        raise AssertionError(name)


def record_poses(seq: Sequential, *callbacks) -> list[torch.Tensor]:
    """追迹并记录逐站点 transformer 正向矩阵（回调追加在扰动回调之后）。"""
    mats: list[torch.Tensor] = []

    def rec(_comp, flow, _i):
        mats.append(flow.transformer.forward.clone())
        return flow

    seq(callback=[*callbacks, rec])
    return mats


def main() -> None:
    seq = Sequential.from_options(P, BLOCKS)
    nom = record_poses(seq)

    # ── 1. 单面倾斜：转轴过面顶点（顶点不平移），旋转 = Rx(α) ──
    cb = build_callback(seq, [Pose(first=1, last=1, alpha=0.01)])
    mats = record_poses(seq, cb)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, -0.01], [0.0, 0.01, 1.0]])  # Rx(0.01) 一阶近似
    rot_err = (mats[0][:, :3, :3] - rx.to(mats[0])).abs().max().item()
    check("tilt vertex fixed", mats[0][:, :3, 3].abs().max().item() < 1e-12)
    check("tilt is Rx", rot_err < 1e-3, f"err={rot_err:.2e}")
    check("tilt resets downstream", (mats[2] - nom[2]).abs().max().item() < 1e-12)

    # ── 2. 元件区间刚性：区间内两面相对位姿 = 名义相对位姿 ──
    cb = build_callback(seq, [Pose(first=1, last=2, dx=0.1, alpha=0.02)])
    mats = record_poses(seq, cb)
    rel_nom = torch.linalg.inv(nom[0]) @ nom[2]  # S1 顶点 → S2 顶点
    rel_pert = torch.linalg.inv(mats[0]) @ mats[2]
    check(
        "element interval rigid",
        (rel_nom - rel_pert).abs().max().item() < 1e-9,
        f"err={(rel_nom - rel_pert).abs().max().item():.2e}",
    )
    check("element decentered", (mats[0][:, 0, 3] - 0.1).abs().max().item() < 1e-12)
    check("interval resets after", (mats[4] - nom[4]).abs().max().item() < 1e-9)

    # ── 3. 间隔误差（last=None）：下游全体继承 dz ──
    cb = build_callback(seq, [Pose(first=2, dz=0.5)])
    mats = record_poses(seq, cb)
    check(
        "spacing shifts downstream",
        (mats[4][:, 2, 3] - 97.5).abs().max().item() < 1e-12,
        f"z={mats[4][0, 2, 3].item()}",
    )

    # ── 4. 逐个体误差：(P,) 张量逐个体生效 ──
    dx = torch.tensor([0.0, 0.05, 0.10, 0.15])
    cb = build_callback(seq, [Pose(first=1, last=1, dx=dx)])
    mats = record_poses(seq, cb)
    check(
        "per-individual dx",
        (mats[0][:, 0, 3] - dx).abs().max().item() < 1e-12,
        f"x={mats[0][:, 0, 3].tolist()}",
    )

    # ── 5. 陈旧指纹：构建后原地改 Gap 厚度 → fail-fast ──
    cb = build_callback(seq, [Pose(first=1, last=1, dx=0.01)])
    seq[2].Thickness.mul_(2.0)  # type: ignore[attr-defined]
    try:
        seq(callback=cb)
        raised = False
    except RuntimeError as e:
        raised = "stale" in str(e)
    check("stale fingerprint raises", raised)
    seq[2].Thickness.mul_(
        0.5
    )  # 复原（版本号已变，仅供后续测试用新回调）  # type: ignore[attr-defined]

    # ── 6. 声明校验 fail-fast ──
    for name, errors in (
        ("empty pose rejected", [Pose(first=1)]),
        (
            "duplicate scatter rejected",
            [Scatter(surface=1, std=1e-3), Scatter(surface=1, std=2e-3)],
        ),
        ("bad interval rejected", [Pose(first=2, last=1, dx=0.1)]),
        ("surface range rejected", [Pose(first=99, dx=0.1)]),
    ):
        try:
            build_callback(seq, errors)
            ok = False
        except ValueError:
            ok = True
        check(name, ok)

    # 首面无前置元件（链首即 Refractor）：注入点不存在，必须报错
    headless = Sequential.from_options(1, [LENS_FRONT, GAP_5, SENSOR])
    try:
        build_callback(headless, [Pose(first=1, dx=0.1)])
        ok = False
    except ValueError as e:
        ok = "no preceding component" in str(e)
    check("headless first surface rejected", ok)

    # ── 7. 散射：主光线（pupil=0）冻结，其余光线加噪且方向保持单位长 ──
    chief_nom: list[torch.Tensor] = []

    def rec_dirs(_comp, flow, i):
        if i == 1:
            chief_nom.append(flow.rays.directions.clone())
        return flow

    seq(callback=rec_dirs)
    torch.manual_seed(0)
    cb = build_callback(seq, [Scatter(surface=1, std=1e-3)])
    pert_dirs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def rec_dirs2(_comp, flow, i):
        if i == 1:
            pert_dirs.append((flow.rays.directions.clone(), flow.rays.pupil.clone()))
        return flow

    seq(callback=[cb, rec_dirs2])
    d_p, pupil = pert_dirs[0]
    chief = pupil.square().sum(dim=-1).eq(0.0)
    check("chief ray frozen", (d_p[chief] - chief_nom[0][chief]).abs().max().item() == 0.0)
    off = d_p[~chief] - chief_nom[0][~chief]
    check("off-chief scattered", off.abs().max().item() > 1e-6)
    check(
        "directions unit norm",
        (d_p.norm(dim=-1) - 1.0).abs().max().item() < 1e-9,
    )
    # 误差幅度 ~ std 量级
    check("scatter magnitude ~ std", 1e-4 < off.norm(dim=-1).median().item() < 1e-2)

    # ── 8. 裁剪：小半径杀死边缘光线，死因 = APERTURE_CLIP；大半径全活 ──
    cb = build_callback(seq, [Clip(radius=1.0)])
    flow = seq(callback=cb)
    dead = ~flow.verdict.hold
    check("clip kills edge rays", dead.any().item())
    check(
        "clip cause",
        (flow.verdict.cause[dead] == 4).all().item(),
    )
    check("clip toll non-negative", (flow.verdict.toll >= 0).all().item())
    cb = build_callback(seq, [Clip(radius=100.0)])
    flow = seq(callback=cb)
    check("wide clip adds no deaths", torch.equal(flow.verdict.hold, seq().verdict.hold))

    # ── 9. 探测不污染 RNG 流（随机光瞳采样器下才可见） ──
    rand_block = {**SOURCE_BLOCK, "pupil": {"method": "random", "region": "disk", "count": 16}}
    seq_r = Sequential.from_options(1, [rand_block, LENS_FRONT, GAP_5, LENS_BACK, GAP_92, SENSOR])
    torch.manual_seed(0)
    build_callback(seq_r, [Pose(first=1, last=1, dx=0.01)])
    after = torch.rand(1)
    torch.manual_seed(0)
    expected = torch.rand(1)
    check("probe does not pollute RNG", torch.equal(after, expected))

    # ── 10. 配置映射分派 ──
    e = from_options({"type": "pose", "first": 1, "dx": {"method": "normal", "std": 0.01}})
    check(
        "from_options dispatch", isinstance(e, Pose) and e.dx == {"method": "normal", "std": 0.01}
    )

    print("\nALL PASS")


if __name__ == "__main__":
    main()
