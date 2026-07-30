"""死亡复活实验：死亡矩阵状态 → 仅用 toll 损失做 Adam 优化 → 对比各死因死亡数。

验证：死亡损失的梯度不仅在数值上存在，而且能真正驱动参数把光线拉回存活。
顺带使用 docs/loss_param_forces.md 推荐的 lr 配置，兼作其正确性验证。

运行: python tests/test_resurrection.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from optimization import AdamOptions, GradientOptimizer, build_sequential
from optimization.loss import LossWeights, total_loss
from optimization.utils import build_target
from tests.measure_loss_forces import (
    CAUSES,
    CONFIG,
    NOUNS,
    apply_death_matrix,
    death_report,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def toll_only(seq, target, blocks) -> float:
    with torch.no_grad():
        flow = seq()
        w = LossWeights(
            effl=0.0,
            spot=0.0,
            toll={c: 1.0 for c in CAUSES},
            bounds={n: 0.0 for n in NOUNS},
        )
        total, _ = total_loss(flow, seq, target, blocks, w)
        return float(total.mean())


def main() -> None:
    torch.manual_seed(0)
    target = build_target(CONFIG)
    seq = build_sequential(CONFIG, target).to(DEVICE)
    blocks = list(CONFIG["component"])
    apply_death_matrix(seq)

    print("== 优化前 ==")
    death_report(seq, "死亡分布")
    loss0 = toll_only(seq, target, blocks)
    print(f"toll loss: {loss0:.4g}")

    # 仅 toll 损失（死亡惩罚），docs 推荐的分参数 lr
    weights = LossWeights(
        effl=0.0,
        spot=0.0,
        toll={c: 1.0 for c in CAUSES},
        bounds={n: 0.0 for n in NOUNS},
    )
    adam = GradientOptimizer(
        AdamOptions(
            step=80,
            scheduler="none",
            grad_norm=10.0,
            default_lr=1e-3,
            lr={
                "curvature": 1e-3,
                "thickness": 0.05,
                "diameter": 0.02,
                "kappa": 0.005,
                "alpha": 1e-4,
            },
        )
    )
    adam.run(seq, target, 0, blocks, weights)

    print("\n== toll-only Adam 80 步后 ==")
    death_report(seq, "死亡分布")
    loss1 = toll_only(seq, target, blocks)
    print(f"toll loss: {loss1:.4g}  ({loss0:.4g} -> {loss1:.4g})")


if __name__ == "__main__":
    main()
