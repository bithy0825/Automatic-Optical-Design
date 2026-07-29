from dataclasses import dataclass

from core import RayFloat3D, Verdict


@dataclass(slots=True, eq=False)
class InteractionResult:
    """光线与表面相互作用的结果。

    Attributes:
        directions: 出射方向（折射/反射后），单位化。
        verdict:    交互裁决（如 TIR 判死，toll = 1 − η²sin²θᵢ）。
    """

    directions: RayFloat3D
    verdict: Verdict
