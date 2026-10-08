"""PSF 采样网格规格：显式网格是一等公民。"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GridSpec:
    """像面采样网格。

    * ``size``：边长 H（H×H）。
    * ``delta``：像素节距 mm/px；``None`` → 自动 λ_min/(4·NA_max)。
    * ``center``：网格中心的传感器局部坐标 (x, y) mm；``None`` → 逐视场自动
      （各波长主光线像点的均值；主光线死亡则存活光线质心兜底）。
    """

    size: int
    delta: float | None = None
    center: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.size < 8:
            raise ValueError(f"grid size must be >= 8, got {self.size}")
        if self.delta is not None and self.delta <= 0:
            raise ValueError(f"grid delta must be positive, got {self.delta}")
