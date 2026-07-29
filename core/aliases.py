"""张量形状别名（jaxtyping 注解）。

* ``Ray*``    —— 逐光线张量，形状 ``(P, F, W, N, ...)``。每个种群个体、
  每个视场、每个波长下的每一根光线各有一个值。
* ``System*`` —— 逐个体张量，形状 ``(P, ...)``。每个种群个体（一套独立
  光学系统）各有一个值。

维度字母含义：

* P: population —— 种群大小（独立光学系统的数量）
* F: field      —— 每个系统的视场数
* W: wavelength —— 每个系统的波长数
* N: ray        —— 每个 (视场, 波长) 组合追踪的光线数
"""

from typing import TypeAlias

import torch
from jaxtyping import Bool, Float, Int64

# ── Ray 世界：逐光线 (P, F, W, N, ...) ──

RayFloatScalar: TypeAlias = Float[torch.Tensor, "P F W N"]
RayFloat2D: TypeAlias = Float[torch.Tensor, "P F W N 2"]
RayFloat3D: TypeAlias = Float[torch.Tensor, "P F W N 3"]

RayBoolScalar: TypeAlias = Bool[torch.Tensor, "P F W N"]

RayLongScalar: TypeAlias = Int64[torch.Tensor, "P F W N"]

RayFloatMatrix2D: TypeAlias = Float[torch.Tensor, "P F W N 2 2"]
RayFloatMatrix3D: TypeAlias = Float[torch.Tensor, "P F W N 3 3"]
RayFloatMatrix4D: TypeAlias = Float[torch.Tensor, "P F W N 4 4"]

# ── System 世界：逐个体 (P, ...) ──

SystemFloatScalar: TypeAlias = Float[torch.Tensor, "P"]
SystemFloat2D: TypeAlias = Float[torch.Tensor, "P 2"]
SystemFloat3D: TypeAlias = Float[torch.Tensor, "P 3"]
SystemFloatND: TypeAlias = Float[torch.Tensor, "P N"]

SystemBoolScalar: TypeAlias = Bool[torch.Tensor, "P"]
SystemBoolND: TypeAlias = Bool[torch.Tensor, "P N"]

SystemLongScalar: TypeAlias = Int64[torch.Tensor, "P"]

SystemFloatMatrix2D: TypeAlias = Float[torch.Tensor, "P 2 2"]
SystemFloatMatrix3D: TypeAlias = Float[torch.Tensor, "P 3 3"]
SystemFloatMatrix4D: TypeAlias = Float[torch.Tensor, "P 4 4"]

HomMatrix: TypeAlias = SystemFloatMatrix4D  # 齐次 4×4 变换矩阵（语义别名）
