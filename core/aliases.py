"""张量形状约定：jaxtyping 注解别名与 System→Ray 形状对齐原语。

* ``Ray*``    —— 逐光线张量，形状 ``(P, F, W, N, ...)``：每个种群个体、
  每个视场、每个波长下的每一根光线各有一个值。
* ``System*`` —— 逐个体张量，形状 ``(P, ...)``：每个种群个体（一套独立
  光学系统）各有一个值。

维度字母含义：

* P: population —— 种群大小（独立光学系统的数量）
* F: field      —— 每个系统的视场数
* W: wavelength —— 每个系统的波长数
* N: ray        —— 每个 (视场, 波长) 组合追踪的光线数
"""

from typing import Final

import torch
from jaxtyping import Bool, Float, Int64

# ── Ray 世界：逐光线 (P, F, W, N, ...) ──

type RayFloatScalar = Float[torch.Tensor, "P F W N"]
type RayFloat2D = Float[torch.Tensor, "P F W N 2"]
type RayFloat3D = Float[torch.Tensor, "P F W N 3"]

type RayBoolScalar = Bool[torch.Tensor, "P F W N"]
type RayLongScalar = Int64[torch.Tensor, "P F W N"]

type RayFloatMatrix2D = Float[torch.Tensor, "P F W N 2 2"]
type RayFloatMatrix3D = Float[torch.Tensor, "P F W N 3 3"]
type RayFloatMatrix4D = Float[torch.Tensor, "P F W N 4 4"]

# ── System 世界：逐个体 (P, ...) ──

type SystemFloatScalar = Float[torch.Tensor, "P"]
type SystemFloat2D = Float[torch.Tensor, "P 2"]
type SystemFloat3D = Float[torch.Tensor, "P 3"]
type SystemFloatND = Float[torch.Tensor, "P N"]

type SystemBoolScalar = Bool[torch.Tensor, "P"]
type SystemBoolND = Bool[torch.Tensor, "P N"]
type SystemLongScalar = Int64[torch.Tensor, "P"]

type SystemFloatMatrix2D = Float[torch.Tensor, "P 2 2"]
type SystemFloatMatrix3D = Float[torch.Tensor, "P 3 3"]
type SystemFloatMatrix4D = Float[torch.Tensor, "P 4 4"]

type HomMatrix = SystemFloatMatrix4D  # 齐次 4×4 变换矩阵（语义别名）

_LEADING_AXES: Final[int] = 4  # 逐光线张量的前导维数 (P, F, W, N)


def broadcast_system_to_ray(system_tensor: torch.Tensor, ray_tensor: torch.Tensor) -> torch.Tensor:
    """把逐个体张量 ``(P, ...)`` 广播对齐到逐光线张量 ``(P, F, W, N, ...)``。

    个体维之后插入 (F, W, N) 三个单维再 ``expand_as`` 对齐；精度跟随
    *ray_tensor*。两个张量的维度数之差必须恰为 3。
    """
    n_leading = ray_tensor.ndim - system_tensor.ndim
    if n_leading != _LEADING_AXES - 1:
        raise ValueError(
            f"system tensor must have exactly {_LEADING_AXES - 1} fewer axes than "
            f"the ray tensor, got {system_tensor.ndim} vs {ray_tensor.ndim}"
        )
    new_shape = (system_tensor.shape[0],) + (1,) * n_leading + tuple(system_tensor.shape[1:])
    return system_tensor.reshape(new_shape).expand_as(ray_tensor).to(ray_tensor.dtype)
