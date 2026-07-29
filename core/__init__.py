# Type aliases (jaxtyping-annotated)
from core.aliases import (
    HomMatrix,
    RayBoolScalar,
    RayFloat2D,
    RayFloat3D,
    RayFloatMatrix2D,
    RayFloatMatrix3D,
    RayFloatMatrix4D,
    RayFloatScalar,
    RayLongScalar,
    SystemBoolND,
    SystemBoolScalar,
    SystemFloat2D,
    SystemFloat3D,
    SystemFloatMatrix2D,
    SystemFloatMatrix3D,
    SystemFloatMatrix4D,
    SystemFloatND,
    SystemFloatScalar,
    SystemLongScalar,
)
from core.container import TensorContainer
from core.module import OpticalModule, init_param
from core.noun import Noun
from core import term
from core.ray_bundle import RayBundle
from core.sturdy_math import sturdy_div, sturdy_inv, sturdy_sqrt
from core.trace_flow import TraceFlow
from core.transformer import Transformer
from core.utils import broadcast_system_to_ray, fmt_param, parse_param
from core.verdict import Verdict

__all__ = [
    # aliases
    "RayBoolScalar",
    "RayFloat2D",
    "RayFloat3D",
    "RayFloatMatrix2D",
    "RayFloatMatrix3D",
    "RayFloatMatrix4D",
    "RayFloatScalar",
    "RayLongScalar",
    "HomMatrix",
    "SystemBoolND",
    "SystemBoolScalar",
    "SystemFloat2D",
    "SystemFloat3D",
    "SystemFloatMatrix2D",
    "SystemFloatMatrix3D",
    "SystemFloatMatrix4D",
    "SystemFloatND",
    "SystemFloatScalar",
    "SystemLongScalar",
    # container & module
    "TensorContainer",
    "OpticalModule",
    "init_param",
    "parse_param",
    # terminology
    "Noun",
    "term",
    # trace data
    "RayBundle",
    "TraceFlow",
    "Transformer",
    "Verdict",
    # math
    "sturdy_div",
    "sturdy_inv",
    "sturdy_sqrt",
    "broadcast_system_to_ray",
    "fmt_param",
]
