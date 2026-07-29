from dataclasses import dataclass


from core.aliases import RayFloat2D, RayFloat3D, RayFloatScalar
from core.container import TensorContainer


@dataclass(slots=True, eq=False, repr=False)
class RayBundle(TensorContainer):
    """一束光线：起点、方向与逐光线标签（光瞳坐标 / 视场 / 波长）。"""

    points: RayFloat3D
    directions: RayFloat3D
    pupil: RayFloat2D
    field: RayFloat2D
    wavelength: RayFloatScalar

    @property
    def population(self) -> int:
        return self.points.shape[0]

    @property
    def num_fields(self) -> int:
        return self.points.shape[1]

    @property
    def num_wavelengths(self) -> int:
        return self.points.shape[2]
