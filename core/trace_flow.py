from dataclasses import dataclass
from typing import Self

from core.container import TensorContainer
from core.ray_bundle import RayBundle
from core.transformer import Transformer
from core.verdict import Verdict


@dataclass(slots=True, eq=False, repr=False)
class TraceFlow(TensorContainer):
    """在元件间流动的追迹状态：光线束 + 当前位姿 + 累计裁决。"""

    rays: RayBundle
    transformer: Transformer
    verdict: Verdict

    def with_rays(self, rays: RayBundle) -> Self:
        return self.replace(rays=rays)

    def with_transformer(self, transformer: Transformer) -> Self:
        return self.replace(transformer=transformer)

    def at_verdict(self, verdict: Verdict) -> Self:
        return self.replace(verdict=self.verdict.at(verdict))
