"""优化回调协议与内置实现。"""

from typing import Any, Protocol

from tqdm import tqdm


class Callback(Protocol):
    def on_step_end(self, gen: int, step: int, stage: str, metrics: dict[str, Any]) -> None: ...
    def on_gen_end(self, gen: int, metrics: dict[str, Any]) -> None: ...


class LossHistory:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def on_step_end(self, gen: int, step: int, stage: str, metrics: dict[str, Any]) -> None:
        self.records.append({"gen": gen, "step": step, "stage": stage, **metrics})

    def on_gen_end(self, gen: int, metrics: dict[str, Any]) -> None:
        self.records.append({"gen": gen, "step": -1, "stage": "ga", **metrics})


class ProgressBar:
    """双层进度条：外层 GA 代数 + 内层优化步数。"""

    def __init__(self, total_gen: int, total_steps: int = 0) -> None:
        self.gen_bar = tqdm(total=total_gen, desc="GA", position=0, unit="gen")
        self.step_bar = (
            tqdm(total=total_steps, desc="  step", position=1, unit="step", leave=False)
            if total_steps else None
        )
        self._last_gen = -1
        self._last_stage = ""

    def on_step_end(self, gen: int, step: int, stage: str, metrics: dict[str, Any]) -> None:
        if stage != self._last_stage:
            if self.step_bar:
                self.step_bar.reset()
            self._last_stage = stage
        if self.step_bar:
            self.step_bar.set_postfix({"loss": f"{metrics.get('loss', metrics.get('spot', 0)):.3g}"})
            self.step_bar.update(1)

    def on_gen_end(self, gen: int, metrics: dict[str, Any]) -> None:
        self.gen_bar.set_postfix({"loss": f"{metrics.get('loss', 0):.3g}"})
        self.gen_bar.update(1)
        if self.step_bar:
            self.step_bar.reset()

    def close(self) -> None:
        self.gen_bar.close()
        if self.step_bar:
            self.step_bar.close()
