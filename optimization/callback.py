"""优化回调协议与内置实现。"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from component import Sequential


class Callback(Protocol):
    def on_step_end(
        self, gen: int, step: int, stage: str, metrics: dict[str, Any]
    ) -> None: ...
    def on_gen_end(self, gen: int, metrics: dict[str, Any]) -> None: ...


class LossHistory:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def on_step_end(
        self, gen: int, step: int, stage: str, metrics: dict[str, Any]
    ) -> None:
        self.records.append({"gen": gen, "step": step, "stage": stage, **metrics})

    def on_gen_end(self, gen: int, metrics: dict[str, Any]) -> None:
        self.records.append({"gen": gen, "step": -1, "stage": "ga", **metrics})


class ProgressBar:
    """双层进度条：外层 GA 代数 + 内层优化步数。

    ``total_steps`` 传 int 表示所有阶段共用；传 ``{阶段名: 步数}`` 映射则
    阶段切换时同步切换内层总数（阶段名大小写不敏感，如 sa/SA）。
    """

    def __init__(
        self, total_gen: int, total_steps: int | Mapping[str, int] = 0
    ) -> None:
        self.gen_bar = tqdm(total=total_gen, desc="GA", position=0, unit="gen")
        self._totals = (
            {k.lower(): int(v) for k, v in total_steps.items()}
            if isinstance(total_steps, Mapping)
            else {}
        )
        self._default_total = (
            max(self._totals.values(), default=0) if self._totals else total_steps
        )
        self.step_bar = (
            tqdm(
                total=self._default_total,  # type: ignore
                desc="  step",
                position=1,
                unit="step",
                leave=False,
            )
            if self._default_total
            else None
        )
        self._last_gen = -1
        self._last_stage = ""

    def on_step_end(
        self, gen: int, step: int, stage: str, metrics: dict[str, Any]
    ) -> None:
        if stage != self._last_stage:
            if self.step_bar:
                self.step_bar.reset(
                    total=self._totals.get(stage.lower(), self._default_total)  # type: ignore
                )
            self._last_stage = stage
        if self.step_bar:
            self.step_bar.set_postfix(
                {"loss": f"{metrics.get('loss', metrics.get('blur', 0)):.3g}"}
            )
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


class PeriodicSaver:
    """每 ``every`` 代把检查点滚动覆盖保存到 ``path``（与最终存档同路径）。"""

    def __init__(
        self, seq: Sequential, cfg: Mapping[str, Any], path: Path, every: int
    ) -> None:
        self.seq = seq
        self.cfg = cfg
        self.path = path
        self.every = every

    def on_step_end(
        self, gen: int, step: int, stage: str, metrics: dict[str, Any]
    ) -> None:
        pass

    def on_gen_end(self, gen: int, metrics: dict[str, Any]) -> None:
        if (gen + 1) % self.every == 0:
            from optimization.utils import save  # 延迟导入，避免循环

            self.path.parent.mkdir(parents=True, exist_ok=True)
            save(self.seq, self.cfg, self.path)
            tqdm.write(f"[save] gen {gen + 1} -> {self.path}")
