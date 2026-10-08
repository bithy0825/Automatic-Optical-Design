"""优化模块：损失函数、目标规格、梯度下降 / 模拟退火 / 遗传算法。

模块组织
--------
* :mod:`optimization.target` —— 设计目标 :class:`Target`。
* :mod:`optimization.loss` —— 五项损失与 :class:`LossWeights`。
* :mod:`optimization.gradient` —— 梯度优化器（Adam / AdamW / SGD）。
* :mod:`optimization.annealing` —— 模拟退火。
* :mod:`optimization.genetic` —— GA 编排器。
* :mod:`optimization.callback` —— 回调协议与内置实现。
* :mod:`optimization.utils` —— 配置装载、系统构建与存档。
"""

from optimization.annealing import SAOptions, SimulatedAnnealing
from optimization.callback import Callback, LossHistory, PeriodicSaver, ProgressBar
from optimization.genetic import GAOptions, GeneticAlgorithm, Stager
from optimization.gradient import (
    AdamOptions,
    AdamWOptions,
    GradientOptimizer,
    SGDOptions,
)
from optimization.loss import (
    LossWeights,
    blur_loss,
    bounds_loss,
    distortion_loss,
    effl_loss,
    toll_loss,
    total_loss,
)
from optimization.target import Target
from optimization.utils import (
    build_sequential,
    build_stage,
    build_target,
    load,
    load_config,
    save,
)

__all__ = [
    # options
    "AdamOptions",
    "AdamWOptions",
    "GAOptions",
    "LossWeights",
    "SAOptions",
    "SGDOptions",
    # optimizers
    "GeneticAlgorithm",
    "GradientOptimizer",
    "SimulatedAnnealing",
    "Stager",
    # loss
    "blur_loss",
    "bounds_loss",
    "distortion_loss",
    "effl_loss",
    "toll_loss",
    "total_loss",
    # target
    "Target",
    # utils
    "build_sequential",
    "build_stage",
    "build_target",
    "load",
    "load_config",
    "save",
    # callback
    "Callback",
    "LossHistory",
    "PeriodicSaver",
    "ProgressBar",
]
