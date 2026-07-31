"""优化模块：损失函数、目标规格、梯度下降 / 模拟退火 / 遗传算法。"""

from optimization.annealing import SAOptions, SimulatedAnnealing
from optimization.callback import Callback, LossHistory, ProgressBar
from optimization.genetic import GAOptions, GeneticAlgorithm, Stager
from optimization.gradient import (
    AdamOptions,
    GradientOptimizer,
    SGDOptions,
)
from optimization.loss import (
    LossWeights,
    bounds_loss,
    effl_loss,
    spot_loss,
    toll_loss,
    total_loss,
)
from optimization.target import Target
from optimization.utils import build_sequential, load, load_config, build_target, save

__all__ = [
    # options
    "AdamOptions",
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
    "bounds_loss",
    "effl_loss",
    "spot_loss",
    "toll_loss",
    "total_loss",
    # target
    "Target",
    # utils
    "load_config",
    "build_target",
    "build_sequential",
    "save",
    "load",
    # callback
    "Callback",
    "LossHistory",
    "ProgressBar",
]
