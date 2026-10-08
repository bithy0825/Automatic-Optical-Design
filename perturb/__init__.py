"""误差注入模块：面向公差/装配仿真的声明式误差与回调编译。

可注入的误差（回调三通道：位姿 / 光线 / 裁决）：

* :class:`~perturb.error.Pose`    —— 位姿类：面/元件/组的偏心、倾斜、楔角、
  间隔误差、传感器离焦倾斜（transformer 通道，绝对复位语义）；
* :class:`~perturb.error.Scatter` —— 面散射近似：折射后方向加角噪声（rays 通道）；
* :class:`~perturb.error.Clip`    —— 机械遮挡：镜筒/垫圈半径限光（verdict 通道）。

元件内部物理（曲率/面形/材料/镀膜/偏振）不在回调可达范围，走参数扰动。

用法::

    from perturb import build_callback

    cb = build_callback(seq, [
        {"type": "pose", "first": 3, "last": 4, "dx": {"method": "normal", "std": 0.01}},
        {"type": "scatter", "surface": 2, "std": 1e-4},
        {"type": "clip", "radius": 12.0},
    ])
    flow = seq(callback=cb)   # 种群维即蒙特卡洛维

模块组织
--------
* :mod:`perturb.error` —— 误差声明（纯数据规格 + 配置解析）。
* :mod:`perturb.inject` —— 误差声明 → 追迹回调的编译。
"""

from perturb.error import Clip, Pose, Scatter, from_options
from perturb.inject import build_callback

__all__ = [
    "Clip",
    "Pose",
    "Scatter",
    "build_callback",
    "from_options",
]
