"""处方转换模块：Zemax .zmx / OSLO .len → 训练配置 TOML。

模块组织
--------
* :mod:`convert._common` —— 共享机械：面数据结构、过滤异常、EFFL/F 数/视场
  推断公共件、TOML 发射器与 CLI 驱动。
* :mod:`convert.zmx` —— Zemax 处方（``python -m convert.zmx``）。
* :mod:`convert.oslo` —— OSLO 处方（``python -m convert.oslo``）。
"""

from convert._common import Skip

__all__ = [
    "Skip",
]
