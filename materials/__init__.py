"""材料模块：折射率数据库（单例）与可变材料容器。

模块组织
--------
* :mod:`materials.protocol` —— 抽象基类 :class:`MaterialDatabase`（逐子类单例
  + 注册表分派）。
* :mod:`materials.constant` —— 常折射率数据库。
* :mod:`materials.sellmeier` —— Sellmeier 色散数据库。
* :mod:`materials.material` —— 可变材料容器 :class:`Material` 与只读视图
  :class:`MaterialRef`。
"""

from materials.constant import ConstantMaterialDatabase
from materials.material import Material, MaterialRef
from materials.protocol import MaterialDatabase
from materials.sellmeier import SellmeierMaterialDatabase

__all__ = [
    "ConstantMaterialDatabase",
    "Material",
    "MaterialDatabase",
    "MaterialRef",
    "SellmeierMaterialDatabase",
]
