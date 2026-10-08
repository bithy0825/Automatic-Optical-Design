"""统一打印系统：rich 树形 + 颜色。

* :func:`styled` —— 单行标签（类名彩色 + 参数灰化，rich 标记文本）。
* :func:`render_line` —— 把标签渲染为字符串（叶子对象的 ``__repr__``）。
* :func:`render_tree` —— 沿 ``nn.Module`` 子树展开 ``_label()`` / ``_params()``
  鸭子类型协议，渲染 rich 树为字符串（模块容器的 ``__repr__``）。
* :func:`fmt_param` —— 张量 / 序列的紧凑单行格式化（参数行用）。

颜色仅在真实终端输出；重定向 / CI 环境自动退化为纯文本树。
"""

from typing import Any

import torch
from rich.console import Console
from rich.markup import escape
from rich.tree import Tree
from torch import nn

_CONSOLE = Console(width=120)

_PALETTE: dict[str, str] = {
    "InfiniteSource": "bright_cyan",
    "Gap": "bright_black",
    "Refractor": "bright_green",
    "Sensor": "bright_yellow",
    "Material": "magenta",
    "MaterialRef": "magenta",
    "MaterialDatabase": "magenta",
    "ConstantMaterialDatabase": "magenta",
    "SellmeierMaterialDatabase": "magenta",
}

# 树中不显示为节点的类（材料以 incident/transmitted 参数行的形式呈现）
_SKIP_NODES: frozenset[str] = frozenset({"Material"})


def styled(name: str, info: str = "") -> str:
    """单行标签：类名彩色，*info* 参数灰化（rich 标记文本）。"""
    head = f"[bold {_PALETTE.get(name, 'cyan')}]{escape(name)}[/]"
    return head if not info else f"{head} [grey70]{escape(info)}[/grey70]"


def render_line(markup: str) -> str:
    """把 rich 标记文本渲染为单行字符串。"""
    with _CONSOLE.capture() as cap:
        _CONSOLE.print(markup, end="")
    return cap.get()


def render_tree(root: Any) -> str:
    """把模块容器渲染为 rich 树字符串（``_label`` / ``_params`` 鸭子类型协议）。"""

    def build(node: Any) -> Tree:
        label = getattr(node, "_label", lambda: type(node).__name__)()
        tree = Tree(label, guide_style="grey42")
        for row in getattr(node, "_params", lambda: ())():
            tree.add(f"[grey70]{escape(str(row))}[/grey70]")
        for child in getattr(node, "children", lambda: ())():
            if type(child) is nn.ModuleList:  # 容器无语义，展平其元素
                for sub in child:
                    tree.add(build(sub))
            elif type(child).__name__ in _SKIP_NODES:
                continue
            else:
                tree.add(build(child))
        return tree

    with _CONSOLE.capture() as cap:
        _CONSOLE.print(build(root))
    return cap.get().rstrip("\n")


def fmt_param(x: torch.Tensor | list | tuple, *, precision: int = 4, max_items: int = 8) -> str:
    """张量 / 序列 → 紧凑单行文本（浮点按 *precision* 位有效数字，超长截断）。"""

    def fmt_item(v: object) -> str:
        return f"{v:.{precision}g}" if type(v) in (int, float) else str(v)

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 0:
            return fmt_item(x.item())
        vals = x.tolist()
    elif isinstance(x, (list, tuple)):
        vals = list(x)
    else:
        raise TypeError(f"unsupported type for fmt_param: {type(x).__name__}")

    head = ", ".join(fmt_item(v) for v in vals[:max_items])
    return f"[{head}]" if len(vals) <= max_items else f"[{head}, …]"
