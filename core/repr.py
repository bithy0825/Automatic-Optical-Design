"""统一打印系统：rich 树形 + 颜色。

* :func:`styled` — 单行标签（类名彩色 + 参数灰化，rich 标记文本）。
* :func:`render_line` — 把标签渲染为字符串（叶子对象的 ``__repr__``）。
* :func:`render_tree` — 沿 ``nn.Module`` 子树展开 ``_label()`` 鸭子类型
  协议，渲染 rich 树为字符串（模块容器的 ``__repr__``）。

颜色仅在真实终端输出；重定向/CI 环境自动退化为纯文本树。
"""

from typing import Any

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
    head = f"[bold {_PALETTE.get(name, 'cyan')}]{escape(name)}[/]"
    return head if not info else f"{head} [grey70]{escape(info)}[/grey70]"


def render_line(markup: str) -> str:
    with _CONSOLE.capture() as cap:
        _CONSOLE.print(markup, end="")
    return cap.get()


def render_tree(root: Any) -> str:
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
