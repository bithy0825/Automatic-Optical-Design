"""光学系统 Rich 树形可视化。

为 ``Sequential`` 提供带颜色、可折叠的树形显示，纯文本回退同样直观。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from component.sequential import Sequential

# ── 紧凑格式化 ────────────────────────────────────────────────────────

_MAX_ITEMS = 6


def _compact_float(v: float, width: int = 5) -> str:
    """单个浮点紧凑显示：大数/小数用科学记数，否则用有效数字。"""
    if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0.0):
        return f"{v:.{width - 3}g}"
    return f"{v:.{width}g}"


def _compact_tensor(t: torch.Tensor, max_items: int = _MAX_ITEMS) -> str:
    """批次张量紧凑表示：``[v1, v2, v3, …, +N]``。"""
    t = t.detach().cpu()
    if t.numel() == 0:
        return "[]"
    flat = t.flatten()
    n = flat.shape[0]
    if n == 0:
        return "[]"
    vals = flat.tolist()
    if n <= max_items:
        items = ", ".join(
            _compact_float(x, 6) if isinstance(x, float) else repr(x) for x in vals
        )
        return f"[{items}]"
    head = ", ".join(
        _compact_float(x, 6) if isinstance(x, float) else repr(x)
        for x in vals[:max_items]
    )
    return f"[{head}, … +{n - max_items}]"


def _compact_names(names, max_items: int = _MAX_ITEMS) -> str:
    """材料名序列紧凑表示：``['n1', 'n2', …, +N]``。"""
    ls = list(names)
    if not ls:
        return "[]"
    if len(ls) <= max_items:
        return "[" + ", ".join(repr(n) for n in ls) + "]"
    head = ", ".join(repr(n) for n in ls[:max_items])
    return f"[{head}, … +{len(ls) - max_items}]"


def _radius_from_curvature(c: torch.Tensor) -> torch.Tensor:
    """曲率→半径：|c| > 1e-12 时取倒数，否则标记为 0。"""
    c = c.detach().cpu()
    return torch.where(c.abs() > 1e-12, 1.0 / c, torch.zeros_like(c))


# ── Rich 树构建 ───────────────────────────────────────────────────────


def _try_import_rich():
    """尝试导入 rich，失败返回 None 元组。"""
    try:
        from rich.style import Style
        from rich.text import Text
        from rich.tree import Tree as RichTree

        return RichTree, Text, Style
    except ImportError:
        return None, None, None


_COLORS: dict[str, str] = {
    "InfiniteSource": "bright_cyan",
    "Gap": "bright_black",
    "Refractor": "bright_green",
    "Sensor": "bright_yellow",
}


def _add_shape_children(node, shape, Text, Style) -> None:
    """将面形参数作为子节点展开，每个参数一行。"""
    GEO_COLOR = "green"
    SHAPE_COLOR = "cyan"
    DIM = Style(dim=True)

    cls_name = type(shape).__name__
    shape_node = node.add(
        Text("shape: ", style=DIM)
        + Text(cls_name, style=Style(color=SHAPE_COLOR, bold=True)),
    )

    shape_node.add(
        Text("D=", style=DIM)
        + Text(_compact_tensor(shape.Diameter), style=Style(color=GEO_COLOR)),
    )
    if hasattr(shape, "Curvature"):
        shape_node.add(
            Text("C=", style=DIM)
            + Text(_compact_tensor(shape.Curvature), style=Style(color=GEO_COLOR)),
        )
        shape_node.add(
            Text("R=", style=DIM)
            + Text(
                _compact_tensor(_radius_from_curvature(shape.Curvature)),
                style=Style(color=GEO_COLOR),
            ),
        )
    if hasattr(shape, "Kappa"):
        shape_node.add(
            Text("κ=", style=DIM)
            + Text(_compact_tensor(shape.Kappa), style=Style(color=GEO_COLOR)),
        )


def _build_rich_tree(seq: Sequential):
    """用 rich.Tree 构建带颜色的光学系统树。"""
    RichTree, Text, Style = _try_import_rich()
    if RichTree is None:
        return None

    first = seq[0]
    epd = str(first.epd) if hasattr(first, "epd") else "?"
    waves = first.wavelengths if hasattr(first, "wavelengths") else None
    wl_str = (
        "/".join(str(int(w)) if w == int(w) else str(w) for w in waves)
        if waves
        else "?"
    )
    header = (
        f"Sequential  {len(seq)} surfaces  "
        f"P={seq.population}  EPD={epd}  λ={wl_str}nm"
    )
    root = RichTree(header, style=Style(color="bright_blue", bold=True))

    SRC_COLOR = "bright_cyan"
    GEO_COLOR = "green"
    MAT_IN_COLOR = "yellow"
    MAT_OUT_COLOR = "bright_magenta"
    DIM = Style(dim=True)

    for i, comp in enumerate(seq):
        type_name = type(comp).__name__
        color = _COLORS.get(type_name, "white")
        label = Text(f"{i}: {type_name}", style=Style(color=color, bold=True))
        node = root.add(label)

        if type_name == "InfiniteSource":
            node.add(
                Text(
                    f"FOV_X={comp.field_x}, FOV_Y={comp.field_y}",
                    style=Style(color=SRC_COLOR),
                ),
            )
            if comp.transmitted is not None:
                node.add(
                    Text("transmitted: ", style=DIM)
                    + Text(
                        _compact_names(comp.transmitted.names()),
                        style=Style(color=MAT_OUT_COLOR),
                    ),
                )

        elif type_name == "Gap":
            node.add(
                Text("T=", style=DIM)
                + Text(_compact_tensor(comp.Thickness), style=Style(color=GEO_COLOR)),
            )

        elif type_name == "Sensor":
            if hasattr(comp, "shape"):
                node.add(
                    Text("D=", style=DIM)
                    + Text(
                        _compact_tensor(comp.shape.Diameter),
                        style=Style(color=GEO_COLOR),
                    ),
                )

        elif type_name == "Refractor":
            if comp.incident is not None:
                node.add(
                    Text("incident: ", style=DIM)
                    + Text(
                        _compact_names(comp.incident.names()),
                        style=Style(color=MAT_IN_COLOR),
                    ),
                )
            if comp.transmitted is not None:
                node.add(
                    Text("transmitted: ", style=DIM)
                    + Text(
                        _compact_names(comp.transmitted.names()),
                        style=Style(color=MAT_OUT_COLOR),
                    ),
                )
            if hasattr(comp, "shape"):
                _add_shape_children(node, comp.shape, Text, Style)

    return root


def _build_text_tree(seq: Sequential) -> str:
    """纯文本树形表示（无 rich 时的回退）。"""
    first = seq[0]
    epd = str(first.epd) if hasattr(first, "epd") else "?"
    waves = first.wavelengths if hasattr(first, "wavelengths") else None
    wl_str = (
        "/".join(str(int(w)) if w == int(w) else str(w) for w in waves)
        if waves
        else "?"
    )
    lines = [
        f"Sequential  {len(seq)} surfaces  "
        f"P={seq.population}  EPD={epd}  λ={wl_str}nm"
    ]

    for i, comp in enumerate(seq):
        type_name = type(comp).__name__

        is_last = i == len(seq) - 1
        branch = "└──" if is_last else "├──"
        child_prefix = "    " if is_last else "│   "

        lines.append(f"{branch} {i}: {type_name}")

        children: list[str] = []

        if type_name == "InfiniteSource":
            children.append(f"FOV_X={comp.field_x}, FOV_Y={comp.field_y}")
            if comp.transmitted is not None:
                children.append(
                    f"transmitted: {_compact_names(comp.transmitted.names())}"
                )

        elif type_name == "Gap":
            children.append(f"T={_compact_tensor(comp.Thickness)}")

        elif type_name == "Sensor":
            if hasattr(comp, "shape"):
                children.append(f"D={_compact_tensor(comp.shape.Diameter)}")

        elif type_name == "Refractor":
            if comp.incident is not None:
                children.append(
                    f"incident: {_compact_names(comp.incident.names())}"
                )
            if comp.transmitted is not None:
                children.append(
                    f"transmitted: {_compact_names(comp.transmitted.names())}"
                )
            if hasattr(comp, "shape"):
                shape = comp.shape
                cls_name = type(shape).__name__
                children.append(f"shape: {cls_name}")
                children.append(f"  D={_compact_tensor(shape.Diameter)}")
                if hasattr(shape, "Curvature"):
                    children.append(f"  C={_compact_tensor(shape.Curvature)}")
                    children.append(
                        f"  R={_compact_tensor(_radius_from_curvature(shape.Curvature))}"
                    )
                if hasattr(shape, "Kappa"):
                    children.append(f"  κ={_compact_tensor(shape.Kappa)}")

        for j, child in enumerate(children):
            is_last_child = j == len(children) - 1
            cb = "└──" if is_last_child else "├──"
            lines.append(f"{child_prefix}{cb} {child}")

    return "\n".join(lines)


def repr_sequential(seq: Sequential) -> str:
    """Sequential 的增强 repr：优先 rich 树，回退纯文本树。"""
    RichTree, *_ = _try_import_rich()
    if RichTree is not None:
        from rich.console import Console

        tree = _build_rich_tree(seq)
        if tree is not None:
            console = Console(no_color=False, width=120)
            with console.capture() as capture:
                console.print(tree)
            return capture.get().rstrip("\n")

    return _build_text_tree(seq)
