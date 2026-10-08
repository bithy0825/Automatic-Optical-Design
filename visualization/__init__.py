"""可视化模块：单页前端 + 本地追迹服务（零前端构建，原生 SVG）。

用法::

    python -m visualization [--port 8000] [--open]

随后在页面里打开 .toml / .pth 文件，切换光路图 / 点列图。

编程入口::

    from visualization import serve
    serve(port=8000)  # 阻塞，Ctrl+C 退出

模块组织
--------
* :mod:`visualization.trace` —— 2D 视图追迹（光路图 / 点列图数据提取）。
* :mod:`visualization.server` —— 本地 HTTP 服务（仅标准库）。
"""

from visualization.server import serve

__all__ = [
    "serve",
]
