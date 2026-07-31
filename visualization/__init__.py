"""Web 可视化:bun + three.js 前端 + 本地追迹服务。

用法::

    python -m visualization --config zebase/A001/config.toml [--port 8000] [--open]
    python -m visualization --pth trained.pth [--port 8000] [--open]

编程入口::

    from visualization import serve
    serve(seq, target=target)  # 阻塞,Ctrl+C 退出
"""

from visualization.server import serve

__all__ = ["serve"]
