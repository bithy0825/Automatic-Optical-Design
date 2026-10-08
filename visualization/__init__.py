"""可视化模块：bun + three.js 前端 + 本地追迹服务（光路图 / 点列图）。

用法::

    python -m visualization [--port 8000] [--open]

随后在页面里打开 .toml / .pth 文件，进入光路图或点列图视图。
前端源码在 ``visualization/web``（bun 管理，依赖 three.js）；改动后需在
该目录执行 ``bun run build`` 生成 ``web/dist``。

编程入口::

    from visualization import serve
    serve(port=8000)  # 阻塞，Ctrl+C 退出

模块组织
--------
* :mod:`visualization.trace` —— 追迹管线与缓存（光路图 / 点列图数据）。
* :mod:`visualization.protocol` —— 二进制帧协议（Python 打包 ↔ TS 解码）。
* :mod:`visualization.server` —— 本地 HTTP 服务（仅标准库）。
"""

from visualization.server import serve

__all__ = [
    "serve",
]
