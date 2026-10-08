"""可视化服务入口。

用法::

    python -m visualization [--port 8000] [--open]

随后在页面里打开 .toml / .pth 文件，切换光路图 / 点列图。
"""

import argparse

from visualization.server import serve


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m visualization",
        description="光学系统可视化服务（浏览器单页：光路图 / 点列图）",
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = p.parse_args()
    serve(port=args.port, open_browser=args.open)


if __name__ == "__main__":
    main()
