"""CLI:python -m visualization (--config <toml> | --pth <checkpoint>) [--port N] [--open]"""

import argparse

import torch

from optimization import (
    LossWeights,
    build_sequential,
    build_target,
    load,
    load_config,
)
from visualization.server import serve


def main() -> None:
    p = argparse.ArgumentParser(
        prog="visualization", description="Web 可视化服务(光路图 / 点列图)"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="配置 TOML 路径(初始种群)")
    src.add_argument("--pth", help="训练检查点 .pth 路径(训练后种群)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = p.parse_args()

    if args.pth:
        seq, target = load(args.pth)
        cfg, _ = torch.load(args.pth, weights_only=True, map_location="cpu")
    else:
        cfg = load_config(args.config)
        target = build_target(cfg)
        seq = build_sequential(cfg, target)
    weights = LossWeights.from_options(cfg)
    serve(
        seq,
        target=target,
        blocks=cfg["component"],
        weights=weights,
        port=args.port,
        open_browser=args.open,
    )


if __name__ == "__main__":
    main()
