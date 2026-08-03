"""训练入口：TOML 配置驱动的光学系统优化。

    python train.py CONFIG [--device D] [--seed N] [--output PATH] [--resume PATH]
                           [--save-every N] [--history PATH] [--no-progress]
                           [--set KEY=VALUE]...

光学系统 / GA / 优化阶段 / 损失权重全部来自配置文件（结构见 demo/config.toml）；
运行控制参数写在配置的 ``[train]`` 节，命令行同名选项临时覆盖::

    [train]
    seed = 0                  # 缺省不固定种子
    device = "auto"           # auto/cuda/cpu，缺省 auto
    output = "runs/demo.pth"  # 缺省与配置同名 .pth
    resume = "ckpt.pth"       # 缺省不续训
    save_every = 10           # 每 N 代滚动存档，缺省仅训完存一次
    history = "loss.json"     # 缺省不导出

配置内相对路径相对配置文件目录解析，CLI 路径相对 CWD。
``--set`` 覆盖任意配置字段，值按 TOML 解析，数字路径段下钻列表::

    --set ga.generation=50 --set optimizer.0.step=100 --set target.F=2.8
"""

import argparse
import json
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from component import Sequential
from core import term
from core.noun import Noun
from optimization import (
    GAOptions,
    GeneticAlgorithm,
    LossHistory,
    LossWeights,
    PeriodicSaver,
    ProgressBar,
    Target,
    build_sequential,
    build_stage,
    build_target,
    load_config,
    save,
    total_loss,
)


def apply_override(cfg: dict[str, Any], expr: str) -> None:
    """``KEY=VALUE``：点分路径下钻（数字段进列表），值按 TOML 解析，缺层自动建表。"""
    key, sep, raw = expr.partition("=")
    if not sep:
        raise ValueError(f"--set 需要 KEY=VALUE 形式: {expr!r}")
    value = tomllib.loads(f"x = {raw}")["x"]
    *parents, leaf = key.split(".")
    node: Any = cfg
    for part in parents:
        node = node[int(part)] if isinstance(node, list) else node.setdefault(part, {})
    if isinstance(node, list):
        node[int(leaf)] = value
    else:
        node[leaf] = value


@dataclass(slots=True)
class RunConfig:
    """运行控制：``[train]`` 节经 CLI 覆盖后的解析结果（路径均已绝对化）。"""

    seed: int | None
    device: str
    output: Path
    resume: Path | None
    save_every: int | None
    history: Path | None
    progress: bool


def resolve_run(
    cfg: Mapping[str, Any], args: argparse.Namespace, config_path: Path
) -> RunConfig:
    """合并 ``[train]`` 节与 CLI（CLI 优先）。

    相对路径：配置值相对配置文件目录，CLI 值相对 CWD。
    """
    table: Mapping[str, Any] = term.TRAIN.resolve(cfg, default={})

    def pick(cli: Any, noun: Noun) -> Any:
        return cli if cli is not None else noun.resolve(table, default=None)

    def path(cli: str | None, noun: Noun) -> Path | None:
        value, base = (
            (cli, Path.cwd())
            if cli is not None
            else (noun.resolve(table, default=None), config_path.parent)
        )
        if value is None:
            return None
        p = Path(value)
        return p if p.is_absolute() else base / p

    device = pick(args.device, term.DEVICE) or "auto"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = pick(args.seed, term.SEED)
    save_every = pick(args.save_every, term.SAVE_EVERY)
    if save_every is not None and save_every < 1:
        raise ValueError(f"save_every must be >= 1, got {save_every}")

    return RunConfig(
        seed=None if seed is None else int(seed),
        device=device,
        output=path(args.output, term.OUTPUT) or config_path.with_suffix(".pth"),
        resume=path(args.resume, term.RESUME),
        save_every=save_every,
        history=path(args.history, term.HISTORY),
        progress=args.progress,
    )


def report(
    seq: Sequential,
    target: Target,
    blocks: Sequence[Mapping[str, Any]],
    weights: LossWeights,
    output: Path,
    elapsed: float,
) -> None:
    """终训报告：最优个体（末代已排序，第 0 行）各损失分项 + 种群均值 + 耗时。"""
    with torch.no_grad():
        flow = seq()
        total, parts = total_loss(flow, seq, target, blocks, weights)
    best = "  ".join(f"{k}={v[0].item():.4g}" for k, v in parts.items())
    print("\n── 训练完成 " + "─" * 40)
    print(f"最优个体:  loss={total[0].item():.4g}  ({best})")
    print(f"种群均值:  loss={total.mean().item():.4g}")
    print(f"耗时:      {elapsed:.1f}s")
    print(f"检查点:    {output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train",
        description="TOML 配置驱动的光学系统优化"
        "（结构见 demo/config.toml；运行控制见 [train] 节）",
    )
    p.add_argument("config", help="配置 TOML 路径")
    p.add_argument("--device", help="训练设备：auto/cuda/cpu…（缺省 auto）")
    p.add_argument("--seed", type=int, help="随机种子（缺省不固定）")
    p.add_argument("--output", help="检查点输出路径（缺省与配置同名 .pth）")
    p.add_argument("--resume", help="从检查点注入参数续训（strict，结构须一致）")
    p.add_argument("--save-every", type=int, help="每 N 代滚动保存检查点")
    p.add_argument("--history", help="损失历史导出 JSON 路径")
    p.add_argument(
        "--no-progress", dest="progress", action="store_false", help="关闭进度条"
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖配置任意字段（值按 TOML 解析；数字段下钻列表），可重复",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))
    for expr in args.overrides:
        apply_override(cfg, expr)
    run = resolve_run(cfg, args, config_path)

    if run.seed is not None:
        torch.manual_seed(run.seed)

    target = build_target(cfg)
    seq = build_sequential(cfg, target).to(run.device)
    if run.resume is not None:
        _saved_cfg, state = torch.load(
            run.resume, weights_only=True, map_location="cpu"
        )
        seq.load_state_dict(state, strict=True)

    blocks = term.COMPONENT.resolve(cfg)
    weights = LossWeights.from_options(cfg)
    optimizer_blocks = term.OPTIMIZER.resolve(cfg, default=[])
    stages = [build_stage(b) for b in optimizer_blocks]
    ga = GeneticAlgorithm(GAOptions.from_options(term.GA.resolve(cfg)), stages)

    o = ga.options
    names = [term.TYPE.resolve(b) for b in optimizer_blocks]
    print(f"config: {config_path}")
    print(target)
    print(seq)
    print(
        f"device: {run.device} | ga: P={o.population} gen={o.generation} "
        f"topk={o.topk if o.topk is not None else 'auto'} | stages: {names or '无(纯 GA)'}"
    )

    history = LossHistory()
    callbacks: list[Any] = [history]
    if run.progress:
        callbacks.append(
            ProgressBar(o.generation, max((s.options.step for s in stages), default=0))
        )
    if run.save_every is not None:
        callbacks.append(PeriodicSaver(seq, cfg, run.output, run.save_every))

    t0 = time.perf_counter()
    try:
        ga.run(seq, target, blocks, weights, callbacks=callbacks)
    finally:
        for cb in callbacks:
            if isinstance(cb, ProgressBar):
                cb.close()
    elapsed = time.perf_counter() - t0

    run.output.parent.mkdir(parents=True, exist_ok=True)
    save(seq, cfg, run.output)
    report(seq, target, blocks, weights, run.output, elapsed)

    if run.history is not None:
        run.history.parent.mkdir(parents=True, exist_ok=True)
        run.history.write_text(json.dumps(history.records, ensure_ascii=False, indent=2))
        print(f"历史:      {run.history}")


if __name__ == "__main__":
    main()
