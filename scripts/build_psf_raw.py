"""从单个 checkpoint 或镜头库文件夹提取原始多波长径向 PSF。

输入可以是单个 .pth，也可以是递归包含多个 .pth 的文件夹。完整镜头会
直接跳过；未完成镜头按结构/scale/field chunk 断点续跑。批量状态写入
输出根目录的 ``batch_metadata.jsonl``，失败镜头不会阻断其他镜头。

流程：加载 .pth → 固定第 0 个结构，可选在前 32 个中 FPS 增选 →
每结构最多遍历 7 个显式 scale → 0..最大半视场角均匀采 64 个径向视场（phi=0）→
每个视场计算 W 个波长 → 保存每个结构的 ``raw[s,n,W,H,H]``。
默认 W=33、n=64、H=91、光瞳=65536。每档完成后，若全部视场和波长的
波前 RMS 最大值不超过 0.075λ，则该结构接近衍射极限并提前停止。

raw 始终保持 ``|U|²/N_emit²``：不做逐样本归一化、能量截断、覆盖率筛选、
负值截断或自动 scale/grow。coverage/throughput/NA/OPD 等仅作元数据保存。
默认 Cartesian 直评用于 phi=0 最终张量；``--extractor polar --save-polar``
使用极坐标提取并只长期保存 polar raw 与坐标。完整角向系数在使用时由
float64/complex128 FFT 重建，供后续无 Cartesian 插值的旋转使用。

断点续跑以 ``(structure,scale,field)`` chunk 为单元；结构最终张量原子写入。
已存在的最终结构文件会跳过。配置改变时不会混用旧 chunk，需换新输出目录
或显式 ``--overwrite``。

示例::

    PY="$HOME/pyenv/Automatic-Optical-Design/bin/python"
    "$PY" scripts/build_psf_raw.py <镜头.pth>

    # 小型端到端测试
    "$PY" scripts/build_psf_raw.py <镜头.pth> \
        --structure-picks 0 --scales 1.0 0.5 \
        --n-fields 2 --wavelengths 500 550 600 --pupil 4096 \
        --out outputs/raw_psf_e2e
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import (
    Issue,
    PolarPsfSpec,
    PsfSpec,
    psf,
    psf_polar,
    render_polar_psf,
)
from component import Sequential
from optimization.target import FOV, Target
from optimization.utils import build_sequential, build_target
from sampling import SampleOptions

DEFAULT_SCALE_TABLE = (1.0, 0.707, 0.5, 0.353, 0.25, 0.1767, 0.125)
DEFAULT_WAVELENGTHS_NM = tuple(float(value) for value in range(400, 721, 10))
DEFAULT_SCALE_STOP_RMS_WAVES = 0.075
_SCHEMA_VERSION = 2
_NM_TO_MM = 1e-6


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[Sequential, Target, dict[str, Any]]:
    """严格加载一个 .pth，返回 ``(population,target,original_config)``。"""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    cfg, state = torch.load(path, weights_only=True, map_location="cpu")
    target = build_target(cfg)
    seq = build_sequential(cfg, target)
    seq.load_state_dict(state, strict=True)
    return seq.to(device), target, cfg


def scale_system(seq: Sequential, factor: float) -> Sequential:
    """返回整个种群按同一倍率缩放后的新系统，不修改输入。"""

    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"scale factor must be positive and finite, got {factor}")
    return seq.scale(factor)


def scale_structure(
    seq: Sequential, structure: int, factor: float = 1.0
) -> Sequential:
    """选择并独立缩放一个结构，返回 ``P=1`` 新系统。"""

    if not 0 <= structure < seq.population:
        raise IndexError(f"structure must be in [0, {seq.population}), got {structure}")
    return scale_system(seq.select(structure), factor)


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def make_scale_table(scales: Iterable[float] | None = None) -> tuple[float, ...]:
    """缺省返回最多 7 档；自定义表从 1.0 开始严格递减且不超过 7 档。"""

    values = DEFAULT_SCALE_TABLE if scales is None else tuple(float(s) for s in scales)
    if not values:
        raise ValueError("scale table must be non-empty")
    if len(values) > 7:
        raise ValueError(f"scale table can contain at most 7 entries, got {len(values)}")
    if any(not math.isfinite(s) or s <= 0 for s in values):
        raise ValueError(f"all scales must be positive and finite, got {values}")
    if values[0] != 1.0 or any(b >= a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(f"scales must start at 1.0 and strictly decrease, got {values}")
    return tuple(values)


def make_wavelength_table(
    wavelengths: Iterable[float] | None = None,
) -> tuple[float, ...]:
    """公共多波长网格；缺省为 400:10:720 nm，共 33 个。"""

    values = (
        DEFAULT_WAVELENGTHS_NM
        if wavelengths is None
        else tuple(float(value) for value in wavelengths)
    )
    if not values or any(not math.isfinite(w) or w <= 0 for w in values):
        raise ValueError(f"wavelengths must be non-empty, positive and finite: {values}")
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(f"wavelengths must be strictly increasing: {values}")
    return tuple(values)


def max_half_field(fov: FOV) -> float:
    """训练 FOV 配置 → 最大半视场角（deg）。"""

    match fov:
        case float() as value:
            result = value / 2.0
        case (float() as a, float() as b):
            result = max(a, b) / 2.0
        case ((x0, x1), (y0, y1)):
            result = max(abs(x0), abs(x1), abs(y0), abs(y1))
        case _:
            raise TypeError(f"invalid fov: {fov!r}")
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"maximum half-field must be positive and finite: {result}")
    return result


def make_field_table(half_field: float, count: int = 64) -> tuple[float, ...]:
    """包含端点的 n 个物理径向视场角；方位角固定 phi=0。"""

    if not math.isfinite(half_field) or half_field <= 0:
        raise ValueError(f"half_field must be positive and finite, got {half_field}")
    if count < 2:
        raise ValueError(f"field count must be >= 2, got {count}")
    values = [half_field * index / (count - 1) for index in range(count)]
    values[0], values[-1] = 0.0, float(half_field)
    return tuple(values)


def _polar_spatial_shape(grid: int, angular_samples: int) -> tuple[int, int]:
    """奇数 H×H Cartesian 参考网格的唯一半径数 × 角向采样数。"""

    if grid < 9 or grid % 2 == 0:
        raise ValueError(f"polar grid must be odd and >=9, got {grid}")
    indices = torch.arange(-(grid // 2), grid // 2 + 1, dtype=torch.int64)
    radius2 = indices[:, None].square() + indices[None, :].square()
    return int(torch.unique(radius2).numel()), angular_samples


def iter_scaled_structures(
    seq: Sequential,
    structures: Iterable[int],
    scales: Iterable[float],
) -> Iterator[tuple[int, float, Sequential]]:
    """惰性遍历结构 → scale，每次只生成一个独立 P=1 系统。"""

    for structure in structures:
        for factor in scales:
            yield structure, factor, scale_structure(seq, structure, factor)


def _normalized_parameters(seq: Sequential, pool: int) -> torch.Tensor:
    """前 pool 个结构 → 去除常量维度的 min-max 归一化参数矩阵。"""

    population = seq.population
    if not 1 <= pool <= population:
        raise ValueError(f"pool must be in [1, {population}], got {pool}")
    columns = []
    for _name, tensor in sorted(seq.state_dict().items()):
        if tensor.ndim == 0 or tensor.shape[0] != population:
            continue
        values = tensor[:pool].detach().to(device="cpu", dtype=torch.float64)
        columns.append(values.reshape(pool, -1))
    if not columns:
        raise ValueError("checkpoint has no population-dependent parameters")
    matrix = torch.cat(columns, dim=1)
    if not torch.isfinite(matrix).all():
        raise ValueError("checkpoint parameters contain NaN/Inf")
    lo = matrix.min(dim=0).values
    span = matrix.max(dim=0).values - lo
    variable = span > 0
    if not bool(variable.any()):
        raise ValueError("all checkpoint parameters are constant across the population")
    return (matrix[:, variable] - lo[variable]) / span[variable]


def _fps(
    distances: torch.Tensor, candidates: list[int], seeds: list[int], count: int
) -> list[int]:
    """确定性地选择到所有已选结构最小距离最大的候选点。"""

    if count < 0:
        raise ValueError(f"FPS count must be nonnegative, got {count}")
    if count > len(candidates):
        raise ValueError(f"requested {count} structures from {len(candidates)} candidates")
    if count and not seeds:
        raise ValueError("FPS requires at least one seed")
    remaining = list(candidates)
    selected: list[int] = []
    reference = list(seeds)
    for _ in range(count):
        min_distance = distances[remaining][:, reference].min(dim=1).values
        chosen = int(min_distance.argmax())
        selected.append(remaining.pop(chosen))
        reference.append(selected[-1])
    return selected


def select_structures(
    seq: Sequential, *, candidate_pool: int = 32, picks: int = 0
) -> tuple[int, ...]:
    """固定第 0 个结构，再从索引 1..candidate_pool-1 中 FPS 选 picks 个。

    默认候选池恰为 0..31；``picks=0`` 时只返回 0 号，``picks=2`` 时
    共返回 3 个结构。参数归一化与距离矩阵也只使用该候选池，避免池外
    结构改变多样性距离。
    """

    population = seq.population
    if not 2 <= candidate_pool <= population:
        raise ValueError(
            f"candidate_pool must be in [2,{population}], got {candidate_pool}"
        )
    if not 0 <= picks <= candidate_pool - 1:
        raise ValueError(f"picks must be in [0,{candidate_pool - 1}], got {picks}")
    parameters = _normalized_parameters(seq, candidate_pool)
    distances = torch.cdist(parameters, parameters)
    selected = _fps(distances, list(range(1, candidate_pool)), [0], picks)
    return (0, *selected)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _atomic_save_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _config_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _opd_statistics(opd: torch.Tensor, alive: torch.Tensor, wavelengths: torch.Tensor):
    """(W,N) OPD/存活掩码 → 逐波长 RMS 和峰谷值（waves），恒 f64。"""

    opd = opd.double()
    alive = alive.bool()
    count = alive.sum(dim=-1)
    safe = torch.where(alive, opd, torch.zeros_like(opd))
    mean = safe.sum(dim=-1) / count.clamp_min(1)
    centered = torch.where(alive, safe - mean.unsqueeze(-1), torch.zeros_like(opd))
    rms = centered.square().sum(dim=-1).div(count.clamp_min(1)).sqrt()
    pos = torch.where(alive, opd, torch.full_like(opd, float("-inf")))
    neg = torch.where(alive, opd, torch.full_like(opd, float("inf")))
    ptv = pos.amax(dim=-1) - neg.amin(dim=-1)
    waves = wavelengths.double().to(opd.device) * _NM_TO_MM
    dead = count == 0
    rms = torch.where(dead, torch.full_like(rms, float("nan")), rms) / waves
    ptv = torch.where(dead, torch.full_like(ptv, float("nan")), ptv) / waves
    return rms.cpu(), ptv.cpu()


def extract_field(
    scaled: Sequential,
    *,
    structure: int,
    scale: float,
    theta_deg: float,
    wavelengths: tuple[float, ...],
    grid: int,
    pupil: int,
    extractor: Literal["cartesian", "polar"],
    angular_samples: int,
    integration_dtype: torch.dtype | None,
    storage_dtype: torch.dtype,
    chunk_elems: int,
    save_polar: bool,
) -> dict[str, Any]:
    """提取一个 ``(structure,scale,theta)`` 的 W 波长 raw PSF。"""

    started = time.perf_counter()
    pupil_spec = SampleOptions(method="fibonacci", region="disk", count=pupil)
    polar_payload: dict[str, Any] = {}
    if extractor == "cartesian":
        dtype = integration_dtype or (
            torch.float32 if scaled.device.type == "cuda" else torch.float64
        )
        rep = psf(
            scaled,
            PsfSpec(
                structures=0,
                field=(theta_deg, 0.0),
                wavelengths=wavelengths,
                pupil=pupil_spec,
                size=grid,
                delta=None,
                energy_threshold=0.0,
                dtype=dtype,
                chunk_elems=chunk_elems,
            ),
        )
        raw = rep.raw[0, 0]
        diagnostic_raw = raw
        opd, alive = rep.opd[0, 0], rep.alive[0, 0]
        coverage, throughput, na = rep.coverage[0, 0], rep.throughput[0, 0], rep.na[0, 0]
        centers, delta = rep.centers[0, 0], rep.delta
        issues = rep.issues
        actual_dtype = dtype
    else:
        rep = psf_polar(
            scaled,
            PolarPsfSpec(
                structure=0,
                field=(theta_deg, 0.0),
                wavelengths=wavelengths,
                pupil=pupil_spec,
                size=grid,
                delta=None,
                angular_samples=angular_samples,
                dtype=integration_dtype,
                chunk_elems=chunk_elems,
            ),
        )
        # polar 的交付数据是直接积分 raw；Cartesian 只在内存中用于诊断。
        raw = rep.raw
        diagnostic_raw = render_polar_psf(rep, 0.0)
        opd, alive = rep.opd, rep.alive
        coverage, throughput, na = rep.coverage, rep.throughput, rep.na
        centers, delta = rep.centers, rep.delta
        issues = rep.issues
        actual_dtype = rep.integration_dtype
        # 不保存 FFT 系数；使用时由 polar_coefficients() 以 complex128 重建。
        polar_payload = {
            "radii": rep.radii.cpu(),
            "beta": rep.beta.cpu(),
            "radius_index": rep.radius_index.cpu(),
            "alpha": rep.alpha.cpu(),
        }
    if not torch.isfinite(raw).all() or not torch.isfinite(diagnostic_raw).all():
        raise RuntimeError(f"raw PSF contains NaN/Inf at structure={structure}, theta={theta_deg}")
    rms_waves, ptv_waves = _opd_statistics(opd, alive, rep.wavelengths)
    raw_sum = diagnostic_raw.sum(dim=(-2, -1)).double().cpu()
    window_energy = raw_sum * delta * delta
    meta = {
        "structure": structure,
        "scale": scale,
        "theta_deg": theta_deg,
        "phi_deg": 0.0,
        "field_xy_deg": [theta_deg, 0.0],
        "wavelengths_nm": list(wavelengths),
        "grid": grid,
        "pupil_count": pupil,
        "pupil_method": "fibonacci",
        "extractor": extractor,
        "angular_samples": angular_samples if extractor == "polar" else None,
        "integration_dtype": str(actual_dtype),
        "storage_dtype": str(storage_dtype),
        "normalization": "emitter_count_only_raw_abs_U2",
        "data_layout": "polar_w_r_beta" if extractor == "polar" else "cartesian_w_x_y",
        "polar_storage": (
            "raw_only_fft_recomputed_on_use" if extractor == "polar" else None
        ),
        "energy_threshold": 0.0,
        "delta_mm": delta,
        "center_xy_mm": centers.double().cpu().tolist(),
        "coverage": coverage.double().cpu().tolist(),
        "throughput": throughput.double().cpu().tolist(),
        "na": na.double().cpu().tolist(),
        "opd_rms_waves": rms_waves.tolist(),
        "opd_ptv_waves": ptv_waves.tolist(),
        "raw_sum": raw_sum.tolist(),
        "window_energy": window_energy.tolist(),
        "issues": [
            {"issue": Issue(int(item[0])).name, "p": item[1], "f": item[2], "w": item[3]}
            for item in issues
        ],
        "seconds": time.perf_counter() - started,
    }
    return {
        "raw": raw.detach().to(device="cpu", dtype=storage_dtype).contiguous(),
        "meta": meta,
        **polar_payload,
    }


def _load_chunk(path: Path, config_hash: str, structure: int, scale: float, theta: float):
    chunk = torch.load(path, weights_only=True, map_location="cpu")
    meta = chunk["meta"]
    if (
        chunk.get("config_hash") != config_hash
        or meta["structure"] != structure
        or meta["scale"] != scale
        or meta["theta_deg"] != theta
    ):
        raise RuntimeError(f"stale/incompatible chunk: {path}; use --overwrite or a new --out")
    return chunk


def extract_structure(
    seq: Sequential,
    *,
    lens_id: str,
    source_pth: Path,
    structure: int,
    scales: tuple[float, ...],
    fields: tuple[float, ...],
    wavelengths: tuple[float, ...],
    out_lens: Path,
    config: dict[str, Any],
    config_hash: str,
    grid: int,
    pupil: int,
    extractor: Literal["cartesian", "polar"],
    angular_samples: int,
    integration_dtype: torch.dtype | None,
    storage_dtype: torch.dtype,
    chunk_elems: int,
    scale_stop_rms_waves: float,
    save_polar: bool,
    keep_chunks: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """提取一个结构；Cartesian 返回 raw，polar 返回 polar_raw，档数动态。"""

    started = time.perf_counter()
    stem = f"{_safe_name(lens_id)}__s{structure:03d}"
    final_path = out_lens / "psf" / f"{stem}.pt"
    if final_path.is_file() and not overwrite:
        payload = torch.load(final_path, weights_only=True, map_location="cpu")
        if payload.get("config_hash") != config_hash:
            raise RuntimeError(f"incompatible final tensor: {final_path}")
        print(f"[skip] {stem}: final tensor exists")
        return payload["summary"]

    spatial_shape = (
        _polar_spatial_shape(grid, angular_samples)
        if extractor == "polar"
        else (grid, grid)
    )
    shape = (len(scales), len(fields), len(wavelengths), *spatial_shape)
    raw = torch.empty(shape, dtype=storage_dtype)
    metric_shape = shape[:3]
    coverage = torch.empty(metric_shape, dtype=torch.float64)
    throughput = torch.empty_like(coverage)
    na = torch.empty_like(coverage)
    rms_waves = torch.empty_like(coverage)
    ptv_waves = torch.empty_like(coverage)
    window_energy = torch.empty_like(coverage)
    raw_sum = torch.empty_like(coverage)
    delta = torch.empty(shape[:2], dtype=torch.float64)
    centers = torch.empty((*shape[:2], 2), dtype=torch.float64)
    records = []
    chunk_paths = []
    scale_diagnostics = []
    polar_coordinates = None
    stopped_early = False
    stop_reason = "scale_table_exhausted"

    for si, scale in enumerate(scales):
        scale_started = time.perf_counter()
        scaled = scale_structure(seq, structure, scale)
        for fi, theta in enumerate(fields):
            chunk_path = (
                out_lens
                / "chunks"
                / _safe_name(lens_id)
                / f"{stem}__scale{si:02d}__field{fi:04d}.pt"
            )
            chunk_paths.append(chunk_path)
            if chunk_path.is_file() and not overwrite:
                chunk = _load_chunk(chunk_path, config_hash, structure, scale, theta)
            else:
                chunk = extract_field(
                    scaled,
                    structure=structure,
                    scale=scale,
                    theta_deg=theta,
                    wavelengths=wavelengths,
                    grid=grid,
                    pupil=pupil,
                    extractor=extractor,
                    angular_samples=angular_samples,
                    integration_dtype=integration_dtype,
                    storage_dtype=storage_dtype,
                    chunk_elems=chunk_elems,
                    save_polar=save_polar,
                )
                chunk["config_hash"] = config_hash
                _atomic_save_torch(chunk_path, chunk)
            meta = chunk["meta"]
            raw[si, fi] = chunk["raw"]
            coverage[si, fi] = torch.tensor(meta["coverage"], dtype=torch.float64)
            throughput[si, fi] = torch.tensor(meta["throughput"], dtype=torch.float64)
            na[si, fi] = torch.tensor(meta["na"], dtype=torch.float64)
            rms_waves[si, fi] = torch.tensor(meta["opd_rms_waves"], dtype=torch.float64)
            ptv_waves[si, fi] = torch.tensor(meta["opd_ptv_waves"], dtype=torch.float64)
            window_energy[si, fi] = torch.tensor(meta["window_energy"], dtype=torch.float64)
            raw_sum[si, fi] = torch.tensor(meta["raw_sum"], dtype=torch.float64)
            delta[si, fi] = meta["delta_mm"]
            centers[si, fi] = torch.tensor(meta["center_xy_mm"], dtype=torch.float64)
            records.append(meta)
            if extractor == "polar" and polar_coordinates is None:
                polar_coordinates = {
                    "radius_nodes_px": chunk["radii"].double() / meta["delta_mm"],
                    "beta": chunk["beta"],
                    "radius_index": chunk["radius_index"],
                    "alpha": chunk["alpha"],
                }
        current_rms = rms_waves[si]
        finite_rms = current_rms[torch.isfinite(current_rms)]
        max_rms = float(finite_rms.max()) if finite_rms.numel() else float("inf")
        median_rms = float(finite_rms.median()) if finite_rms.numel() else float("nan")
        reached = max_rms <= scale_stop_rms_waves
        scale_diagnostics.append(
            {
                "scale_index": si,
                "scale": scale,
                "n_fields": len(fields),
                "n_wavelengths": len(wavelengths),
                "opd_rms_max_waves": max_rms,
                "opd_rms_median_waves": median_rms,
                "diffraction_limit_reached": reached,
                "seconds": time.perf_counter() - scale_started,
            }
        )
        print(
            f"[scale done] {stem} s={scale:g} fields={len(fields)} "
            f"max_RMS={max_rms:.5f}λ elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if reached:
            stopped_early = si + 1 < len(scales)
            stop_reason = "diffraction_limit_reached"
            print(
                f"[scale stop] {stem}: max_RMS={max_rms:.5f}λ <= "
                f"{scale_stop_rms_waves:g}λ; saved_scales={si + 1}",
                flush=True,
            )
            break

    # 动态停止后只保留实际计算的档数，避免未初始化区域进入最终文件。
    n_saved = len(scale_diagnostics)
    saved_scales = scales[:n_saved]
    shape = (n_saved, len(fields), len(wavelengths), *spatial_shape)
    raw = raw[:n_saved].clone()
    coverage = coverage[:n_saved]
    throughput = throughput[:n_saved]
    na = na[:n_saved]
    rms_waves = rms_waves[:n_saved]
    ptv_waves = ptv_waves[:n_saved]
    window_energy = window_energy[:n_saved]
    raw_sum = raw_sum[:n_saved]
    delta = delta[:n_saved]
    centers = centers[:n_saved]
    scale_stop = {
        "criterion": "max_over_fields_and_wavelengths_opd_rms_waves",
        "threshold_waves": scale_stop_rms_waves,
        "stopped_early": stopped_early,
        "reason": stop_reason,
        "requested_scales": len(scales),
        "saved_scales": n_saved,
        "final_scale": saved_scales[-1],
        "final_max_rms_waves": scale_diagnostics[-1]["opd_rms_max_waves"],
    }

    summary = {
        "schema_version": _SCHEMA_VERSION,
        "lens_id": lens_id,
        "source_pth": str(source_pth),
        "structure": structure,
        "file": str(final_path.relative_to(out_lens)),
        "shape": list(shape),
        "num_elements": math.prod(shape),
        "scales": list(saved_scales),
        "n_scales_requested": len(scales),
        "n_scales_saved": n_saved,
        "scale_stopped_early": stopped_early,
        "scale_stop_reason": stop_reason,
        "scale_stop": scale_stop,
        "data_key": "polar_raw" if extractor == "polar" else "raw",
        "data_layout": "polar_w_r_beta" if extractor == "polar" else "cartesian_w_x_y",
        "n_fields": len(fields),
        "n_wavelengths": len(wavelengths),
        "coverage_min": float(coverage.min()),
        "coverage_max": float(coverage.max()),
        "seconds": time.perf_counter() - started,
        "config_hash": config_hash,
    }
    data_key = "polar_raw" if extractor == "polar" else "raw"
    payload = {
        data_key: raw,
        **(polar_coordinates or {}),
        "scales": torch.tensor(saved_scales, dtype=torch.float64),
        "theta_deg": torch.tensor(fields, dtype=torch.float64),
        "phi_deg": 0.0,
        "wavelengths_nm": torch.tensor(wavelengths, dtype=torch.float64),
        "coverage": coverage,
        "throughput": throughput,
        "na": na,
        "opd_rms_waves": rms_waves,
        "opd_ptv_waves": ptv_waves,
        "window_energy": window_energy,
        "raw_sum": raw_sum,
        "delta_mm": delta,
        "center_xy_mm": centers,
        "field_records": records,
        "scale_diagnostics": scale_diagnostics,
        "scale_stop": scale_stop,
        "config": config,
        "config_hash": config_hash,
        "summary": summary,
        "normalization": "emitter_count_only_raw_abs_U2",
    }
    _atomic_save_torch(final_path, payload)
    if not keep_chunks:
        for path in chunk_paths:
            path.unlink(missing_ok=True)
    print(f"[structure done] {stem} shape={tuple(shape)} → {final_path}", flush=True)
    return summary


def find_checkpoints(library: Path) -> list[Path]:
    """递归枚举镜头库中的 .pth，跳过 _rejected，路径排序保证确定性。"""

    if not library.is_dir():
        raise NotADirectoryError(f"library is not a directory: {library}")
    return sorted(
        path for path in library.rglob("*.pth") if "_rejected" not in path.parts
    )


def _checkpoint_lens_id(pth: Path, library: Path) -> str:
    """相对父目录作为镜头 ID；直接位于库根时使用文件名防止 '.' 冲突。"""

    relative = pth.parent.relative_to(library).as_posix()
    return pth.stem if relative == "." else relative


def _completed_lens_records(
    out_lens: Path, pth: Path, lens_id: str, settings: dict[str, Any]
) -> list[dict[str, Any]] | None:
    """无需加载 checkpoint，判断一个镜头的全部结构是否已按当前配置完成。"""

    config_path = out_lens / "meta" / f"{_safe_name(lens_id)}.config.json"
    manifest_path = out_lens / "meta" / f"{_safe_name(lens_id)}.jsonl"
    if not config_path.is_file() or not manifest_path.is_file():
        return None
    try:
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        config = saved["config"]
        stat = pth.stat()
        expected = {
            "schema_version": _SCHEMA_VERSION,
            "source_pth": str(pth.resolve()),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "git_commit": _git_commit(),
            "candidate_pool": settings["candidate_pool"],
            "structure_picks": settings["structure_picks"],
            "scales": list(settings["scales"]),
            "n_fields": settings["n_fields"],
            "wavelengths_nm": list(settings["wavelengths"]),
            "grid": settings["grid"],
            "pupil": settings["pupil"],
            "extractor": settings["extractor"],
            "angular_samples": (
                settings["angular_samples"] if settings["extractor"] == "polar" else None
            ),
            "integration_dtype": str(settings["integration_dtype"]),
            "storage_dtype": str(settings["storage_dtype"]),
            "chunk_elems": settings["chunk_elems"],
            "scale_stop_rms_waves": settings["scale_stop_rms_waves"],
            "save_polar": settings["save_polar"],
        }
        if any(config.get(key) != value for key, value in expected.items()):
            return None
        records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected = config["selected_structures"]
        if len(records) != len(selected) or {r["structure"] for r in records} != set(selected):
            return None
        for record in records:
            if record["config_hash"] != saved["config_hash"]:
                return None
            if not (out_lens / record["file"]).is_file():
                return None
            if record["n_fields"] != settings["n_fields"]:
                return None
            if record["n_wavelengths"] != len(settings["wavelengths"]):
                return None
        return records
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def build_raw_dataset(
    pth: Path,
    out: Path,
    *,
    device: str,
    candidate_pool: int,
    structure_picks: int,
    scales: tuple[float, ...],
    n_fields: int,
    wavelengths: tuple[float, ...],
    grid: int,
    pupil: int,
    extractor: Literal["cartesian", "polar"],
    angular_samples: int,
    integration_dtype: torch.dtype | None,
    storage_dtype: torch.dtype,
    chunk_elems: int,
    scale_stop_rms_waves: float,
    save_polar: bool,
    keep_chunks: bool,
    overwrite: bool,
    lens_id: str | None = None,
) -> list[dict[str, Any]]:
    """单镜头完整提取入口。"""

    seq, target, source_config = load_checkpoint(pth, device=device)
    population = seq.population
    selected = select_structures(
        seq, candidate_pool=candidate_pool, picks=structure_picks
    )
    half_field = max_half_field(target.fov)
    fields = make_field_table(half_field, n_fields)
    lens_id = lens_id or pth.parent.name
    out_lens = out  # 沿用旧版：psf/、meta/、chunks/ 直接位于类别输出根目录
    (out_lens / "psf").mkdir(parents=True, exist_ok=True)
    (out_lens / "meta").mkdir(parents=True, exist_ok=True)
    stat = pth.stat()
    config = {
        "schema_version": _SCHEMA_VERSION,
        "source_pth": str(pth.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "git_commit": _git_commit(),
        "selected_structures": list(selected),
        "candidate_pool": candidate_pool,
        "structure_picks": structure_picks,
        "scales": list(scales),
        "n_fields": n_fields,
        "half_field_deg": half_field,
        "field_azimuth_deg": 0.0,
        "wavelengths_nm": list(wavelengths),
        "grid": grid,
        "pupil": pupil,
        "pupil_method": "fibonacci",
        "extractor": extractor,
        "angular_samples": angular_samples if extractor == "polar" else None,
        "integration_dtype": str(integration_dtype),
        "storage_dtype": str(storage_dtype),
        "chunk_elems": chunk_elems,
        "scale_stop_rms_waves": scale_stop_rms_waves,
        "scale_stop_criterion": "max_over_fields_and_wavelengths_opd_rms_waves",
        "save_polar": save_polar,
        "normalization": "emitter_count_only_raw_abs_U2",
        "energy_threshold": 0.0,
    }
    config_hash = _config_hash(config)
    config_path = out_lens / "meta" / f"{_safe_name(lens_id)}.config.json"
    if config_path.is_file() and not overwrite:
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("config_hash") != config_hash:
            raise RuntimeError(f"output config differs: {config_path}; use a new --out or --overwrite")
    _atomic_write_json(
        config_path,
        {"config_hash": config_hash, "config": config, "source_config": source_config},
    )

    print(
        f"[plan] lens={lens_id} structures={list(selected)} scales={len(scales)} "
        f"fields={n_fields} W={len(wavelengths)} grid={grid} pupil={pupil} "
        f"extractor={extractor} out={out_lens}",
        flush=True,
    )
    summaries = []
    for structure in selected:
        summaries.append(
            extract_structure(
                seq,
                lens_id=lens_id,
                source_pth=pth,
                structure=structure,
                scales=scales,
                fields=fields,
                wavelengths=wavelengths,
                out_lens=out_lens,
                config=config,
                config_hash=config_hash,
                grid=grid,
                pupil=pupil,
                extractor=extractor,
                angular_samples=angular_samples,
                integration_dtype=integration_dtype,
                storage_dtype=storage_dtype,
                chunk_elems=chunk_elems,
                scale_stop_rms_waves=scale_stop_rms_waves,
                save_polar=save_polar,
                keep_chunks=keep_chunks,
                overwrite=overwrite,
            )
        )
    lines = [json.dumps(item, ensure_ascii=False, default=str) for item in summaries]
    manifest_path = out_lens / "meta" / f"{_safe_name(lens_id)}.jsonl"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(
        f"[all] {len(summaries)} structures; "
        f"{sum(item['num_elements'] for item in summaries)} raw values → {out_lens}",
        flush=True,
    )
    return summaries


def _rebuild_library_metadata(out: Path) -> None:
    """合并旧版布局中 meta/*.jsonl 为类别根目录的 metadata.jsonl。"""

    records = []
    for path in sorted((out / "meta").glob("*.jsonl")):
        records.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    temporary = out / "metadata.jsonl.tmp"
    temporary.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")
    temporary.replace(out / "metadata.jsonl")


def build_raw_library(
    source: Path,
    out: Path,
    *,
    settings: dict[str, Any],
    keep_chunks: bool,
    overwrite: bool,
    limit_lenses: int | None = None,
) -> list[dict[str, Any]]:
    """单文件或镜头库批量入口；完整镜头直接过滤，失败镜头保留待重试。"""

    source = source.resolve()
    if source.is_file():
        if source.suffix != ".pth":
            raise ValueError(f"input file must be .pth: {source}")
        jobs = [(source, source.parent.name)]
    else:
        checkpoints = find_checkpoints(source)
        if not checkpoints:
            raise FileNotFoundError(f"no .pth checkpoints under {source}")
        jobs = [(path, _checkpoint_lens_id(path, source)) for path in checkpoints]
    if limit_lenses is not None:
        if limit_lenses < 1:
            raise ValueError(f"limit_lenses must be >=1, got {limit_lenses}")
        jobs = jobs[:limit_lenses]
    identities = {}
    for pth, lens_id in jobs:
        if lens_id in identities:
            raise RuntimeError(
                f"multiple checkpoints map to lens_id={lens_id!r}: "
                f"{identities[lens_id]} and {pth}"
            )
        identities[lens_id] = pth

    out.mkdir(parents=True, exist_ok=True)
    batch_path = out / "batch_metadata.jsonl"
    batch_records = []
    print(f"[library] checkpoints={len(jobs)} out={out}", flush=True)
    for index, (pth, lens_id) in enumerate(jobs, start=1):
        started = time.perf_counter()
        out_lens = out
        error_text = None
        completed = (
            None
            if overwrite
            else _completed_lens_records(out_lens, pth, lens_id, settings)
        )
        if completed is not None:
            status = "skipped"
            summaries = completed
            print(f"[lens {index}/{len(jobs)}] {lens_id}: already complete", flush=True)
        else:
            print(f"[lens {index}/{len(jobs)}] {lens_id}: extracting", flush=True)
            try:
                summaries = build_raw_dataset(
                    pth,
                    out,
                    **settings,
                    keep_chunks=keep_chunks,
                    overwrite=overwrite,
                    lens_id=lens_id,
                )
                status = "complete"
            except Exception as exc:  # 单镜头失败不阻断后续镜头，下次运行会重试
                summaries = []
                status = "error"
                error_text = f"{type(exc).__name__}: {exc}"
                print(
                    f"[lens error] {lens_id}: {error_text}",
                    flush=True,
                )
        record = {
            "source_pth": str(pth),
            "lens_id": lens_id,
            "status": status,
            "structures": len(summaries),
            "raw_values": sum(item.get("num_elements", 0) for item in summaries),
            "seconds": time.perf_counter() - started,
        }
        if status == "error":
            record["error"] = error_text
        batch_records.append(record)
        temporary = batch_path.with_name(batch_path.name + ".tmp")
        temporary.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in batch_records) + "\n",
            encoding="utf-8",
        )
        temporary.replace(batch_path)
        if (out / "meta").is_dir():
            _rebuild_library_metadata(out)

    failed = [record for record in batch_records if record["status"] == "error"]
    print(
        f"[library done] total={len(batch_records)} "
        f"complete={sum(r['status'] == 'complete' for r in batch_records)} "
        f"skipped={sum(r['status'] == 'skipped' for r in batch_records)} "
        f"errors={len(failed)}",
        flush=True,
    )
    if failed:
        raise RuntimeError(
            f"{len(failed)} lenses failed; fix inputs and rerun to retry only incomplete lenses"
        )
    return batch_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="一个 .pth，或包含多个 .pth 的镜头库文件夹")
    parser.add_argument("--out", type=Path, default=Path.home() / "workspace/Automatic-Optical-Design-PSF-Raw")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--candidate-pool", type=int, default=32, help="结构候选池：0..N-1")
    parser.add_argument("--structure-picks", type=int, default=0, help="除固定0号外的FPS选取数")
    parser.add_argument("--limit-lenses", type=int, default=None, help="只处理排序后的前N个pth，用于试提取")
    parser.add_argument("--scales", type=float, nargs="+", default=None)
    parser.add_argument("--n-fields", type=int, default=64)
    parser.add_argument("--wavelengths", type=float, nargs="+", default=None)
    parser.add_argument("--grid", type=int, default=91)
    parser.add_argument("--pupil", type=int, default=65536)
    parser.add_argument("--extractor", choices=("cartesian", "polar"), default="cartesian")
    parser.add_argument("--angular-samples", type=int, default=257)
    parser.add_argument("--dtype", choices=("auto", "float32", "float64"), default="auto")
    parser.add_argument("--storage-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--chunk-elems", type=int, default=64_000_000)
    parser.add_argument(
        "--scale-stop-rms-waves",
        type=float,
        default=DEFAULT_SCALE_STOP_RMS_WAVES,
        help="每档全部视场/波长的最大波前RMS不超过此值即提前停止；默认0.075",
    )
    parser.add_argument("--save-polar", action="store_true")
    parser.add_argument("--keep-chunks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.grid < 9 or args.pupil < 1 or args.chunk_elems < 1:
        parser.error("grid must be >=9; pupil and chunk-elems must be positive")
    if args.save_polar and args.extractor != "polar":
        parser.error("--save-polar requires --extractor polar")
    if not math.isfinite(args.scale_stop_rms_waves) or args.scale_stop_rms_waves <= 0:
        parser.error("--scale-stop-rms-waves must be positive and finite")

    integration_dtype = {
        "auto": None,
        "float32": torch.float32,
        "float64": torch.float64,
    }[args.dtype]
    settings = {
        "device": args.device,
        "candidate_pool": args.candidate_pool,
        "structure_picks": args.structure_picks,
        "scales": make_scale_table(args.scales),
        "n_fields": args.n_fields,
        "wavelengths": make_wavelength_table(args.wavelengths),
        "grid": args.grid,
        "pupil": args.pupil,
        "extractor": args.extractor,
        "angular_samples": args.angular_samples,
        "integration_dtype": integration_dtype,
        "storage_dtype": torch.float32 if args.storage_dtype == "float32" else torch.float64,
        "chunk_elems": args.chunk_elems,
        "scale_stop_rms_waves": args.scale_stop_rms_waves,
        "save_polar": args.save_polar,
    }
    build_raw_library(
        args.source,
        args.out,
        settings=settings,
        keep_chunks=args.keep_chunks,
        overwrite=args.overwrite,
        limit_lenses=args.limit_lenses,
    )


if __name__ == "__main__":
    main()
