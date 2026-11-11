"""PSF 构建脚本：逐步搭建。"""

import argparse
import gc
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from analysis import Issue, PsfSpec
from analysis._kernel import (
    _NM_TO_MM,
    ReferenceSphere,
    TraceSnapshot,
    integrate,
)
from analysis.psf import _check_sampling, _coverage, _na, _probe
from component import Sequential
from optimization.target import FOV, Target
from optimization.utils import load
from sampling import SampleOptions


DEFAULT_WAVELENGTHS_NM = tuple(float(value) for value in range(400, 721, 10))


def load_structure0(path: str | Path) -> tuple[Sequential, Target]:
    """加载 checkpoint，返回第 0 个结构及其目标参数。"""

    seq, target = load(path)
    return seq.select(0), target


def final_grid_size(grid: int) -> int:
    """计算最终提取边长，并保持与 grid 相同的奇偶性。"""

    if grid < 1:
        raise ValueError(f"grid must be positive, got {grid}")
    size = math.ceil(grid * math.sqrt(2))
    if size % 2 != grid % 2:
        size += 1
    return size


def max_half_field(fov: FOV) -> float:
    """从目标 FOV 配置计算最大半视场角（deg）。"""

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


def make_field_table(half_field: float, n_fields: int) -> tuple[float, ...]:
    """在 0 到最大半视场角之间均匀采样，包含两个端点。"""

    if not math.isfinite(half_field) or half_field <= 0:
        raise ValueError(f"half_field must be positive and finite, got {half_field}")
    if n_fields < 2:
        raise ValueError(f"n_fields must be >= 2, got {n_fields}")
    return tuple(half_field * index / (n_fields - 1) for index in range(n_fields))


def make_wavelength_table(
    wavelengths: Iterable[float] | None = None,
    n_wavelengths: int | None = None,
) -> tuple[float, ...]:
    """生成波长表：显式波长列表，或波长范围 + 个数；缺省匹配相机网格。"""

    if wavelengths is None:
        if n_wavelengths is not None:
            raise ValueError("n_wavelengths requires an explicit (start, stop) range")
        values = DEFAULT_WAVELENGTHS_NM
    elif n_wavelengths is None:
        values = tuple(float(wavelength) for wavelength in wavelengths)
    else:
        bounds = tuple(float(wavelength) for wavelength in wavelengths)
        if len(bounds) != 2:
            raise ValueError(f"wavelength range must contain (start, stop), got {bounds}")
        if n_wavelengths < 2:
            raise ValueError(f"n_wavelengths must be >= 2, got {n_wavelengths}")
        start, stop = bounds
        values = tuple(
            start + (stop - start) * index / (n_wavelengths - 1)
            for index in range(n_wavelengths)
        )
        values = (start, *values[1:-1], stop)

    if not values or any(not math.isfinite(w) or w <= 0 for w in values):
        raise ValueError(f"wavelengths must be non-empty, positive and finite: {values}")
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(f"wavelengths must be strictly increasing: {values}")
    return tuple(values)


def axis_rms_waves(
    seq: Sequential,
    wavelengths: Iterable[float] | None = None,
    n_wavelengths: int | None = None,
    pupil: int = 4096,
) -> torch.Tensor:
    """只做轴上追迹，返回每个波长的波前 RMS（waves），不计算 PSF。"""

    if pupil < 1:
        raise ValueError(f"pupil must be positive, got {pupil}")
    table = make_wavelength_table(wavelengths, n_wavelengths)
    spec = PsfSpec(
        field=(0.0, 0.0),
        wavelengths=table,
        pupil=SampleOptions(method="fibonacci", region="disk", count=pupil),
        size=9,
        dtype=torch.float64,
    )
    probe = _probe(seq, 0, spec)
    opd = probe.sp.opd[0, 0].double()
    alive = probe.sp.valid[0, 0]
    count = alive.sum(dim=-1)
    safe = torch.where(alive, opd, torch.zeros_like(opd))
    mean = safe.sum(dim=-1) / count.clamp_min(1)
    centered = torch.where(alive, safe - mean.unsqueeze(-1), torch.zeros_like(opd))
    rms = centered.square().sum(dim=-1).div(count.clamp_min(1)).sqrt()
    waves = probe.tr.wavelengths.double() * _NM_TO_MM
    rms = torch.where(count > 0, rms / waves, torch.full_like(rms, float("nan")))
    return rms.cpu()


def diffraction_scale(
    seq: Sequential,
    wavelengths: Iterable[float] | None = None,
    n_wavelengths: int | None = None,
    pupil: int = 4096,
    target_rms_waves: float = 0.075,
) -> float:
    """由轴上全波长最差 RMS 一步计算衍射极限 scale，不修改输入系统。"""

    if not math.isfinite(target_rms_waves) or target_rms_waves <= 0:
        raise ValueError(f"target_rms_waves must be positive and finite, got {target_rms_waves}")
    rms = axis_rms_waves(seq, wavelengths, n_wavelengths, pupil)
    if not torch.isfinite(rms).all():
        raise RuntimeError("axis wavefront RMS contains NaN/Inf; check for dead rays")
    reference = float(rms.max())
    if reference <= 0:
        raise ValueError("axis wavefront RMS is zero; diffraction scale is undefined")
    return target_rms_waves / reference


def sample_scales(
    s_diffraction: float,
    m: int,
    n: int,
    ratio: float = math.sqrt(2.0),
    jitter: float = 0.2,
    seed: int = 0,
) -> tuple[float, ...]:
    """生成 m+n+1 档分层几何 scale；中心和两个端点固定，内部可随机扰动。"""

    if not math.isfinite(s_diffraction) or s_diffraction <= 0:
        raise ValueError(f"s_diffraction must be positive and finite, got {s_diffraction}")
    if m < 0 or n < 1 or m >= n:
        raise ValueError(f"require 0 <= m < n, got m={m}, n={n}")
    if not math.isfinite(ratio) or ratio <= 1:
        raise ValueError(f"ratio must be greater than 1 and finite, got {ratio}")
    if not 0 <= jitter < 0.5:
        raise ValueError(f"jitter must be in [0, 0.5), got {jitter}")

    rng = random.Random(seed)
    scales = []
    for k in range(-m, n + 1):
        offset = 0.0 if k in (-m, 0, n) else rng.uniform(-jitter, jitter)
        scales.append(s_diffraction * ratio ** (k + offset))
    return tuple(scales)


def _wavefront_statistics(
    opd: torch.Tensor, alive: torch.Tensor, wavelengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(...,W,N) OPD → RMS/PTV（waves），仅使用存活光线。"""

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
    waves = wavelengths.double() * _NM_TO_MM
    shape = (1,) * (opd.ndim - 1) + (-1,)
    waves = waves.reshape(shape)
    dead = count == 0
    rms = torch.where(dead, torch.full_like(rms, float("nan")), rms) / waves
    ptv = torch.where(dead, torch.full_like(ptv, float("nan")), ptv) / waves
    return rms.cpu(), ptv.cpu()


def _concat_field_probes(probes: list[Any]) -> tuple[TraceSnapshot, ReferenceSphere]:
    """将 F=1 的独立追迹沿视场维拼接；各视场仍保留自己的 delta。"""

    def cat_tr(name: str) -> torch.Tensor:
        return torch.cat([getattr(probe.tr, name) for probe in probes], dim=1)

    def cat_sp(name: str) -> torch.Tensor:
        return torch.cat([getattr(probe.sp, name) for probe in probes], dim=1)

    first = probes[0].tr
    tr = TraceSnapshot(
        points=cat_tr("points"),
        dirs=cat_tr("dirs"),
        opl=cat_tr("opl"),
        n_img=cat_tr("n_img"),
        sensor_pts=cat_tr("sensor_pts"),
        sensor_tf=first.sensor_tf,
        alive=cat_tr("alive"),
        wavelengths=first.wavelengths,
    )
    sp = ReferenceSphere(
        points=cat_sp("points"),
        normals=cat_sp("normals"),
        opd=cat_sp("opd"),
        valid=cat_sp("valid"),
        centers_g=cat_sp("centers_g"),
        radius=cat_sp("radius"),
    )
    return tr, sp


def extract_field_batch(
    seq: Sequential,
    fields: tuple[float, ...],
    wavelengths: tuple[float, ...],
    grid: int,
    pupil: int,
    chunk_elems: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """一批径向视场合批积分；raw 为 (F,W,H,H)，不归一化、不裁剪。"""

    pupil_spec = SampleOptions(method="fibonacci", region="disk", count=pupil)
    probes = []
    deltas = []
    for theta in fields:
        probe = _probe(
            seq,
            0,
            PsfSpec(
                field=(theta, 0.0),
                wavelengths=wavelengths,
                pupil=pupil_spec,
                size=grid,
                delta=None,
                energy_threshold=0.0,
                dtype=dtype,
                chunk_elems=chunk_elems,
            ),
        )
        delta = float((probe.tr.wavelengths.min() * _NM_TO_MM) / (4.0 * probe.na.max()))
        if not math.isfinite(delta) or delta <= 0:
            raise RuntimeError(f"no alive rays at theta={theta}")
        probes.append(probe)
        deltas.append(delta)

    tr, sp = _concat_field_probes(probes)
    epd = probes[0].epd
    del probes, probe
    P, F, W, N = tr.points.shape[:4]
    device = tr.points.device
    centers_g = sp.centers_g.mean(dim=2)
    centers = tr.sensor_tf.transform_points(centers_g, inverse=True)[..., :2]

    # 每个视场使用自己单独探测得到的 Nyquist delta；批大小不改变物理网格。
    delta = torch.tensor(deltas, dtype=tr.points.dtype, device=device)
    axis = torch.arange(grid, dtype=tr.points.dtype, device=device)
    axis = (axis - (grid - 1) / 2)[None, :] * delta[:, None]
    qx = axis[:, :, None].expand(F, grid, grid)
    qy = axis[:, None, :].expand(F, grid, grid)
    q_local = torch.stack((qx, qy, torch.zeros_like(qx)), dim=-1)
    c3 = torch.cat((centers, torch.zeros_like(centers[..., :1])), dim=-1)
    q_local = q_local.unsqueeze(0) + c3.view(P, F, 1, 1, 3)
    points = tr.sensor_tf.transform_points(q_local)

    k0 = 2.0 * torch.pi / (tr.wavelengths * _NM_TO_MM)
    raw = integrate(sp, tr.dirs, tr.n_img, points, k0, chunk_elems, dtype=dtype)
    raw = raw / float(N) ** 2
    if not torch.isfinite(raw).all():
        raise RuntimeError("raw PSF contains NaN/Inf")

    delta_metric = delta.double().view(1, F, 1)
    coverage = _coverage(raw, delta_metric, epd, tr, sp)
    raw_sum = raw.sum(dim=(-2, -1)).double()
    window_energy = raw_sum * delta_metric.square()
    rms, ptv = _wavefront_statistics(sp.opd[0], sp.valid[0], tr.wavelengths)
    throughput = sp.valid.sum(dim=-1).double() / N
    na = _na(tr, sp)

    issues = list(_check_sampling(sp, tr.wavelengths * _NM_TO_MM)[1])
    edge = torch.zeros(grid, grid, dtype=torch.bool, device=device)
    edge[:2, :] = edge[-2:, :] = edge[:, :2] = edge[:, -2:] = True
    total = raw.sum(dim=(-2, -1))
    edge_fraction = raw.mul(edge).sum(dim=(-2, -1)) / total.clamp_min(1e-300)
    for p, f, w in (edge_fraction > 0.02).nonzero().tolist():
        issues.append((Issue.EDGE_TRUNCATION, p, f, w))

    return {
        "raw": raw[0].detach().cpu().contiguous(),
        "delta_mm": delta.double().cpu(),
        "center_xy_mm": centers[0].double().cpu(),
        "coverage": coverage[0].double().cpu(),
        "throughput": throughput[0].double().cpu(),
        "na": na[0].double().cpu(),
        "opd_rms_waves": rms,
        "opd_ptv_waves": ptv,
        "raw_sum": raw_sum[0].cpu(),
        "window_energy": window_energy[0].cpu(),
        "issues": [
            {"issue": Issue(int(issue)).name, "field": f, "wavelength": w}
            for issue, _p, f, w in issues
        ],
    }


def find_checkpoints(source: str | Path) -> list[Path]:
    """递归查找全部 .pth，跳过 _rejected；也接受单个 checkpoint。"""

    source = Path(source)
    if source.is_file():
        if source.suffix != ".pth":
            raise ValueError(f"input file must be .pth: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"source does not exist: {source}")
    checkpoints = sorted(
        path for path in source.rglob("*.pth") if "_rejected" not in path.parts
    )
    if not checkpoints:
        raise FileNotFoundError(f"no .pth checkpoints under {source}")
    return checkpoints


def checkpoint_lens_id(pth: str | Path, source: str | Path) -> str:
    """以相对目录作为镜头 ID；库根下的单个文件使用文件名。"""

    pth, source = Path(pth), Path(source)
    if source.is_file():
        return pth.parent.name
    relative = pth.parent.relative_to(source).as_posix()
    return pth.stem if relative == "." else relative


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def prepare_output(out: str | Path) -> Path:
    """创建 PSF 与逐镜头元数据目录。"""

    out = Path(out)
    (out / "psf").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    return out


def lens_output_paths(
    out: str | Path, lens_id: str, scale_index: int | None = None
) -> tuple[Path, Path]:
    """返回第 0 结构的 PSF 文件和逐镜头元数据路径。"""

    out, safe_id = Path(out), _safe_name(lens_id)
    suffix = "" if scale_index is None else f"__scale{scale_index:03d}"
    return (
        out / "psf" / f"{safe_id}__s000{suffix}.pt",
        out / "meta" / f"{safe_id}.jsonl",
    )


def rebuild_metadata(out: str | Path) -> Path:
    """将全部逐镜头 JSONL 按文件名顺序合并为 metadata.jsonl。"""

    out = Path(out)
    records = []
    for path in sorted((out / "meta").glob("*.jsonl")):
        records.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line)
    metadata = out / "metadata.jsonl"
    metadata.write_text(
        "\n".join(records) + ("\n" if records else ""), encoding="utf-8"
    )
    return metadata


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


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_save_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, default=str) for record in records)
        + ("\n" if records else ""),
        encoding="utf-8",
    )
    temporary.replace(path)


def _is_cuda_oom(error: Exception) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    text = str(error).lower()
    return isinstance(error, RuntimeError) and "cuda" in text and "out of memory" in text


def extract_scale(
    seq: Sequential,
    scale: float,
    fields: tuple[float, ...],
    wavelengths: tuple[float, ...],
    grid: int,
    pupil: int,
    field_batch_size: int,
    chunk_elems: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """提取一个 scale 的全部径向视场；OOM 只降低并行批次/chunk，不改变物理采样。"""

    if field_batch_size < 1 or chunk_elems < 1:
        raise ValueError("field_batch_size and chunk_elems must be positive")
    started = time.perf_counter()
    scaled = seq.scale(scale)
    n_fields, n_wavelengths = len(fields), len(wavelengths)
    raw = torch.empty((n_fields, n_wavelengths, grid, grid), dtype=dtype)
    metric_shape = (n_fields, n_wavelengths)
    metrics = {
        key: torch.empty(metric_shape, dtype=torch.float64)
        for key in (
            "coverage",
            "throughput",
            "na",
            "opd_rms_waves",
            "opd_ptv_waves",
            "raw_sum",
            "window_energy",
        )
    }
    delta = torch.empty(n_fields, dtype=torch.float64)
    centers = torch.empty((n_fields, 2), dtype=torch.float64)
    issues = []
    batches = []
    current_batch = min(field_batch_size, n_fields)
    current_chunk = chunk_elems
    start = 0

    while start < n_fields:
        stop = min(start + current_batch, n_fields)
        batch_started = time.perf_counter()
        try:
            result = extract_field_batch(
                scaled,
                fields[start:stop],
                wavelengths,
                grid,
                pupil,
                current_chunk,
                dtype,
            )
        except Exception as error:
            if not _is_cuda_oom(error):
                raise
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if current_batch > 1:
                current_batch = max(1, current_batch // 2)
                action = f"field_batch={current_batch}"
            elif current_chunk > 1:
                current_chunk = max(1, current_chunk // 4)
                action = f"chunk_elems={current_chunk}"
            else:
                raise RuntimeError("CUDA OOM even with one field and chunk_elems=1") from error
            print(
                f"[OOM retry] scale={scale:g}, fields={start}:{stop} → {action}",
                flush=True,
            )
            continue

        raw[start:stop] = result["raw"]
        delta[start:stop] = result["delta_mm"]
        centers[start:stop] = result["center_xy_mm"]
        for key in metrics:
            metrics[key][start:stop] = result[key]
        for item in result["issues"]:
            issues.append({**item, "field": item["field"] + start})
        batches.append(
            {
                "field_start": start,
                "field_stop": stop,
                "field_batch_size": stop - start,
                "chunk_elems": current_chunk,
                "seconds": time.perf_counter() - batch_started,
            }
        )
        print(
            f"[fields] scale={scale:g} {start}:{stop}/{n_fields} "
            f"batch={stop - start} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        start = stop

    if not torch.isfinite(raw).all():
        raise RuntimeError("assembled raw PSF contains NaN/Inf")
    return {
        "raw": raw,
        "delta_mm": delta,
        "center_xy_mm": centers,
        **metrics,
        "issues": issues,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "initial_field_batch_size": field_batch_size,
            "initial_chunk_elems": chunk_elems,
            "final_chunk_elems": current_chunk,
            "field_batches": batches,
        },
    }


def extract_lens(
    pth: str | Path,
    out: str | Path,
    *,
    lens_id: str | None = None,
    grid: int = 63,
    n_fields: int = 64,
    m: int = 1,
    n: int = 5,
    ratio: float = math.sqrt(2.0),
    jitter: float = 0.2,
    seed: int = 0,
    wavelengths: Iterable[float] | None = None,
    n_wavelengths: int | None = None,
    pupil: int = 65536,
    calibration_pupil: int = 4096,
    target_rms_waves: float = 0.075,
    field_batch_size: int = 4,
    chunk_elems: int = 64_000_000,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """提取一个 checkpoint 的第 0 结构，每个 scale 原子写入一个独立文件。"""

    started = time.perf_counter()
    pth, out = Path(pth), prepare_output(out)
    lens_id = lens_id or pth.parent.name
    seq, target = load_structure0(pth)
    seq = seq.to(device)
    fields = make_field_table(max_half_field(target.fov), n_fields)
    wavelength_table = make_wavelength_table(wavelengths, n_wavelengths)
    s_diffraction = diffraction_scale(
        seq,
        wavelength_table,
        pupil=calibration_pupil,
        target_rms_waves=target_rms_waves,
    )
    lens_seed = (
        seed + int.from_bytes(hashlib.sha256(lens_id.encode()).digest()[:8], "little")
    ) % (2**63)
    scales = sample_scales(s_diffraction, m, n, ratio, jitter, lens_seed)
    output_grid = final_grid_size(grid)
    stat = pth.stat()
    config = {
        "schema_version": 1,
        "source_pth": str(pth.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "git_commit": _git_commit(),
        "lens_id": lens_id,
        "structure": 0,
        "requested_grid": grid,
        "grid": output_grid,
        "n_fields": n_fields,
        "half_field_deg": fields[-1],
        "field_azimuth_deg": 0.0,
        "wavelengths_nm": list(wavelength_table),
        "pupil": pupil,
        "calibration_pupil": calibration_pupil,
        "target_rms_waves": target_rms_waves,
        "diffraction_scale": s_diffraction,
        "m": m,
        "n": n,
        "ratio": ratio,
        "jitter": jitter,
        "base_seed": seed,
        "lens_seed": lens_seed,
        "scales": list(scales),
        "dtype": str(dtype),
        "normalization": "emitter_count_only_raw_abs_U2",
        "target": target.to_dict(),
    }
    config_hash = _config_hash(config)
    print(
        f"[lens] {lens_id}: s_diffraction={s_diffraction:.6g}, "
        f"scales={len(scales)}, fields={n_fields}, W={len(wavelength_table)}, grid={output_grid}",
        flush=True,
    )

    summaries = []
    for scale_index, scale in enumerate(scales):
        path, meta_path = lens_output_paths(out, lens_id, scale_index)
        if path.is_file() and not overwrite:
            payload = torch.load(path, weights_only=True, map_location="cpu")
            if payload.get("config_hash") != config_hash:
                raise RuntimeError(f"incompatible existing output: {path}")
            summaries.append(payload["summary"])
            print(f"[skip] {path.name}", flush=True)
            continue

        result = extract_scale(
            seq,
            scale,
            fields,
            wavelength_table,
            output_grid,
            pupil,
            field_batch_size,
            chunk_elems,
            dtype,
        )
        shape = list(result["raw"].shape)
        summary = {
            "lens_id": lens_id,
            "structure": 0,
            "scale_index": scale_index,
            "scale_k": scale_index - m,
            "scale": scale,
            "file": f"psf/{path.name}",
            "shape": shape,
            "num_elements": result["raw"].numel(),
            "axis_rms_max_waves": float(result["opd_rms_waves"][0].max()),
            "predicted_axis_rms_waves": target_rms_waves * scale / s_diffraction,
            "coverage_min": float(result["coverage"].min()),
            "coverage_max": float(result["coverage"].max()),
            "seconds": result["runtime"]["seconds"],
            "config_hash": config_hash,
        }
        payload = {
            **result,
            "lens_id": lens_id,
            "structure": 0,
            "scale_index": scale_index,
            "scale_k": scale_index - m,
            "scale": scale,
            "diffraction_scale": s_diffraction,
            "scale_table": torch.tensor(scales, dtype=torch.float64),
            "theta_deg": torch.tensor(fields, dtype=torch.float64),
            "phi_deg": 0.0,
            "wavelengths_nm": torch.tensor(wavelength_table, dtype=torch.float64),
            "config": config,
            "config_hash": config_hash,
            "summary": summary,
            "normalization": "emitter_count_only_raw_abs_U2",
        }
        _atomic_save_torch(path, payload)
        summaries.append(summary)
        _atomic_write_jsonl(meta_path, summaries)
        print(f"[scale done] {path.name} shape={tuple(shape)}", flush=True)

    _atomic_write_jsonl(meta_path, summaries)
    print(f"[lens done] {lens_id}: {len(summaries)} scales in {time.perf_counter() - started:.1f}s")
    return summaries


def build_library(
    source: str | Path,
    out: str | Path,
    *,
    settings: dict[str, Any],
    limit_lenses: int | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """递归处理镜头库；单镜头失败记录下来并继续，最后统一报告。"""

    source, out = Path(source), prepare_output(out)
    checkpoints = find_checkpoints(source)
    if limit_lenses is not None:
        if limit_lenses < 1:
            raise ValueError("limit_lenses must be positive")
        checkpoints = checkpoints[:limit_lenses]
    jobs = [(path, checkpoint_lens_id(path, source)) for path in checkpoints]
    identities = {}
    for path, lens_id in jobs:
        if lens_id in identities:
            raise RuntimeError(f"duplicate lens_id {lens_id!r}: {identities[lens_id]} and {path}")
        identities[lens_id] = path

    batch_records = []
    batch_path = out / "batch_metadata.jsonl"
    for index, (path, lens_id) in enumerate(jobs, start=1):
        started = time.perf_counter()
        print(f"[library {index}/{len(jobs)}] {lens_id}", flush=True)
        try:
            summaries = extract_lens(
                path, out, lens_id=lens_id, overwrite=overwrite, **settings
            )
            record = {
                "lens_id": lens_id,
                "source_pth": str(path),
                "status": "complete",
                "scales": len(summaries),
                "raw_values": sum(item["num_elements"] for item in summaries),
            }
        except Exception as error:
            record = {
                "lens_id": lens_id,
                "source_pth": str(path),
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"[lens error] {lens_id}: {record['error']}", flush=True)
        record["seconds"] = time.perf_counter() - started
        batch_records.append(record)
        _atomic_write_jsonl(batch_path, batch_records)
        rebuild_metadata(out)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    failed = [record for record in batch_records if record["status"] == "error"]
    if failed:
        raise RuntimeError(f"{len(failed)} lenses failed; rerun to retry their incomplete scales")
    return batch_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="单个 .pth 或镜头库根目录")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=63)
    parser.add_argument("--n-fields", type=int, default=64)
    parser.add_argument("--m", type=int, default=1, help="小于衍射 scale 的档位数")
    parser.add_argument("--n", type=int, default=5, help="大于衍射 scale 的档位数")
    parser.add_argument("--ratio", type=float, default=math.sqrt(2.0))
    parser.add_argument("--jitter", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wavelengths", type=float, nargs="*", default=None)
    parser.add_argument("--n-wavelengths", type=int, default=None)
    parser.add_argument("--pupil", type=int, default=65536)
    parser.add_argument("--calibration-pupil", type=int, default=4096)
    parser.add_argument("--target-rms-waves", type=float, default=0.075)
    parser.add_argument("--field-batch-size", type=int, default=4)
    parser.add_argument("--chunk-elems", type=int, default=64_000_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--limit-lenses", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = {
        "grid": args.grid,
        "n_fields": args.n_fields,
        "m": args.m,
        "n": args.n,
        "ratio": args.ratio,
        "jitter": args.jitter,
        "seed": args.seed,
        "wavelengths": args.wavelengths,
        "n_wavelengths": args.n_wavelengths,
        "pupil": args.pupil,
        "calibration_pupil": args.calibration_pupil,
        "target_rms_waves": args.target_rms_waves,
        "field_batch_size": args.field_batch_size,
        "chunk_elems": args.chunk_elems,
        "device": args.device,
        "dtype": torch.float32 if args.dtype == "float32" else torch.float64,
    }
    build_library(
        args.source,
        args.out,
        settings=settings,
        limit_lenses=args.limit_lenses,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()