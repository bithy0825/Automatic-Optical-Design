"""校验 Zebase raw PSF 文件并统计旧版能量覆盖率阈值通过率。

只统计 psf/*.pt 最终结构文件；chunks 不纳入正式样本。逐文件加载，
不会一次将全部 raw 读入内存。输出 JSON 和阈值汇总 CSV，不改写原始数据。

用法::

    "$HOME/pyenv/Automatic-Optical-Design/bin/python" scripts/validate_zebase_raw.py
"""

import argparse
import csv
import json
import math
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

THRESHOLDS = (0.80, 0.90, 0.95, 0.97)
EXPECTED_SCALES = (1.0, 0.707, 0.5, 0.353, 0.25, 0.1767, 0.125)


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _check_payload(payload: dict[str, Any], path: Path, config_doc: dict[str, Any]):
    errors = []
    if "raw" not in payload:
        return ["missing raw"]
    raw = payload["raw"]
    if raw.ndim != 5 or raw.shape[1:] != (64, 33, 91, 91):
        errors.append(f"raw shape {tuple(raw.shape)} != (s,64,33,91,91)")
    if raw.dtype != torch.float32:
        errors.append(f"raw dtype {raw.dtype} != float32")
    if not torch.isfinite(raw).all():
        errors.append("raw contains NaN/Inf")
    s = raw.shape[0]
    if not 1 <= s <= 7:
        errors.append(f"scale count {s} outside 1..7")
    summary = payload.get("summary", {})
    config = payload.get("config", {})
    if summary.get("shape") != list(raw.shape):
        errors.append("summary shape mismatch")
    if summary.get("num_elements") != raw.numel():
        errors.append("summary num_elements mismatch")
    if payload.get("config_hash") != config_doc.get("config_hash"):
        errors.append("config hash mismatch with meta config")
    if summary.get("config_hash") != payload.get("config_hash"):
        errors.append("summary/payload config hash mismatch")
    if summary.get("file") != f"psf/{path.name}":
        errors.append("summary file path mismatch")
    if not Path(summary.get("source_pth", "")).is_file():
        errors.append("source checkpoint missing")
    if config.get("n_fields") != 64:
        errors.append("config n_fields != 64")
    if len(config.get("wavelengths_nm", [])) != 33:
        errors.append("config wavelength count != 33")
    if config.get("grid") != 91:
        errors.append("config grid != 91")
    if payload.get("scales").shape != (s,):
        errors.append("scales shape mismatch")
    else:
        scales = payload["scales"].tolist()
        if scales != list(EXPECTED_SCALES[:s]):
            errors.append(f"unexpected scales {scales}")
    for key in ("coverage", "throughput", "na", "opd_rms_waves", "opd_ptv_waves", "window_energy", "raw_sum"):
        if key not in payload or payload[key].shape != (s, 64, 33):
            errors.append(f"{key} shape mismatch")
    if payload["theta_deg"].shape != (64,):
        errors.append("theta_deg shape mismatch")
    if payload["wavelengths_nm"].shape != (33,):
        errors.append("wavelengths_nm shape mismatch")
    if payload["delta_mm"].shape != (s, 64):
        errors.append("delta_mm shape mismatch")
    if payload["center_xy_mm"].shape != (s, 64, 2):
        errors.append("center_xy_mm shape mismatch")
    if len(payload.get("field_records", [])) != s * 64:
        errors.append("field_records count mismatch")
    if len(payload.get("scale_diagnostics", [])) != s:
        errors.append("scale_diagnostics count mismatch")
    if payload.get("normalization") != "emitter_count_only_raw_abs_U2":
        errors.append("unexpected normalization label")
    return errors


def main() -> None:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=home / "workspace/Automatic-Optical-Design-PSF/Zebase")
    parser.add_argument("--lib", type=Path, default=home / "workspace/Automatic-Optical-Design-Lib/Zebase")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()
    source = args.source.resolve()
    out_json = args.out_json or source / "validation_zebase.json"
    out_csv = args.out_csv or source / "coverage_thresholds.csv"
    files = sorted((source / "psf").glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no final .pt files in {source / 'psf'}")

    started = time.perf_counter()
    plane_total = field_total = 0
    plane_pass = Counter()
    field_pass = Counter()
    structure_pass = Counter()
    inflated = low_throughput = 0
    coverage_values = []
    field_min_values = []
    invalid = []
    per_lens = []
    scale_counts = Counter()
    structure_picks = Counter()
    stop_reasons = Counter()
    total_raw_values = 0
    total_bytes = 0

    for index, path in enumerate(files, start=1):
        lens_id = path.name.split("__s", 1)[0]
        try:
            config_path = source / "meta" / f"{lens_id}.config.json"
            config_doc = json.loads(config_path.read_text(encoding="utf-8"))
            payload = torch.load(path, weights_only=True, map_location="cpu")
            errors = _check_payload(payload, path, config_doc)
        except Exception as exc:
            invalid.append({"file": path.name, "errors": [f"{type(exc).__name__}: {exc}"]})
            continue
        if errors:
            invalid.append({"file": path.name, "errors": errors})
            continue

        raw = payload["raw"]
        coverage = payload["coverage"].double()
        throughput = payload["throughput"].double()
        s = raw.shape[0]
        scale_counts[s] += 1
        structure_picks[payload["config"]["structure_picks"]] += 1
        stop_reasons[payload["scale_stop"]["reason"]] += 1
        total_raw_values += raw.numel()
        total_bytes += path.stat().st_size
        c = coverage.numpy()
        t = throughput.numpy()
        plane_total += c.size
        field_total += c.shape[0] * c.shape[1]
        coverage_values.append(c.reshape(-1))
        field_min_values.append(c.min(axis=1))
        inflated += int((c > 1.02).sum())
        low_throughput += int((t < 0.5).sum())
        for threshold in THRESHOLDS:
            plane_pass[threshold] += int((c >= threshold).sum())
            field_pass[threshold] += int((c.min(axis=1) >= threshold).sum())
            structure_pass[threshold] += int(bool((c >= threshold).all()))
        per_lens.append({
            "lens_id": lens_id,
            "source_pth": payload["summary"]["source_pth"],
            "structure": payload["summary"]["structure"],
            "shape": list(raw.shape),
            "scales": s,
            "coverage_min": float(c.min()),
            "coverage_max": float(c.max()),
            "raw_min": float(raw.min()),
            "raw_max": float(raw.max()),
        })
        if index % 25 == 0 or index == len(files):
            print(f"[validate] {index}/{len(files)} files", flush=True)
        del payload, raw, coverage, throughput

    coverage_all = np.concatenate(coverage_values)
    field_all = np.concatenate(field_min_values)
    library_files = sorted(args.lib.rglob("*.pth")) if args.lib.is_dir() else []
    completed_sources = {
        str(Path(item["source_pth"]).resolve())
        for item in per_lens
    }
    missing_library = [str(path) for path in library_files if str(path.resolve()) not in completed_sources]
    batch_path = source / "batch_metadata.jsonl"
    batch_rows = (
        [json.loads(line) for line in batch_path.read_text().splitlines() if line.strip()]
        if batch_path.is_file() else []
    )
    chunk_files = list((source / "chunks").rglob("*.pt")) if (source / "chunks").is_dir() else []
    threshold_report = {}
    for threshold in THRESHOLDS:
        threshold_report[str(threshold)] = {
            "plane_pass": plane_pass[threshold],
            "plane_total": plane_total,
            "plane_pass_fraction": plane_pass[threshold] / plane_total,
            "field_min_wavelength_pass": field_pass[threshold],
            "field_total": field_total,
            "field_min_wavelength_pass_fraction": field_pass[threshold] / field_total,
            "structure_all_fields_wavelengths_pass": structure_pass[threshold],
            "structure_total": len(per_lens),
            "structure_all_fields_wavelengths_pass_fraction": structure_pass[threshold] / len(per_lens),
        }

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "library": str(args.lib),
        "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "scope": "final psf/*.pt only; chunks excluded; raw not modified",
        "threshold_definition": {
            "plane": "each scale/field/wavelength coverage independently",
            "field": "minimum coverage over all 33 wavelengths, matching old min_w gate",
            "note": "old v2 tau_j is field-dependent random in [0.80,0.90]; 0.97 is a separate historical fixed threshold, not a universal old tau",
        },
        "file_validation": {
            "valid_files": len(per_lens),
            "invalid_files": len(invalid),
            "invalid_details": invalid,
            "all_raw_finite": not any("NaN/Inf" in e for item in invalid for e in item["errors"]),
            "total_raw_values": total_raw_values,
            "total_final_bytes": total_bytes,
            "scale_count_distribution": dict(sorted(scale_counts.items())),
            "structure_picks_distribution": dict(sorted(structure_picks.items())),
            "scale_stop_reason_distribution": dict(sorted(stop_reasons.items())),
        },
        "batch": {
            "library_checkpoint_count": len(library_files),
            "valid_final_structure_files": len(per_lens),
            "library_lens_with_valid_structure0": len({item["lens_id"] for item in per_lens}),
            "missing_library_checkpoints": missing_library,
            "batch_status_counts": dict(Counter(row["status"] for row in batch_rows)),
            "chunk_file_count_excluded": len(chunk_files),
            "note": "each current final file contains only structure 0; completeness is per configured structure_picks, not the earlier 3-structure target",
        },
        "coverage": {
            "plane_total": plane_total,
            "field_total": field_total,
            "min": float(coverage_all.min()),
            "max": float(coverage_all.max()),
            "mean": float(coverage_all.mean()),
            "median": float(np.median(coverage_all)),
            "p05": float(np.quantile(coverage_all, 0.05)),
            "p95": float(np.quantile(coverage_all, 0.95)),
            "coverage_inflated_over_1_02": inflated,
            "coverage_inflated_over_1_02_fraction": inflated / plane_total,
            "throughput_below_0_5": low_throughput,
            "throughput_below_0_5_fraction": low_throughput / plane_total,
            "thresholds": threshold_report,
        },
        "per_lens": per_lens,
        "elapsed_seconds": time.perf_counter() - started,
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("threshold", "plane_pass", "plane_total", "plane_percent", "field_min_w_pass", "field_total", "field_percent", "structure_pass", "structure_total", "structure_percent"))
        for threshold in THRESHOLDS:
            row = threshold_report[str(threshold)]
            writer.writerow((threshold, row["plane_pass"], row["plane_total"], 100 * row["plane_pass_fraction"], row["field_min_wavelength_pass"], row["field_total"], 100 * row["field_min_wavelength_pass_fraction"], row["structure_all_fields_wavelengths_pass"], row["structure_total"], 100 * row["structure_all_fields_wavelengths_pass_fraction"]))
    print(json.dumps({"valid_files": len(per_lens), "invalid_files": len(invalid), "plane_total": plane_total, "thresholds": threshold_report, "out_json": str(out_json), "out_csv": str(out_csv)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
