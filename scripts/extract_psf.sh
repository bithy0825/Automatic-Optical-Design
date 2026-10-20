#!/usr/bin/env bash
# PSF 数据集批量提取：一行命令跑完全部镜头类（断点续跑，可反复执行）。
#
#   ~/workspace/Automatic-Optical-Design/scripts/extract_psf.sh
#
# 已存在 meta/<镜头>.jsonl 的镜头自动跳过；重提某个镜头 = 删除其 meta jsonl
# 与 psf/<镜头>__*.pt 后再运行本脚本。额外参数（如 --lenses XXX）原样透传。
# 覆盖：6 个已提取类（重提 τ0.97 时代不满 232 的镜头）+ Telescopes。
set -euo pipefail
PY="$HOME/pyenv/Automatic-Optical-Design/bin/python"
REPO="$HOME/workspace/Automatic-Optical-Design"
LIB="$HOME/workspace/Automatic-Optical-Design-Lib"
OUT="$HOME/workspace/Automatic-Optical-Design-PSF"
cd "$REPO"

# (lib 子路径, out 子路径)；Microscope 历史上 lib/out 同用类根（lens_id 含 zmx/ 前缀）
PAIRS=(
  "Zebase|Zebase"
  "Navigation/Eyepieces/len|Navigation/Eyepieces/len"
  "Navigation/Eyepieces/zmx|Navigation/Eyepieces/zmx"
  "Navigation/Microscope objectives|Navigation/Microscope objectives"
  "Navigation/Photographic lenses - prime/len|Navigation/Photographic lenses - prime/len"
  "Navigation/Photographic lenses - prime/zmx|Navigation/Photographic lenses - prime/zmx"
  "Navigation/Telescopes/zmx|Navigation/Telescopes/zmx"
)
for pair in "${PAIRS[@]}"; do
    L="${pair%%|*}"; O="${pair##*|}"
    echo "=== $L ==="
    "$PY" scripts/build_psf_dataset_v2.py \
        --lib "$LIB/$L" --out "$OUT/$O" --device cuda "$@"
done
