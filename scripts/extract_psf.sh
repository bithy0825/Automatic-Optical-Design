#!/usr/bin/env bash
# PSF 数据集批量提取：一行命令跑完全部剩余镜头类（断点续跑，可反复执行）。
#
#   ~/workspace/Automatic-Optical-Design/scripts/extract_psf.sh
#
# 已存在 meta/<镜头>.jsonl 的镜头自动跳过；重提某个镜头 = 删除其 meta jsonl
# 与 psf/<镜头>__*.pt 后再运行本脚本。额外参数（如 --lenses XXX）原样透传。
set -euo pipefail
PY="$HOME/pyenv/Automatic-Optical-Design/bin/python"
REPO="$HOME/workspace/Automatic-Optical-Design"
LIB="$HOME/workspace/Automatic-Optical-Design-Lib"
OUT="$HOME/workspace/Automatic-Optical-Design-PSF"
cd "$REPO"

for CLS in "Navigation/Photographic lenses - prime/zmx" "Navigation/Telescopes/zmx"; do
    echo "=== $CLS ==="
    "$PY" scripts/build_psf_dataset_v2.py \
        --lib "$LIB/$CLS" --out "$OUT/$CLS" --device cuda "$@"
done
