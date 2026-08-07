#!/usr/bin/env bash
# 批量转换目录下的 .zmx 为训练配置。
# 跨平台:Linux bash 直接运行;Windows 用 Git Bash 或 PowerShell 中 bash 调用。
#
# 用法:
#   bash scripts/zmx_batch.sh <输入目录> <输出目录>
#
# 每个可转换文件生成 <输出目录>/<文件名>/config.toml;被过滤的文件不产出,
# 其完整路径与原因汇总写入 <输出目录>/skipped.txt(本次运行无过滤则删除旧文件)。
set -u

if [ $# -ne 2 ]; then
  cat <<'EOF'
用法: bash scripts/zmx_batch.sh <输入目录> <输出目录>
示例:
  bash scripts/zmx_batch.sh "E:/Users/资源/ZEBASE镜头库 目录/ZEBASE镜头库+目录" out/zmx
  bash scripts/zmx_batch.sh /data/lenses/zmx /data/configs/zmx
EOF
  exit 2
fi

in_dir=$1
out_dir=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
converter="$root/optimization/zmx2toml.py"

PY=${PYTHON:-python}
command -v "$PY" >/dev/null 2>&1 || PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python not found (set PYTHON env var to your interpreter)"
  exit 2
fi

if [ ! -d "$in_dir" ]; then
  echo "not a directory: $in_dir"
  exit 2
fi

shopt -s nullglob nocaseglob
files=("$in_dir"/*.zmx)
if [ ${#files[@]} -eq 0 ]; then
  echo "no .zmx files in $in_dir"
  exit 1
fi

skipped_file="$out_dir/skipped.txt"
rm -f "$skipped_file" 2>/dev/null
n_ok=0
n_skip=0
i=0
total=${#files[@]}

for f in "${files[@]}"; do
  i=$((i + 1))
  stem=$(basename "$f")
  stem=${stem%.*}
  sub="$out_dir/$stem"
  msg=$("$PY" "$converter" "$f" "$sub/config.toml" 2>&1)
  if [ $? -eq 0 ]; then
    n_ok=$((n_ok + 1))
    echo "[$i/$total] OK   $stem"
  else
    n_skip=$((n_skip + 1))
    mkdir -p "$out_dir"
    short=$(printf '%s' "$msg" | tail -n 1)
    short=${short#SKIP *: }  # 剥掉转换器消息自带的 "SKIP <名>: " 前缀
    printf '%s\t%s\n' "$f" "$short" >> "$skipped_file"
    echo "[$i/$total] SKIP $stem: $short"
  fi
done

echo "done: $n_ok converted, $n_skip skipped -> $out_dir"
[ "$n_ok" -gt 0 ]
