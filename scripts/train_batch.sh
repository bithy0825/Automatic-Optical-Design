#!/usr/bin/env bash
# 递归批量训练:为目标文件夹下每个 config.toml 以 3 个随机种子依次运行优化。
# 结果写在 config 所在目录:<config 里的名字>_<seed>.pth / .json(如 A001_42.pth)。
# 跨平台:Linux bash 直接运行;Windows 用 Git Bash 或 PowerShell 中 bash 调用。
#
# 用法:
#   bash scripts/train_batch.sh <目标文件夹>
set -u

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  cat <<'EOF'
用法: bash scripts/train_batch.sh <目标文件夹> [每个config的结果数, 默认3]
  递归寻找其下的全部 config.toml;结果(<名字>_<seed>.pth/.json)写在
  config 所在目录。已有结果不足指定数量的,抽新随机种子补足;达到数量
  的整个跳过(重复执行安全)。失败的运行汇总到 <目标文件夹>/failed.txt。
示例:
  bash scripts/train_batch.sh out/zmx        # 每个 config 训 3 个种子
  bash scripts/train_batch.sh out/zmx 5      # 补足到 5 个
EOF
  exit 2
fi

target=$1
want=${2:-3}
case $want in *[!0-9]* | 0 | "")
  echo "结果数必须是正整数: $want"
  exit 2
  ;;
esac
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
trainer="$root/train.py"

PY=${PYTHON:-python}
command -v "$PY" >/dev/null 2>&1 || PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python not found (set PYTHON env var to your interpreter)"
  exit 2
fi

if [ ! -d "$target" ]; then
  echo "not a directory: $target"
  exit 2
fi

mapfile -t configs < <(find "$target" -type f -name config.toml | sort)
total=${#configs[@]}
if [ "$total" -eq 0 ]; then
  echo "no config.toml found under $target"
  exit 1
fi

failed_file="$target/failed.txt"
rm -f "$failed_file" 2>/dev/null
i=0

for cfg in "${configs[@]}"; do
  i=$((i + 1))
  dir=$(dirname "$cfg")
  # 配置里的名字:[train] 节的 output 字段;缺省回退为目录名
  base=$(sed -n 's/^[[:space:]]*output[[:space:]]*=[[:space:]]*"\([^"]*\)"/\1/p' "$cfg" | head -n 1)
  base=${base%.pth}
  [ -n "$base" ] || base=$(basename "$dir")

  # 已有结果数,抽新种子补足到 want 个
  have=$(compgen -G "$dir/${base}_*.pth" | wc -l)
  need=$((want - have))
  if [ "$need" -le 0 ]; then
    echo "[$i/$total] skip  $base (已有 $have 个结果)"
    continue
  fi

  seeds=()
  while [ ${#seeds[@]} -lt "$need" ]; do
    s=$RANDOM
    case " ${seeds[*]-} " in *" $s "*) continue ;; esac
    [ -e "$dir/${base}_${s}.pth" ] && continue
    seeds+=("$s")
  done

  j=0
  for s in "${seeds[@]}"; do
    j=$((j + 1))
    out="$dir/${base}_${s}.pth"
    hist="$dir/${base}_${s}.json"
    # 进度条默认开(PROGRESS=0 关闭,重定向到日志文件时建议关)
    extra=()
    [ "${PROGRESS:-1}" = "0" ] && extra+=("--no-progress")
    echo "[$i/$total] train $base seed=$s ($j/$need)"
    if "$PY" "$trainer" "$cfg" --seed "$s" --output "$out" --history "$hist" "${extra[@]}"; then
      echo "[$i/$total] done  $base seed=$s"
    else
      echo "[$i/$total] FAIL  $base seed=$s"
      printf '%s\tseed=%s\n' "$cfg" "$s" >> "$failed_file"
    fi
  done
done

echo "all done: $total config(s), target $want result(s) each -> $target"
