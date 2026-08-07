#!/usr/bin/env bash
# 递归批量训练:为目标文件夹下每个 config.toml 以随机种子补足优化结果。
# 跨平台:Linux bash 直接运行;Windows 用 Git Bash 或 PowerShell 中 bash 调用。
#
# 用法:
#   bash scripts/train_batch.sh <目标文件夹> [每个config的结果数, 默认3]
#   JOBS=4 bash scripts/train_batch.sh lib 3     # 4 进程并行
set -u

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  cat <<'EOF'
用法: bash scripts/train_batch.sh <目标文件夹> [每个config的结果数, 默认1]
  递归寻找其下的全部 config.toml;结果(<名字>_<seed>.pth/.json)写在
  config 所在目录。已有结果不足指定数量的,抽新随机种子补足;达到的跳过。
  JOBS=N 并行训练(默认 1 串行,可手动开,如 JOBS=4;并行时进度条自动关闭,
  每个进程的完整输出写在 config 同目录的 <名字>_<seed>.log)。
  失败的运行汇总到 <目标文件夹>/failed.txt(本次运行无失败则删除旧文件)。
示例:
  bash scripts/train_batch.sh lib          # 每个 config 训 1 个种子
  bash scripts/train_batch.sh lib 3        # 补足到 3 个
  JOBS=4 bash scripts/train_batch.sh lib   # 4 进程并行
EOF
  exit 2
fi

target=$1
want=${2:-1}
case $want in *[!0-9]* | 0 | "")
  echo "结果数必须是正整数: $want"
  exit 2
  ;;
esac

JOBS=${JOBS:-1}
case $JOBS in *[!0-9]* | 0 | "")
  echo "JOBS 必须是正整数: $JOBS"
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

# 并行时进度条会交错刷屏,自动关闭;PROGRESS=0 可手动关
extra=()
if [ "${PROGRESS:-1}" = "0" ] || [ "$JOBS" -gt 1 ]; then
  extra+=("--no-progress")
fi

failed_file="$target/failed.txt"
rm -f "$failed_file" 2>/dev/null

# 从 config 里读 [train].output 的名字(缺省回退目录名)
name_of() {
  local b
  b=$(sed -n 's/^[[:space:]]*output[[:space:]]*=[[:space:]]*"\([^"]*\)"/\1/p' "$1" | head -n 1 | tr -d '\r')
  b=${b%.pth}
  b=${b%.json}
  if [ -n "$b" ]; then
    printf '%s' "$b"
  else
    basename "$(dirname "$1")"
  fi
}

# ── 阶段 1:枚举 config,为每个抽足新种子,生成任务清单 ──
mapfile -t configs < <(find "$target" -type f -name config.toml | sort)
total=${#configs[@]}
if [ "$total" -eq 0 ]; then
  echo "no config.toml found under $target"
  exit 1
fi

tasks=$(mktemp)
i=0
for cfg in "${configs[@]}"; do
  i=$((i + 1))
  dir=$(dirname "$cfg")
  base=$(name_of "$cfg")

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

  for s in "${seeds[@]}"; do
    printf '%s\t%s\n' "$cfg" "$s" >> "$tasks"
  done
  echo "[$i/$total] queue $base: 补 $need 个种子 (${seeds[*]})"
done

n_tasks=$(wc -l < "$tasks")
if [ "$n_tasks" -eq 0 ]; then
  rm -f "$tasks"
  echo "nothing to do: all configs already have $want result(s)"
  exit 0
fi
echo "== $n_tasks 个训练任务,JOBS=$JOBS =="

# ── 阶段 2:任务池执行 ──
run_one() {
  local cfg=$1 s=$2 dir base out hist
  dir=$(dirname "$cfg")
  base=$(name_of "$cfg")
  out="$dir/${base}_${s}.pth"
  hist="$dir/${base}_${s}.json"
  echo "[start] $base seed=$s"
  # 并行:各进程输出写入 <名字>_<seed>.log,控制台只留调度行;失败时回显末尾
  if [ "$JOBS" -gt 1 ]; then
    if "$PY" "$trainer" "$cfg" --seed "$s" --output "$out" --history "$hist" "${extra[@]}" > "$dir/${base}_${s}.log" 2>&1; then
      echo "[done ] $base seed=$s"
    else
      echo "[FAIL ] $base seed=$s (日志末尾:)"
      tail -n 5 "$dir/${base}_${s}.log"
      printf '%s\tseed=%s\n' "$cfg" "$s" >> "$failed_file"
    fi
  elif "$PY" "$trainer" "$cfg" --seed "$s" --output "$out" --history "$hist" "${extra[@]}"; then
    echo "[done ] $base seed=$s"
  else
    echo "[FAIL ] $base seed=$s"
    printf '%s\tseed=%s\n' "$cfg" "$s" >> "$failed_file"
  fi
}

while IFS=$'\t' read -r cfg s; do
  run_one "$cfg" "$s" &
  while [ "$(jobs -r | wc -l)" -ge "$JOBS" ]; do
    wait -n
  done
done < "$tasks"
wait
rm -f "$tasks"

if [ -f "$failed_file" ]; then
  echo "done, with failures listed in $failed_file"
else
  echo "done: all $n_tasks run(s) succeeded"
fi
