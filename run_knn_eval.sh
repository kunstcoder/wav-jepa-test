#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 --features-dir <path> --splits <path> --output-dir <path> [options]

Required:
  --features-dir PATH   Directory containing extracted embeddings (*.npy)
  --splits PATH         CSV split definition (id,label,split[,task])
  --output-dir PATH     Directory to save logs and result files

Optional:
  --k INT               k for kNN (default: 200)
  --metric NAME         distance metric (default: cosine)
  --weighting NAME      weighting strategy: uniform|distance (default: distance)
  --seed INT            random seed (default: 42)
  --tasks LIST          comma-separated task list (default: all from split CSV)
  --retry-failed INT    retry count for failed task runs (default: 2)
  --dry-run             validate inputs and write metadata only
  --help                show this message
USAGE
}

FEATURES_DIR=""
SPLITS=""
OUTPUT_DIR=""
K=200
METRIC="cosine"
WEIGHTING="distance"
SEED=42
TASKS=""
RETRY_FAILED=2
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --features-dir) FEATURES_DIR="$2"; shift 2 ;;
    --splits) SPLITS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --metric) METRIC="$2"; shift 2 ;;
    --weighting) WEIGHTING="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --retry-failed) RETRY_FAILED="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

[[ -n "$FEATURES_DIR" ]] || { echo "--features-dir is required"; exit 2; }
[[ -n "$SPLITS" ]] || { echo "--splits is required"; exit 2; }
[[ -n "$OUTPUT_DIR" ]] || { echo "--output-dir is required"; exit 2; }

mkdir -p "$OUTPUT_DIR"
START_TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
COMMIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

RUN_LOG="$OUTPUT_DIR/run_meta.json"
cat > "$RUN_LOG" <<JSON
{
  "start_time_utc": "$START_TS",
  "commit_hash": "$COMMIT_HASH",
  "seed": $SEED,
  "features_dir": "${FEATURES_DIR}",
  "splits": "${SPLITS}",
  "knn": {
    "k": $K,
    "metric": "${METRIC}",
    "weighting": "${WEIGHTING}"
  }
}
JSON

echo "[INFO] metadata written: $RUN_LOG"

echo "[INFO] validating inputs..."
[[ -d "$FEATURES_DIR" ]] || { echo "[ERROR] features directory not found: $FEATURES_DIR"; exit 1; }
[[ -f "$SPLITS" ]] || { echo "[ERROR] split file not found: $SPLITS"; exit 1; }
[[ "$RETRY_FAILED" =~ ^[0-9]+$ ]] || { echo "[ERROR] --retry-failed must be non-negative int"; exit 2; }

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[INFO] dry-run mode: validation complete"
  exit 0
fi

TASK_FILE="$OUTPUT_DIR/tasks.txt"
if [[ -n "$TASKS" ]]; then
  echo "$TASKS" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sed '/^$/d' > "$TASK_FILE"
else
  python3 - "$SPLITS" > "$TASK_FILE" <<'PY'
import pandas as pd
import sys
df = pd.read_csv(sys.argv[1])
if "task" not in df.columns:
    print("default")
else:
    for task in sorted(df["task"].dropna().astype(str).unique()):
        print(task)
PY
fi

echo "[INFO] task list:"
cat "$TASK_FILE" | sed 's/^/  - /'

while IFS= read -r TASK; do
  [[ -n "$TASK" ]] || continue
  TASK_SAFE="$(echo "$TASK" | tr -c '[:alnum:]_.-' '_')"
  ATTEMPT=0
  OK=0
  while [[ "$ATTEMPT" -le "$RETRY_FAILED" ]]; do
    echo "[INFO] task=$TASK attempt=$((ATTEMPT + 1))/$((RETRY_FAILED + 1))"
    set +e
    python3 scripts/knn_eval.py \
      --features-dir "$FEATURES_DIR" \
      --splits "$SPLITS" \
      --output-dir "$OUTPUT_DIR" \
      --task "$TASK" \
      --results-csv "results_${TASK_SAFE}.csv" \
      --results-json "results_${TASK_SAFE}.json" \
      --k "$K" \
      --metric "$METRIC" \
      --weighting "$WEIGHTING" \
      --seed "$SEED"
    RC=$?
    set -e
    if [[ "$RC" -eq 0 ]]; then
      OK=1
      break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    if [[ "$ATTEMPT" -le "$RETRY_FAILED" ]]; then
      echo "[WARN] retrying task=$TASK due to failure (exit=$RC)"
    fi
  done

  if [[ "$OK" -ne 1 ]]; then
    echo "[ERROR] task failed after retries: $TASK"
    exit 1
  fi
done < "$TASK_FILE"

python3 scripts/collect_results.py \
  --input-dir "$OUTPUT_DIR" \
  --output-csv "$OUTPUT_DIR/results.csv" \
  --output-json "$OUTPUT_DIR/results.json"

echo "[INFO] evaluation completed"
echo "       - $OUTPUT_DIR/results.csv"
echo "       - $OUTPUT_DIR/results.json"
