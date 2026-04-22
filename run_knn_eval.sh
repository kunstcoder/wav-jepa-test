#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  1) pre-extracted features:
     $0 --features-dir <path> --splits <path> --output-dir <path> [options]
  2) end-to-end (audio -> feature extraction -> kNN):
     $0 --splits <path> --output-dir <path> --extract [extract options] [kNN options]

Required:
  --splits PATH         CSV split definition (id,label,split[,task])
  --output-dir PATH     Directory to save logs and result files
  (Either --features-dir OR --extract must be set)

Extraction mode:
  --extract             Run feature extraction before kNN
  --audio-dir PATH      Audio directory for extraction (recursive)
  --features-dir PATH   Input feature dir (pre-extracted mode) or extraction output dir
  --backend NAME        auto|torchscript|python|python-ckpt|python-safetensors (default: auto)
  --model-path PATH     Local model/ckpt/safetensors path for extractor
  --hf-model-id NAME    Hugging Face repo id (optional, with --hf-filename)
  --hf-filename NAME    Hugging Face model filename (default: model.safetensors)
  --module NAME         Python module path for python/python-ckpt backend
  --class-name NAME     Python class name for python/python-ckpt backend
  --encoder-output NAME context|target|auto (default: context)
  --sample-rate INT     extractor sample rate (default: 16000)
  --batch-size INT      extractor batch size (default: 16)
  --device NAME         extractor device (default: cpu)
  --audio-exts LIST     extractor extension list (default: .wav,.flac,.mp3,.ogg,.m4a)
  --extract-task NAME   task name when using --audio-dir (default: default)

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
DO_EXTRACT=0
AUDIO_DIR=""
BACKEND="auto"
MODEL_PATH=""
HF_MODEL_ID=""
HF_FILENAME="model.safetensors"
MODULE_NAME=""
CLASS_NAME=""
ENCODER_OUTPUT="context"
SAMPLE_RATE=16000
BATCH_SIZE=16
DEVICE="cpu"
AUDIO_EXTS=".wav,.flac,.mp3,.ogg,.m4a"
EXTRACT_TASK="default"
K=200
METRIC="cosine"
WEIGHTING="distance"
SEED=42
TASKS=""
RETRY_FAILED=2
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --extract) DO_EXTRACT=1; shift ;;
    --audio-dir) AUDIO_DIR="$2"; shift 2 ;;
    --features-dir) FEATURES_DIR="$2"; shift 2 ;;
    --splits) SPLITS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --hf-model-id) HF_MODEL_ID="$2"; shift 2 ;;
    --hf-filename) HF_FILENAME="$2"; shift 2 ;;
    --module) MODULE_NAME="$2"; shift 2 ;;
    --class-name) CLASS_NAME="$2"; shift 2 ;;
    --encoder-output) ENCODER_OUTPUT="$2"; shift 2 ;;
    --sample-rate) SAMPLE_RATE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --audio-exts) AUDIO_EXTS="$2"; shift 2 ;;
    --extract-task) EXTRACT_TASK="$2"; shift 2 ;;
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

[[ -n "$SPLITS" ]] || { echo "--splits is required"; exit 2; }
[[ -n "$OUTPUT_DIR" ]] || { echo "--output-dir is required"; exit 2; }
if [[ "$DO_EXTRACT" -ne 1 && -z "$FEATURES_DIR" ]]; then
  echo "[ERROR] set --features-dir (pre-extracted mode) or use --extract"
  exit 2
fi
if [[ "$DO_EXTRACT" -eq 1 ]]; then
  if [[ -z "$FEATURES_DIR" ]]; then
    FEATURES_DIR="$OUTPUT_DIR/features"
  fi
  if [[ -z "$AUDIO_DIR" ]]; then
    echo "[ERROR] extraction mode requires --audio-dir"
    exit 2
  fi
fi

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
  "extract": {
    "enabled": $DO_EXTRACT,
    "audio_dir": "${AUDIO_DIR}",
    "backend": "${BACKEND}",
    "model_path": "${MODEL_PATH}",
    "hf_model_id": "${HF_MODEL_ID}",
    "hf_filename": "${HF_FILENAME}",
    "module": "${MODULE_NAME}",
    "class_name": "${CLASS_NAME}",
    "encoder_output": "${ENCODER_OUTPUT}",
    "sample_rate": $SAMPLE_RATE,
    "batch_size": $BATCH_SIZE,
    "device": "${DEVICE}",
    "audio_exts": "${AUDIO_EXTS}",
    "task": "${EXTRACT_TASK}"
  },
  "knn": {
    "k": $K,
    "metric": "${METRIC}",
    "weighting": "${WEIGHTING}"
  }
}
JSON

echo "[INFO] metadata written: $RUN_LOG"

echo "[INFO] validating inputs..."
[[ -f "$SPLITS" ]] || { echo "[ERROR] split file not found: $SPLITS"; exit 1; }
[[ "$RETRY_FAILED" =~ ^[0-9]+$ ]] || { echo "[ERROR] --retry-failed must be non-negative int"; exit 2; }
if [[ "$DO_EXTRACT" -eq 1 ]]; then
  if [[ -n "$AUDIO_DIR" ]]; then
    [[ -d "$AUDIO_DIR" ]] || { echo "[ERROR] audio directory not found: $AUDIO_DIR"; exit 1; }
  fi
  if [[ -z "$MODEL_PATH" && -z "$HF_MODEL_ID" ]]; then
    echo "[ERROR] extraction mode requires --model-path or --hf-model-id"
    exit 2
  fi
  if [[ -n "$MODEL_PATH" ]]; then
    [[ -f "$MODEL_PATH" ]] || { echo "[ERROR] model path not found: $MODEL_PATH"; exit 1; }
  fi
else
  [[ -d "$FEATURES_DIR" ]] || { echo "[ERROR] features directory not found: $FEATURES_DIR"; exit 1; }
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[INFO] dry-run mode: validation complete"
  exit 0
fi

if [[ "$DO_EXTRACT" -eq 1 ]]; then
  mkdir -p "$FEATURES_DIR"
  EXTRACT_CMD=(
    python3 scripts/extract_wavjepa_features.py
    --output-dir "$FEATURES_DIR"
    --backend "$BACKEND"
    --model-path "$MODEL_PATH"
    --hf-model-id "$HF_MODEL_ID"
    --hf-filename "$HF_FILENAME"
    --encoder-output "$ENCODER_OUTPUT"
    --sample-rate "$SAMPLE_RATE"
    --batch-size "$BATCH_SIZE"
    --device "$DEVICE"
    --audio-exts "$AUDIO_EXTS"
  )
  if [[ -n "$MODULE_NAME" ]]; then
    EXTRACT_CMD+=(--module "$MODULE_NAME")
  fi
  if [[ -n "$CLASS_NAME" ]]; then
    EXTRACT_CMD+=(--class-name "$CLASS_NAME")
  fi
  EXTRACT_CMD+=(--audio-dir "$AUDIO_DIR" --task "$EXTRACT_TASK")
  echo "[INFO] running extraction..."
  "${EXTRACT_CMD[@]}"
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
