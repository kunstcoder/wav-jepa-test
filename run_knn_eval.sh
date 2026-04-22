#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $0 --model-path <path> --data-path <path> --encoder <context|target|auto>

Required arguments (only 3):
  --model-path PATH    WavJEPA model file path (.ts/.torchscript recommended)
  --data-path PATH     Dataset root containing:
                       - splits.csv
                       - audio/ (recursive audio files)
  --encoder NAME       context | target | auto

Defaults (fixed sane values):
  output dir: artifacts/knn_eval
  k: 200, metric: cosine, weighting: distance
  sample rate: 16000, batch size: 16, device: cpu, seed: 42

Example:
  $0 --model-path ./models/wavjepa.ts --data-path ./data/myset --encoder context
USAGE
}

MODEL_PATH=""
DATA_PATH=""
ENCODER=""
OUTPUT_DIR="artifacts/knn_eval"
K=200
METRIC="cosine"
WEIGHTING="distance"
SAMPLE_RATE=16000
BATCH_SIZE=16
DEVICE="cpu"
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --encoder) ENCODER="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

[[ -n "$MODEL_PATH" ]] || { echo "[ERROR] --model-path is required"; exit 2; }
[[ -n "$DATA_PATH" ]] || { echo "[ERROR] --data-path is required"; exit 2; }
[[ -n "$ENCODER" ]] || { echo "[ERROR] --encoder is required"; exit 2; }
[[ "$ENCODER" =~ ^(context|target|auto)$ ]] || { echo "[ERROR] --encoder must be context|target|auto"; exit 2; }

[[ -f "$MODEL_PATH" ]] || { echo "[ERROR] model file not found: $MODEL_PATH"; exit 1; }
[[ -d "$DATA_PATH" ]] || { echo "[ERROR] data path not found: $DATA_PATH"; exit 1; }
[[ -f "$DATA_PATH/splits.csv" ]] || { echo "[ERROR] splits.csv not found: $DATA_PATH/splits.csv"; exit 1; }
[[ -d "$DATA_PATH/audio" ]] || { echo "[ERROR] audio dir not found: $DATA_PATH/audio"; exit 1; }

mkdir -p "$OUTPUT_DIR"
START_TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
COMMIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

cat > "$OUTPUT_DIR/run_meta.json" <<JSON
{
  "start_time_utc": "$START_TS",
  "commit_hash": "$COMMIT_HASH",
  "seed": $SEED
}
JSON

python3 scripts/knn_eval.py \
  --model-path "$MODEL_PATH" \
  --data-path "$DATA_PATH" \
  --encoder "$ENCODER" \
  --output-dir "$OUTPUT_DIR" \
  --k "$K" \
  --metric "$METRIC" \
  --weighting "$WEIGHTING" \
  --sample-rate "$SAMPLE_RATE" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --seed "$SEED"

echo "[INFO] done"
echo "       - $OUTPUT_DIR/results.csv"
echo "       - $OUTPUT_DIR/results.json"
echo "       - $OUTPUT_DIR/run_meta.json"
