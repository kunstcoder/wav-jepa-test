# WavJEPA vs Audio-JEPA kNN Comparison Results

## 1. Experiment metadata
- Date (UTC):
- Commit hash:
- Python version:
- Environment file: `requirements.txt`
- Runner command:

```bash
./run_knn_eval.sh \
  --features-dir <features_dir> \
  --splits <splits_file.csv> \
  --output-dir <output_dir> \
  --k 200 \
  --metric cosine \
  --weighting distance \
  --seed 42
```

## 2. Input format
### Split CSV schema
- Required columns: `id`, `label`, `split`
- Optional column: `task` (없으면 `default`로 처리)
- `split` 값은 최소한 `train`, `test`를 포함해야 함

### Feature file schema
- 파일 경로: `<features_dir>/<id>.npy`
- shape: `(D,)` 또는 `(T, D)`
  - `(T, D)`의 경우 시간축 평균으로 pooling
- 값 검증: NaN/Inf/empty 벡터 금지, task 내 차원 불일치 금지

## 3. Protocol alignment checklist
- [ ] sample rate / audio length aligned with Audio-JEPA
- [ ] pooling / normalization aligned
- [ ] kNN hyperparameters aligned (k, metric, weighting)

## 4. Task-level scores
| Task | Audio-JEPA | WavJEPA | Delta (Wav - Audio) | Notes |
|---|---:|---:|---:|---|
| ESC-50 | - | - | - | |
| SpeechCommands | - | - | - | |
| GTZAN | - | - | - | |

## 5. Aggregate scores
| Aggregate | Audio-JEPA | WavJEPA | Delta |
|---|---:|---:|---:|
| Macro average | - | - | - |
| Weighted average (optional) | - | - | - |

## 6. Mismatch notes (strict vs best-effort)
### strict setting
- 

### best-effort setting
- 

## 7. Insights
- Domain pattern (speech/music/environment):
- Strengths:
- Weaknesses:
