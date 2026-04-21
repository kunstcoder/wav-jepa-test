# WavJEPA 임베딩 추출 + kNN 평가 실행 가이드

이 문서는 `TODO.md`의 **3) WavJEPA 임베딩 추출 어댑터 구현** 완료 기준에 맞춰,
로컬에서 바로 재현 가능한 실행 절차를 설명합니다.

## 1. 환경 준비

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 입력 파일 준비

### 2-1. 임베딩 추출용 manifest CSV
아래 컬럼을 가진 CSV를 준비합니다.

- `id` (필수): 샘플 식별자
- `audio_path` (필수): 오디오 파일 경로
- `task` (선택): 태스크 이름 (없으면 `default`)

예시(`data/manifest.csv`):

```csv
id,audio_path,task
esc_0001,/data/audio/esc_0001.wav,ESC-50
sc_1203,/data/audio/sc_1203.wav,SpeechCommands
```

### 2-2. kNN split CSV
아래 컬럼을 가진 split CSV를 준비합니다.

- `id`, `label`, `split` 필수
- `task` 선택

예시(`data/splits.csv`):

```csv
id,label,split,task
esc_0001,dog,train,ESC-50
esc_0002,cat,test,ESC-50
```

## 3. WavJEPA 임베딩 추출

### A) TorchScript 백엔드 (권장)

```bash
python3 scripts/extract_wavjepa_features.py \
  --manifest data/manifest.csv \
  --output-dir artifacts/features \
  --backend torchscript \
  --model-path /path/to/wavjepa_encoder.ts \
  --sample-rate 16000 \
  --batch-size 16 \
  --device cpu
```

### B) Python 클래스 백엔드
커스텀 로더를 쓰는 경우:

```bash
python3 scripts/extract_wavjepa_features.py \
  --manifest data/manifest.csv \
  --output-dir artifacts/features \
  --backend python \
  --module my_models.wavjepa_adapter \
  --class-name MyWavJEPAEncoder \
  --model-path /path/to/checkpoint.pt
```

### C) 폴더 입력 + recursive 로드
manifest 없이 폴더를 직접 지정하면 하위 디렉터리를 재귀적으로 스캔해 오디오 파일을 로드합니다.

```bash
python3 scripts/extract_wavjepa_features.py \
  --audio-dir /data/audio \
  --output-dir artifacts/features \
  --backend torchscript \
  --model-path /path/to/wavjepa_encoder.ts \
  --encoder-output context
```

- `--audio-exts`로 확장자 목록을 지정할 수 있습니다(기본: `.wav,.flac,.mp3,.ogg,.m4a`).
- `--task`로 폴더 입력 시 공통 task 값을 지정할 수 있습니다.

## 4. 추출 결과 확인

`artifacts/features` 하위에 다음이 생성됩니다.

- `<id>.npy` : X-ARES 호환 feature 파일
- `features_manifest.csv` : exporter 메타데이터
- `export_summary.json` : 샘플 수/임베딩 차원 요약

검증 포인트:
- `.npy`는 1D 벡터여야 함
- NaN/Inf가 없어야 함
- 전체 샘플 임베딩 차원이 동일해야 함

## 5. kNN 평가 실행

```bash
./run_knn_eval.sh \
  --features-dir artifacts/features \
  --splits data/splits.csv \
  --output-dir artifacts/knn \
  --k 200 \
  --metric cosine \
  --weighting distance \
  --seed 42
```

결과:
- `artifacts/knn/results.csv`
- `artifacts/knn/results.json`
- `artifacts/knn/run_meta.json`

## 6. 빠른 점검 명령어

```bash
python3 scripts/extract_wavjepa_features.py --help
python3 scripts/knn_eval.py --help
python3 scripts/collect_results.py --help
```

## 7. 참고

- `results_knn_comparison.md`에 태스크별/평균 성능표를 채워 리포트로 사용하세요.
- `run_knn_eval.sh --retry-failed` 옵션으로 실패 태스크 자동 재시도가 가능합니다.
