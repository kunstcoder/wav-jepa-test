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

### 2-1. 오디오 폴더
오디오 루트 폴더를 준비합니다. 스크립트가 하위 디렉터리를 재귀적으로 탐색합니다.

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
  --audio-dir /data/audio \
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
  --audio-dir /data/audio \
  --output-dir artifacts/features \
  --backend python \
  --module my_models.wavjepa_adapter \
  --class-name MyWavJEPAEncoder \
  --model-path /path/to/checkpoint.pt
```

### B-1) ckpt 직접 로드 백엔드
`.ckpt/.pt/.pth`를 state_dict로 로드해야 할 때:

```bash
python3 scripts/extract_wavjepa_features.py \
  --audio-dir /data/audio \
  --output-dir artifacts/features \
  --backend python-ckpt \
  --module my_models.wavjepa_adapter \
  --class-name MyWavJEPAEncoder \
  --model-path /path/to/model.ckpt
```

### C) Hugging Face safetensors 사용
Hugging Face repo에서 safetensors를 직접 받아 로드할 수 있습니다.

```bash
python3 scripts/extract_wavjepa_features.py \
  --audio-dir /data/audio \
  --output-dir artifacts/features \
  --backend python-safetensors \
  --hf-model-id org_or_user/model_repo \
  --hf-filename model.safetensors \
  --module my_models.wavjepa_adapter \
  --class-name MyWavJEPAEncoder \
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

## 5. 오디오부터 kNN까지 1-커맨드 실행 (권장)

`run_knn_eval.sh`가 이제 **feature 사전추출 없이**, 오디오 입력부터 kNN 평가까지 한 번에 실행합니다.

```bash
./run_knn_eval.sh \
  --extract \
  --audio-dir /data/audio \
  --splits data/splits.csv \
  --output-dir artifacts/e2e \
  --features-dir artifacts/e2e/features \
  --backend python-ckpt \
  --model-path /path/to/model.ckpt \
  --module my_models.wavjepa_adapter \
  --class-name MyWavJEPAEncoder \
  --k 200 \
  --metric cosine \
  --weighting distance \
  --seed 42
```

`--features-dir`를 생략하면 기본값으로 `<output-dir>/features`를 사용합니다.

## 6. kNN 평가만 실행 (사전 추출 feature 사용)

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

## 7. 빠른 점검 명령어

```bash
python3 scripts/extract_wavjepa_features.py --help
python3 scripts/knn_eval.py --help
python3 scripts/collect_results.py --help
```

## 8. 참고

- `results_knn_comparison.md`에 태스크별/평균 성능표를 채워 리포트로 사용하세요.
- `run_knn_eval.sh --retry-failed` 옵션으로 실패 태스크 자동 재시도가 가능합니다.
