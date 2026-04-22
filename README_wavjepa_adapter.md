# WavJEPA kNN 실행 가이드 (단일 커맨드)

## 실행 방식
아래 **한 줄 명령**으로 `load -> encode -> kNN` 전체를 수행합니다.

```bash
./run_knn_eval.sh --model-path <MODEL_FILE> --data-path <DATA_ROOT> --encoder <context|target|auto>
```

### 공식 WavJEPA(Hugging Face remote code) 추론 경로
패키지 설치 없이 공식 코드 기반 추론을 사용하려면 `scripts/knn_eval.py`를 직접 실행해
`--backend wavjepa-hf`와 `--hf-model-id`를 지정하세요.

```bash
python3 scripts/knn_eval.py \
  --backend wavjepa-hf \
  --hf-model-id labhamlet/wavjepa-base \
  --data-path <DATA_ROOT> \
  --encoder context
```

### 공식 WavJEPA GitHub 학습 ckpt 추론 경로
GitHub 학습 코드로 만든 `.ckpt`를 사용할 때는, 공식 repo를 클론한 경로를 `--source-root`로 넘기면
패키지 설치 없이 클래스 자동 탐색 후 ckpt 로딩을 시도합니다.

```bash
python3 scripts/knn_eval.py \
  --model-path /path/to/your.ckpt \
  --backend python-ckpt \
  --source-root /path/to/cloned/wavjepa \
  --data-path <DATA_ROOT> \
  --encoder context
```

## 입력 규약
`--data-path`는 아래 두 구조 중 하나를 가져야 합니다.

```text
<DATA_ROOT>/
  [옵션 A: CSV 기반]
    splits.csv
    audio/
      ... (재귀 탐색)

  [옵션 B: 폴더 기반]
    train/
      *.wav (+ optional same-name *.json with "labels" or "label")
    test/
      *.wav (+ optional same-name *.json with "labels" or "label")
```

### `splits.csv` 필수 컬럼
- `id`, `label`, `split` (필수)
- `task` (선택, 없으면 `default`)

예시:

```csv
id,label,split,task
esc_0001,dog,train,ESC-50
esc_0002,cat,test,ESC-50
```

`id`는 오디오 파일 stem(확장자 제외 이름)과 매칭됩니다.

> `splits.csv`가 없으면 코드가 `train/`, `test/`를 자동 스캔합니다.  
> 이때 라벨은 우선 같은 이름의 `.json`(`labels` 또는 `label`)에서 읽고, 없으면 하위 폴더명을 라벨로 사용합니다.

## 출력
기본 출력 경로: `artifacts/knn_eval`

- `results.csv`
- `results.json`
- `run_meta.json` (`start_time_utc`, `commit_hash`, `seed`만 기록)

중간 feature `.npy` 파일은 저장하지 않습니다.

## 도움말

```bash
./run_knn_eval.sh --help
```
