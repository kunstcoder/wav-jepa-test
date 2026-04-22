# WavJEPA kNN 실행 가이드 (단일 커맨드)

## 실행 방식
아래 **한 줄 명령**으로 `load -> encode -> kNN` 전체를 수행합니다.

```bash
./run_knn_eval.sh --model-path <MODEL_FILE> --data-path <DATA_ROOT> --encoder <context|target|auto>
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
