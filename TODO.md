# WavJEPA kNN 실행 단순화 TODO

## 목표
- [x] 실행 진입점을 **단일 커맨드**로 단순화
- [x] 사용자가 설정해야 할 인자를 아래 3개로 제한
  - [x] 모델 파일 경로 (`--model-path`)
  - [x] 데이터 경로 (`--data-path`)
  - [x] encoder 선택 (`--encoder`)
- [x] 특징추출(feature extraction)과 kNN 평가를 내부 파이프라인으로 통합
- [x] 중간 feature 파일 저장 없이(in-memory) 즉시 평가

## 1) 실행 인터페이스 정리
- [x] `run_knn_eval.sh` 인터페이스 최소화
  - [x] `--features-dir`, `--extract` 등 단계 분리형 옵션 제거
  - [x] 필수 인자 3개만 받도록 변경
  - [x] 나머지 값은 sane default로 고정
- [x] `--help` 메시지를 단순화된 사용 예시 중심으로 재작성
- [x] README 실행 예시를 1줄 커맨드 기준으로 갱신

## 2) 파이프라인 구조 단순화
- [x] 추출 스크립트와 평가 스크립트의 분리 실행 의존성 제거
- [x] 내부 처리 흐름을 `load -> encode -> kNN` 단일 경로로 고정
- [x] split/task 반복 로직은 유지하되 외부 노출 옵션 최소화
- [x] 실패 재시도/수집 로직은 유지 여부 재평가 후 불필요 시 제거

## 3) 저장/출력 정책 정리
- [x] 중간 산출물 정책 변경
  - [x] feature `.npy` 저장 비활성화(기본/권장 경로)
  - [x] 최종 평가지표(`results.csv/json`)만 저장
- [x] 메타로그는 핵심 필드만 유지
  - [x] `start_time_utc`, `commit_hash`, `seed`

## 4) 불필요 코드 제거
- [x] 현재 코드 기준 dead path 식별
  - [x] placeholder/레거시 옵션
  - [x] 문서와 동작이 불일치하는 분기
  - [ ] 더 이상 사용하지 않는 유틸 함수
- [ ] 제거 후 lint/type/basic run으로 회귀 확인

## 5) 검증 체크리스트
- [x] 아래 명령 1개로 전체 실행 가능
  - [x] `./run_knn_eval.sh --model-path <...> --data-path <...> --encoder <...>`
- [x] feature 파일 없이 동일 결과 재현 가능
- [x] 기존 대비 사용자 입력 인자 수가 줄었는지 확인
- [x] 실패 시 에러 메시지가 입력 누락/경로 오류를 명확히 안내

---

## 작업 요약 (2026-04-22)
- `run_knn_eval.sh`를 단일 실행 경로로 재작성해 필수 인자를 3개(`--model-path`, `--data-path`, `--encoder`)로 고정함.
- `scripts/knn_eval.py`를 end-to-end 파이프라인으로 변경해 오디오 로드/임베딩/평가를 한 프로세스에서 in-memory로 수행하도록 구현함.
- 기본 출력을 최종 결과(`results.csv/json`)와 최소 메타로그(`run_meta.json`)만 남기도록 정리함.
- README를 단일 커맨드 기준으로 갱신하고 데이터 입력 규약(`splits.csv`, `audio/`)을 명시함.
