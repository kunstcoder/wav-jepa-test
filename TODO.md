# WavJEPA vs Audio-JEPA kNN 비교 TODO

## 1) 기준 프로토콜 정리 (Audio-JEPA)
- [ ] Audio-JEPA 논문(2507.02915)에서 kNN 평가 설정 추출
  - [ ] 사용 벤치마크/태스크 목록 정리
  - [ ] 평가 지표(metric) 정리
  - [ ] 제외/예외 태스크 조건 정리
- [ ] Table III(비교표) 기준 baseline 표 초안 작성
- [ ] 재현 체크리스트 작성
  - [ ] 입력 길이/샘플레이트
  - [ ] pooling/normalization
  - [ ] kNN 하이퍼파라미터(k, metric, weighting)

## 2) 평가 프레임워크 준비 (X-ARES)
- [ ] X-ARES 코드/환경 준비
  - [ ] Python 버전 및 패키지 버전 고정
  - [ ] 실행 스크립트 구조 파악
- [ ] kNN 실행 경로 확인
  - [ ] feature 파일 포맷 확인
  - [ ] split(train/val/test) 규칙 확인
- [ ] 실험 로그 규칙 정하기
  - [ ] seed, commit hash, 실행시간 저장

## 3) WavJEPA 임베딩 추출 어댑터 구현
- [ ] WavJEPA inference wrapper 작성
  - [ ] 입력 오디오 로딩/리샘플링
  - [ ] 배치 처리/패딩 처리
- [ ] X-ARES 호환 feature exporter 작성
  - [ ] 파일명/메타데이터 스키마 맞춤
  - [ ] dtype/shape 검증
- [ ] 품질 검증 유틸 추가
  - [ ] NaN/Inf 검사
  - [ ] 빈 임베딩 검사

## 4) 단계별 평가 실행
- [ ] Smoke test (소규모 태스크)
  - [ ] ESC-50
  - [ ] SpeechCommands
  - [ ] GTZAN
- [ ] Full run (전체 태스크)
  - [ ] 실패 태스크 자동 재시도
  - [ ] 결과 파일 자동 수집(csv/json)

## 5) 공정 비교 보정
- [ ] Audio-JEPA와 설정 일치 항목 점검표 작성
- [ ] 불일치 항목 문서화
  - [ ] strict setting 결과
  - [ ] best-effort setting 결과
- [ ] 비교 실험 메모 작성(차이 원인/영향)

## 6) 결과 분석 및 리포팅
- [ ] 태스크별 비교표 작성
  - [ ] WavJEPA vs Audio-JEPA
  - [ ] (선택) wav2vec2/data2vec 포함
- [ ] 평균 성능 계산
  - [ ] macro average
  - [ ] (가능 시) weighted average
- [ ] 핵심 인사이트 정리
  - [ ] 도메인별(음성/음악/환경음) 패턴
  - [ ] 강점/약점 분석

## 7) 재현성 패키징
- [ ] `run_knn_eval.sh` 작성
- [ ] `requirements.txt` 또는 `environment.yml` 정리
- [ ] `results_knn_comparison.md` 작성
- [ ] raw 결과 아카이브 정리

## 완료 기준 (Definition of Done)
- [ ] 동일 환경에서 1-command로 재실행 가능
- [ ] 태스크별 및 평균 점수 표가 재생성 가능
- [ ] 설정 차이/제약 사항이 문서화됨
- [ ] 최종 비교 리포트 공유 가능 상태
