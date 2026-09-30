# Git 관리 규칙

## 1. 브랜치
- `main` 하나만 유지. 직접 push 금지, PR 머지로만 변경.
- 작업 전 issue 먼저 생성 → issue 번호로 브랜치 생성.
  - 예: `feat/12-face-preprocess`, `fix/15-camera-fps`, `docs/3-labels`
- 브랜치는 `main`에서 따고, 머지 후 삭제.

## 2. 커밋
- 작은 단위로, 테스트 통과 상태에서 커밋.
- 메시지 형식: `<type>: <설명> (#<issue>)`
  - type: `feat` / `fix` / `docs` / `test` / `refactor` / `chore`
  - 예: `feat: 얼굴 크롭 전처리 추가 (#12)`

## 3. PR
- 제목에 issue 번호, 본문에 "무엇을 / 왜 / 테스트 방법" + `Closes #<issue>`.
- 리뷰어 1명 이상 approve 후 머지. 본인 머지 금지.
- 머지 방식: 일반 merge (merge commit). 브랜치 커밋 이력이 `main`에 그대로 남음.
- 충돌 시 작성자가 `main`을 merge 또는 rebase 후 다시 push.

## 4. 리뷰
- 24시간 내 응답.
- 코멘트는 `제안` / `질문` / `필수` 로 구분.
- `필수`만 반영 후 머지, 나머지는 후속 issue로.

## 5. 커밋 금지 대상
- 데이터셋 원본, 모델 가중치(`.pt`, `.tflite` → 릴리스 또는 Google Drive), `.venv`, 캐시
- `.gitignore`로 차단.

## 6. Issue 라벨
- 담당 영역: `face` / `game` / `data` / `eval` / `docs`
- 종류: `bug` / `enhancement` / `question`
