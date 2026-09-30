# 개발 요구사항

- `docs/biz_spec.md` 의 기능/비기능 요구사항을 구현 단위로 옮긴 문서
- 모델은 `model.py` 의 인터페이스로만 정의. 실제 모델 준비 전까지는 가짜 모델로 대체

## 1. 화면 상태와 전환

| 현재 상태 | 이벤트 | 다음 상태 |
|---|---|---|
| HOME | 촬영 시작 버튼 클릭 | PREVIEW |
| PREVIEW | 손 들기 감지 (얼굴 1명 이상, 안내 표시 중이 아님) `HAND_ON_COUNT` 회 연속 | PREPARE |
| PREVIEW | 초기화면 버튼 | HOME |
| PREPARE | 모든 얼굴이 웃음 `SMILE_ON_COUNT` 회 연속 | CAPTURE |
| PREPARE | 준비 제한 시간 초과 | PREVIEW (취소 문구 표시) |
| CAPTURE | 촬영 제한 시간 종료 | CHECKING |
| PREPARE, CAPTURE | 카메라 끊김 (`CAMERA_LOST_SEC` 동안 새 프레임 없음) | PREVIEW (촬영 취소, 찍던 프레임 버림. 취소 문구는 띄우지 않음) |
| CHECKING | 모든 사진의 조건 충족 여부 확인 완료 | RESULT |
| RESULT | 미리보기 버튼 | PREVIEW (사진 정리) |
| RESULT | 초기화면 버튼 | HOME (사진 정리) |
| HOME | 모델 평가 버튼 (dev 모드에만 있음) | MODEL_EVAL |
| MODEL_EVAL | 처음으로 버튼 | HOME |

- PREPARE, CAPTURE, CHECKING 상태에서는 초기화면 버튼 미표시
- PREPARE 상태에서 얼굴이 0명이면 "모든 얼굴이 웃음"으로 판정하지 않음
- 단계별 목적: PREVIEW 는 구도를 잡는 단계, PREPARE 는 미소를 짓는 단계, CAPTURE 는 사진을 찍는 단계
- 안내 대상 얼굴: 너무 작은 얼굴(높이 < `MIN_FACE_H`), 보이지 않는 얼굴(입이 안 보이거나 두 눈이 모두 안 보임. 옆모습, 가림. 한쪽 눈만 가려진 얼굴과 선글라스류는 보이는 얼굴)
- 안내 표시 안정화: 판정 1회의 오판으로 안내가 깜빡이거나 촬영 시작이 막히지 않도록
  - 판정 결과에 안내 대상이 `GUIDE_ON_COUNT` 회 연속 있으면 안내 표시 시작
  - 안내 대상이 `GUIDE_OFF_COUNT` 회 연속 없으면 안내 표시 끝
  - 안내 표시 중에는 PREVIEW 에서 손을 들어도 PREPARE 로 가지 않음
  - 얼굴 추적 없이 프레임 단위로 셈. 바운딩 박스는 가장 최근 판정의 안내 대상 얼굴에 그림
  - 근거: 보이는 얼굴을 보이지 않음으로 잘못 판정하는 비율이 얼굴당 약 3% (10절 5번). 5명이면 판정마다 약 14%, 10명이면 약 26% 확률로 한 명 이상 오판
- PREVIEW -> PREPARE 판단: 안내 표시 중이 아니고 `Prediction.ready_to_start(MIN_FACE_H)` (4절)
- 전환 안정화: 판정 1회의 오판으로 촬영이 시작되거나 촬영 단계로 넘어가지 않도록 (10절 19번)
  - 손 들기 (시작 가능) 가 `HAND_ON_COUNT` 회 연속이면 PREPARE, 모두 웃음이 `SMILE_ON_COUNT` 회 연속이면 CAPTURE
  - 조건이 한 번 끊기면 처음부터 다시 셈. 상태가 바뀌면 초기화
  - 얼굴 추적 없이 프레임 단위로 셈
- CHECKING: 찍은 사진마다 조건 충족 여부를 확인하는 동안 "사진을 고르는 중" 문구를 띄우는 상태
- MODEL_EVAL (dev 모드): 앱 흐름과 별개로 모델별 판정을 실시간으로 보는 화면. 판정 결과로 상태가 바뀌지 않음
  - 모델 선택 (`models.NAMES`), 판정 항목 선택 (smile, eye, hand, visibility). 판정 스레드가 고른 모델과 켠 항목으로 판정
  - 모델은 처음 고를 때 만들고 보관 (`model_eval.ModelEval`). 나가면 판정 스레드의 모델을 앱 모델(`config.MODEL`)로 되돌림
  - 모델, 항목을 바꾸기 전 요청으로 만든 판정은 버림
  - 찍기: 판정한 프레임과 판정 결과를 `{SAVE_DIR}/eval/{시각}.jpg`, `.json` 으로 저장 (모델, 요청 항목, 얼굴별 판정과 scores, 손, 단계별 시간)
    - 판정 스레드가 판정한 프레임을 결과와 함께 보관 -> 저장 사진과 판정이 같은 프레임
  - 연속 촬영: 연속 버튼으로 켜고 끔. 켜 두면 새 판정이 올 때 `EVAL_BURST_SEC` 간격으로 저장. 켤 때마다 `{SAVE_DIR}/eval/burst_{시각}/` 에 모음. 평가 화면을 나가면 꺼짐

## 2. 상태별 화면 요소

| 상태 | 표시 요소 |
|---|---|
| HOME | 앱 이름, 촬영 시작 버튼, 하단 경고 "N명 이상 촬영시 앱이 느려질 수 있습니다." (N = `WARN_PEOPLE`). 왼쪽 위 모드 전환 안내 "d: 개발자 모드 켜기" (dev 면 "끄기"). dev 모드면 모델 평가 버튼 |
| PREVIEW | 카메라 미리보기, 사람 수, 하단 문구 "손가락을 펴고 들면 사진 촬영이 시작됩니다.", 초기화면 버튼, 안내 대상 얼굴에 바운딩 박스와 문구 ("조금 더 가까이 와주세요" / "얼굴이 보이도록 해주세요") |
| PREVIEW (취소 직후) | 위 요소 + "웃지 않으신 분이 있어 촬영을 취소했어요. 준비되면 다시 손을 들어 촬영을 시작해주세요." (`CANCEL_MSG_SEC` 동안) |
| PREPARE | 카메라 미리보기, 남은 준비 시간, 웃지 않는 얼굴에 바운딩 박스와 "미소 지어주세요" |
| CAPTURE | 카메라 미리보기, 남은 촬영 시간 |
| CHECKING | "사진을 고르는 중" 문구 |
| 카메라 끊김 (PREVIEW, PREPARE, CAPTURE, MODEL_EVAL) | 상태별 화면 요소 대신 "카메라 연결을 확인해 주세요" (버튼은 그대로). 다시 연결되면 원래 화면 |
| RESULT | 조건을 충족한 사진 최대 5장. 없으면 조건에 가장 가까운 사진 최대 5장과 "모두 조건에 맞는 사진은 없어요. 가장 가까운 사진을 보여드려요" (얼굴이 있는 사진도 없으면 "조건에 맞는 사진이 없어요"), 미리보기 버튼, 초기화면 버튼 |
| MODEL_EVAL | 카메라 미리보기, 모델 선택 버튼, 판정 항목 버튼 (선택된 것은 초록), 얼굴 박스 (모든 항목 통과 초록, 아니면 빨강) 와 항목별 "점수/기준값 O\|X", 손 박스와 판정 이유, 판정 시간 (전체, 단계별 ms), 연속 버튼 (켜지면 초록, 저장한 장수 표시), 찍기 버튼, 처음으로 버튼 |

- 안내 문구는 PREPARE 의 "미소 지어주세요" 와 같은 방식 (얼굴 박스 위 띠). 한 얼굴이 두 조건에 모두 해당하면 "얼굴이 보이도록 해주세요" 우선
- 사람 수가 `WARN_PEOPLE` 이상이면 PREVIEW 의 사람 수 표시를 경고 색으로 변경. 촬영은 막지 않음
- 한글 문구는 PIL 로 그림 (OpenCV `putText` 는 한글 미지원)
- dev 모드: 카메라 화면에 검출된 얼굴 bbox 와 판정 점수를 항상 표시 (박스 아래 띠 "웃음 0.95 O", "눈 뜸 0.97 O". 판정한 항목만, 보이지 않으면 "얼굴 가림", 점수가 없으면 가림 점수 또는 "face". 모두 통과면 초록, 아니면 빨강. 안내, 미소 띠는 박스 위라 겹치지 않음. CAPTURE 는 dev 모드에서만 `DEV_CAPTURE_PREDICT_SEC` 간격 (약 5 FPS) 으로 보임, 웃음, 눈을 판정해 표시. prod 는 CAPTURE 중 판정 안 함)
- dev 모드: 카메라 화면에 손 bbox 표시 (든 손은 노란색, 아닌 손은 하늘색. 글자는 판정 이유 "open up", "fist", "not up")
- dev 모드: 모든 화면 오른쪽 아래에 실행 모드와 모델 표시 (예: `dev | model: landmark`)

## 3. 모듈 구성

| 파일 | 역할 |
|---|---|
| `app.py` | 메인 루프. 카메라 스레드, 판정 스레드(`Agent`), `states`, `ui` 연결 |
| `model_eval.py` | 모델 평가 화면(dev)의 선택 상태(모델, 판정 항목), 모델 보관, 찍기 저장 |
| `states.py` | 상태 전환 로직. 카메라, 화면과 분리. 시계(`now`) 주입 |
| `camera.py` | 카메라 스레드. 최신 프레임 1장 보관, CAPTURE 상태에서는 프레임 전부 전달, 연결 상태 확인과 재연결 (10절 6번) |
| `model.py` | 모델 인터페이스(`Face`, `Hand`, `Prediction`, `Model`). 구현은 `models/` |
| `models/__init__.py` | `create_model(name)`. 선택된 모델의 의존성만 import |
| `models/fake/` | 가짜 모델. 얼굴은 YuNet, 웃음, 눈, 손 들기는 키보드 흉내 |
| `models/landmark/` | 전체 화면 FaceLandmarker + HandLandmarker 모델, blendshape 룰(`rules.py`) |
| `models/yunet_landmark/` | YuNet 얼굴 검출 + 잘라낸 얼굴 FaceLandmarker 2단계 모델. 손은 landmark 와 같음 |
| `models/yunet_cnn/` | YuNet 얼굴 검출 + 얼굴별 CNN(smile, eye, occlusion, LiteRT) 모델. 손은 landmark 와 같음 |
| `models/util/` | 모델 공통 도구. 키보드 손 들기 흉내(`keyboard.py`), 손 검출과 손 들기 판정(`hand.py`) |
| `selection.py` | 결과 사진 선택 규칙 |
| `storage.py` | 세션 폴더 생성, 사진 저장, 결과 사진 외 삭제 |
| `ui.py` | 화면 그리기, 버튼 클릭 처리 |
| `wireframe.py` | 카메라 없이 상태별 화면을 PNG 로 저장. 모은 grid.png (README 화면 예시) |
| `metrics.py` | 시간 기록 로그 (JSON Lines). 9절 |
| `config.py` | 설정값 |

- `photo_app/` 밖의 `app/analysis/`: 로그 분석 (duckdb). 9절

## 4. 모델 인터페이스 (`model.py`)

- 프레임 1장 요청 -> 그 프레임에 대한 판정 결과 1개 반환
- 요청 인자는 "돌려받을 결과" 선택. 무엇을 실제로 계산할지는 구현이 결정
- 얼굴 bbox 는 항상 반환. 요청하지 않은 결과는 None

```python
@dataclass
class Face:
    bbox: tuple[int, int, int, int]   # x, y, w, h
    smile: bool | None                # 웃음 여부. smile 미요청이면 None
    eyes_open: bool | None            # 눈 뜸 여부. eye 미요청이면 None
    visible: bool | None              # 입과 눈(한쪽 이상)이 보이는지 (옆모습, 입 가림, 두 눈 가림이면 False. 한쪽 눈만 가림, 선글라스류는 True). visibility 미요청이면 None
    eye_score: float | None           # 눈 뜸 정도 (0~1, 클수록 뜸). eye 미요청 또는 판정 불가면 None
                                      # - yunet_cnn: eye CNN 점수 / landmark 계열: 1 - eyeBlink 최대 / fake: 뜸 1.0, 감음 0.0
                                      # - 결과 사진 선택에서 같은 촬영의 사진끼리 비교하는 용도 (척도는 모델마다 다름)
    occlusion: float | None           # 가림 점수 (0~1, 클수록 가림). yunet_cnn 이 visibility 요청 때 채움 (yaw 로 이미 보이지 않으면 None). 다른 모델은 None. dev 표시용
    scores: dict[str, tuple[float, float, bool]]   # 판정 항목별 (점수, 기준값, 통과). 실행한 항목만. 모델 평가 화면 표시, 찍기 저장용
                                      # - yunet_cnn: yaw, occ, smile, eye / landmark, yunet_landmark: yaw, smile, blink / fake: 없음
                                      # - 통과 = 보임 쪽, 웃음, 눈 뜸 쪽 판정

@dataclass
class Hand:
    bbox: tuple[int, int, int, int]   # x, y, w, h
    raised: bool                      # 손 들기 여부
    pose: str = ""                    # 판정 이유 (dev 표시용. "open up", "fist", "not up"). 없으면 빈 칸

@dataclass
class Prediction:
    faces: list[Face]                 # 항상 반환
    hands: list[Hand] | None          # hand 미요청이면 None
    hand_raised: bool | None          # 손 들기 여부 (키보드 흉내 포함). hand 미요청이면 None
    elapsed_ms: float                 # 요청 처리 시간
    requested: frozenset[str]         # 요청한 결과 {"smile", "eye", "hand", "visibility"}
    timings: dict[str, float]         # 단계별 시간(ms)

    def meets_condition(self) -> bool:
        # 사진의 조건 충족 여부: 얼굴 1명 이상, 모든 얼굴이 웃고 눈을 뜸
        # smile, eye 를 함께 요청하지 않은 결과면 ValueError
        ...

    def eye_score(self) -> float | None:
        # 사진의 눈 뜸 점수: 얼굴 중 가장 낮은 eye_score (눈을 가장 작게 뜬 사람 기준)
        # 얼굴이 없거나 점수가 없는 얼굴이 있으면 None
        ...

    def guidance(self, min_face_h: int) -> list[tuple[Face, str]]:
        # 안내 대상 얼굴과 이유 [(얼굴, "not_visible" | "too_small"), ...]
        # - "not_visible": visible 이 False
        # - "too_small": bbox 높이 < min_face_h
        # - 한 얼굴이 둘 다 해당하면 "not_visible" 만
        # visibility 를 요청하지 않은 결과면 ValueError
        ...

    def ready_to_start(self, min_face_h: int) -> bool:
        # 촬영 시작 가능 여부: 손 들기, 얼굴 1명 이상, 안내 대상 얼굴 없음
        # hand, visibility 를 함께 요청하지 않은 결과면 ValueError
        ...

class Model:
    def predict(self, frame, *, smile: bool, eye: bool, hand: bool, visibility: bool) -> Prediction: ...
```

- 호출하는 쪽은 `Model.predict` 만 사용. 내부 모델 종류는 모름
- 요청 인자는 기본값 없이 매번 지정
- 웃음, 눈 뜸, 손 들기, 보임의 판정 기준(임계값 등)은 `Model` 구현 내부에 둠
- 보이지 않는 얼굴(`visible=False`)은 웃음, 눈을 판정하지 않음 (`smile`, `eyes_open` 은 None). `meets_condition` 에서 미충족
- 선글라스류(선글라스, 색이 짙은 고글 등)로 눈이 보이지 않는 얼굴은 눈 조건 충족으로 처리. 투명한 안경은 눈을 그대로 판정. 인터페이스 표현은 10절 8번
- 얼굴 크기 기준(`MIN_FACE_H`)은 제품 요구사항이라 앱이 `guidance`, `ready_to_start` 에 넘김
- 상태별 요청

| 상태 | smile | eye | hand | visibility |
|---|---|---|---|---|
| PREVIEW | x | x | o | o |
| PREPARE | o | x | x | x |
| CHECKING | o | o | x | o |

- CHECKING 에서 visibility 요청: 옆모습, 가린 얼굴이 웃음, 눈 판정으로 조건 충족이 되지 않도록 (10절 11번 측정도 yaw 규칙을 켠 결과)
- 구현별 최적화 예
  - landmark: 얼굴마다 FaceLandmarker 1회 실행, 룰은 요청받은 결과만 계산. HandLandmarker 는 hand 요청 시에만 실행
  - 결과마다 모델이 따로 있는 경우: 요청받은 결과에 필요한 모델만 실행

### 모델 구현 (`models/`)

- `config.MODEL` 로 선택: `"fake"`, `"landmark"`, `"yunet_landmark"`, `"yunet_cnn"`
- fake
  - 얼굴: OpenCV YuNet. 모델 파일이 없으면 화면 가운데 가짜 얼굴 1개
  - 웃음, 눈, 손 들기, 보임: 키보드 흉내 (s: 웃음 토글, e: 눈 뜸 토글, h: 손 들기, o: 보이지 않음 토글)
- landmark
  - 얼굴: 전체 화면 FaceLandmarker. 랜드마크 최소, 최대 좌표로 bbox (작은 얼굴을 못 찾음. 10절 7번)
  - 보임: FaceLandmarker 랜드마크로 고개 돌림 yaw 계산, yaw >= `YAW_TH` 이면 `visible=False`
    - 눈 가운데는 눈꼬리 두 점의 평균, 코는 코끝. 기준값은 yunet_landmark 측정값을 그대로 씀 (landmark 기준 미측정)
  - 웃음: 양쪽 `mouthSmile` 평균 >= `SMILE_TH`
  - 눈 뜸: 양쪽 `eyeBlink` 중 큰 값 < `BLINK_TH` (한쪽만 감아도 감은 것으로 봄)
  - 손 들기: 손가락을 펴고 위를 향함 (높이는 보지 않음). 키보드 h 도 보조로 허용
    - 손가락 펴짐: 검지 ~ 새끼 중 `HAND_MIN_FINGERS` 개 이상 (끝이 두 번째 마디보다 손목에서 멂). 아니면 "fist"
    - 손바닥, 손등은 구분하지 않음 (2026-09-27 thinkcat). 손바닥 방향 판정(MediaPipe 왼손/오른손 이용)을 시도했으나 손등도 통과해 뺌
      - 영상 속 오른손 손등과 왼손 손바닥은 랜드마크 배치가 같아 랜드마크만으로 구분 불가
    - 위를 향함: 손목 -> 중지 끝 벡터의 위쪽 성분 비율 >= `HAND_UP_RATIO`. 아니면 "not up"
  - 실행 모드: IMAGE (판정 스레드와 CHECKING 이 같은 인스턴스를 써서 VIDEO 모드의 증가 타임스탬프 조건을 지킬 수 없음)
- yunet_landmark (기본 모델)
  - 얼굴: 2단계
    - YuNet 으로 얼굴 bbox 검출 (신뢰도 >= `SCORE_TH`. 10절 11번)
    - 얼굴마다 `CROP_MARGIN` 여백을 두고 잘라 FaceLandmarker 실행 (전체 화면을 넣으면 작은 얼굴을 못 찾음. 10절 7번)
  - 보임: YuNet 랜드마크의 고개 돌림 yaw >= `YAW_TH` 이면 `visible=False` (10절 5번)
    - yaw = |코 x - 두 눈 가운데 x| / 두 눈 사이 거리
    - YuNet 결과만 쓰므로 추가 모델 실행 없음 (계산은 얼굴당 약 5us, 맥). 다른 모델(CNN 등)로 바꿔도 같은 방식 사용
  - 보이지 않는 얼굴은 FaceLandmarker 를 실행하지 않음 (웃음, 눈은 None)
  - FaceLandmarker 가 잘라낸 얼굴에서 얼굴을 못 찾으면 웃음, 눈은 None (판정 불가 -> 조건 미충족)
  - 웃음, 눈 뜸, 손 들기: landmark 와 같은 룰. 기준값은 yunet_landmark 설정에 따로 둠
- yunet_cnn
  - 얼굴: YuNet (신뢰도 >= `SCORE_TH`)
  - 보임: YuNet yaw < `YAW_TH` 이고 occlusion CNN 가림 점수 < `OCC_TH` 이면 `visible=True` (10절 12번)
    - yaw 로 이미 보이지 않으면 occlusion CNN 은 실행하지 않음. visibility 미요청이면 실행하지 않음 (PREPARE)
    - 보이지 않는 얼굴은 smile, eye CNN 을 실행하지 않음 (웃음, 눈은 None)
  - CNN 입력: YuNet 박스 중심 정사각형 (한 변 = 박스 긴 변, 화면 밖은 가장자리 복제), RGB 0~255
    - occlusion: 정사각형 전체를 `OCC_SIZE` 로
    - smile: 정사각형 아래쪽 `SMILE_BOTTOM` 을 `SMILE_SIZE` 로 늘림 (눈 영역을 넣지 않음. 선글라스 렌즈 반사를 웃음으로 보는 문제, 10절 17번)
    - eye: 정사각형 위쪽 `EYE_TOP` 을 `EYE_SIZE` 로 늘림
  - 웃음: smile CNN 점수 >= `SMILE_TH` / 눈 뜸: eye CNN 점수 >= `EYE_TH`
    - `SMILE_TH` 는 웃었는데 안 웃음으로 보는 경우를 줄이는 쪽으로 정한 값 (10절 14번)
    - `EYE_TH` 는 감은 눈 오판(감은 눈을 뜸으로)을 줄이는 쪽으로 높인 값 (10절 13번)
    - 선글라스류는 eye CNN 이 뜸으로 판정 (합성 선글라스로 학습. 10절 8번)
  - 좌우 뒤집기 평균은 모델 안에 포함
  - 요청받은 결과의 CNN 만 실행 (PREPARE 는 smile 만)
  - 추론: LiteRT (`ai-edge-litert`) Interpreter, `NUM_THREADS`. 모델 파일은 fp32 (평가와 같은 파일. 라즈베리파이 측정 후 dynamic, int8 검토)
  - 손: landmark 와 같은 HandLandmarker + 룰 (`models/util/hand.py`)
- 키보드 흉내는 dev 모드에서만 동작

## 5. 실행 구조 (스레드)

- 카메라 스레드: 20 FPS 이상으로 프레임 읽기
- 판정 스레드(`Agent`): 최신 프레임만 가져와 `Model.predict` 호출 (5 FPS 이상). 결과는 타임스탬프와 함께 보관
  - PREVIEW, PREPARE 상태에서만 동작. 상태별 요청 인자는 4절 표
- UI 스레드(메인): 15 FPS 이상으로 화면 그리기. 바운딩 박스는 가장 최근 판정 결과로 그림
- CAPTURE 상태: 판정 없이 프레임 저장만. 단, dev 모드는 화면 표시용으로 `DEV_CAPTURE_PREDICT_SEC` 간격 판정 (판정 결과로 상태가 바뀌지 않음)
- CHECKING 상태: 모든 사진에 `predict(smile=True, eye=True, hand=False, visibility=True)` 후 `meets_condition()` 확인

### 스레드와 GIL

- 파이썬 GIL 때문에 파이썬 코드는 한 번에 한 스레드만 실행
- 카메라 읽기, OpenCV 연산, LiteRT 추론은 C/C++ 실행 중 GIL 을 풀어 여러 코어에서 동시 실행 가능
  - LiteRT 의 GIL 해제 여부는 사용할 버전에서 확인
- LiteRT(`num_threads`), OpenCV 는 호출 하나 안에서도 내부 스레드로 여러 코어 사용
- 전처리, 후처리는 파이썬 반복문 대신 numpy, OpenCV 로 처리
  - 파이썬 반복문은 GIL 을 잡고 있어 다른 스레드를 멈춤

## 6. 설정값 (`config.py`)

| 이름 | 기본값 | 설명 |
|---|---|---|
| `MODE` | `prod` | 시작 실행 모드. 환경변수 `PHOTO_APP_MODE` 로 변경. 앱에서 d 키로 dev, prod 전환 (모드 전환은 `mode` 로그). dev: 얼굴 bbox 와 판정 점수 (웃음, 눈 뜸 O/X), 손 bbox, 모델 이름, 모델 평가 버튼, 키보드 흉내 / prod: 모두 끔 |
| `APP_NAME` | 모두 웃는 단체 사진 | 초기화면 앱 이름 |
| `SCREEN_W`, `SCREEN_H` | 640, 480 | 화면 크기 |
| `SCREEN_FPS` | 30 | 화면 갱신 상한 |
| `CAMERA_INDEX` | 0 | 카메라 장치 번호 |
| `CAMERA_W`, `CAMERA_H`, `CAMERA_FPS` | 640, 480, 30 | 카메라 요청 해상도, FPS |
| `CAMERA_LOST_SEC`, `CAMERA_RETRY_SEC` | 2.0, 1.0 | 새 프레임이 이 시간(초) 이상 없으면 카메라 끊김 / 끊긴 동안 다시 여는 간격(초) |
| `PREPARE_SEC` | 10 | 준비 제한 시간 |
| `CAPTURE_SEC` | 2 | 촬영 제한 시간 |
| `CANCEL_MSG_SEC` | 3 | 취소 문구 표시 시간 |
| `MAX_RESULT` | 5 | 결과로 보여주고 남길 최대 사진 수 |
| `WARN_PEOPLE` | 측정 후 결정 | 경고를 띄우는 인원 수 (제한 아님) |
| `SAVE_DIR` | `app/photos/` | 사진 저장 위치 |
| `LOG_DIR` | `app/logs/` | 시간 기록 로그 위치 |
| `MODEL` | `yunet_landmark` | 사용할 모델: `fake`, `landmark`, `yunet_landmark`, `yunet_cnn` |
| `MODEL_DIR` | `app/models/` | 모델 파일(가중치) 위치 |
| `NUM_FACES` | 10 | 최대 검출 인원 (제품 요구사항) |
| `MIN_FACE_H` | 50 | 최소 얼굴 높이 (카메라 프레임 px). 미만이면 PREVIEW 에서 안내 (제품 요구사항). 근거 10절 5번 |
| `GUIDE_ON_COUNT` | 3 | 안내 대상이 이 횟수만큼 연속 판정되면 안내 표시 시작 (판정 5 FPS 에서 약 0.6초) |
| `DEV_CAPTURE_PREDICT_SEC` | 0.2 | dev 모드 CAPTURE 중 표시용 판정 간격 (초, 약 5 FPS). 촬영 FPS 가 줄 수 있어 FPS 측정은 prod 로 |
| `GUIDE_OFF_COUNT` | 2 | 안내 대상 없음이 이 횟수만큼 연속 판정되면 안내 표시 끝 |
| `HAND_ON_COUNT` | 2 | 손 들기가 이 횟수만큼 연속 판정되면 PREPARE 로 (판정 5 FPS 에서 약 0.4초) |
| `SMILE_ON_COUNT` | 2 | 모두 웃음이 이 횟수만큼 연속 판정되면 CAPTURE 로 |
| `EVAL_BURST_SEC` | 0.5 | 모델 평가 화면(dev) 연속 촬영 간격 (초) |
| `FONT_PATHS` | 저장소 `app/fonts/NanumGothic-Regular.ttf`, Debian NanumGothic, macOS AppleSDGothicNeo | 한글 문구 폰트 후보. 앞에서부터 존재하는 파일 사용. 저장소 폰트는 한글 폰트가 없는 라즈베리파이에서 글자가 깨져 포함 (나눔고딕, SIL OFL. `app/fonts/OFL.txt`) |

### 모델별 설정 (`models/<이름>/config.py`)

- 한 모델만 쓰는 값은 모델 폴더에서 관리. 모델 추가, 삭제 시 전역 `config.py` 를 수정하지 않기 위함
- 규칙
  - 모델이 쓰는 설정값은 모두 자기 폴더의 `config.py` 에 둠
  - 다른 모델의 config 를 import 하지 않음. 같은 값이 필요하면 자기 config 에 따로 적음 (한 모델의 값을 바꿀 때 다른 모델이 따라 바뀌지 않도록)
  - 전역 `config.py` 는 앱 전체 값(`MODEL_DIR`, `NUM_FACES` 등)만. 모델 config 는 전역 config 를 참조 가능
  - 판정 코드(`rules.py` 등)는 다른 모델과 같이 써도 됨. 임계값은 import 하지 않고 인자로 넘김

| 파일 | 이름 | 기본값 | 설명 |
|---|---|---|---|
| `models/landmark/config.py` | `FACE_MODEL_PATH` | `app/models/face_landmarker.task` | FaceLandmarker 모델 파일 |
| `models/landmark/config.py` | `HAND_MODEL_PATH` | `app/models/hand_landmarker.task` | HandLandmarker 모델 파일 |
| `models/landmark/config.py` | `NUM_HANDS` | 4 | 최대 검출 손 수 |
| `models/landmark/config.py` | `SMILE_TH` | 0.5 | 웃음 판정 임계값 |
| `models/landmark/config.py` | `BLINK_TH` | 0.5 | 눈 뜸 판정 임계값 |
| `models/landmark/config.py` | `YAW_TH` | 0.5 | 보임 판정. 고개 돌림 yaw 가 이 값 이상이면 보이지 않음 (yunet_landmark 측정값) |
| `models/landmark/config.py` | `HAND_UP_RATIO` | 0.7 | 손 들기 판정. 손목 -> 중지 끝 방향의 위쪽 성분 비율 (수직에서 약 45도 이내) |
| `models/landmark/config.py` | `HAND_MIN_FINGERS` | 3 | 손 들기 판정. 편 손가락 수 (검지 ~ 새끼 중). 미만이면 주먹 |
| `models/yunet_landmark/config.py` | `YUNET_PATH` | `app/models/face_detection_yunet_2023mar.onnx` | 얼굴 검출 (1단계) |
| `models/yunet_landmark/config.py` | `SCORE_TH` | 0.7 | YuNet 얼굴 신뢰도 기준. 미만이면 버림 (10절 11번) |
| `models/yunet_landmark/config.py` | `CROP_MARGIN` | 0.5 | 얼굴을 잘라낼 때 여백 (얼굴 크기 비율) |
| `models/yunet_landmark/config.py` | `YAW_TH` | 0.5 | 보임 판정. 고개 돌림 yaw 가 이 값 이상이면 보이지 않음 (10절 5번) |
| `models/yunet_landmark/config.py` | `FACE_MODEL_PATH`, `HAND_MODEL_PATH`, `NUM_HANDS`, `SMILE_TH`, `BLINK_TH`, `HAND_UP_RATIO`, `HAND_MIN_FINGERS` | landmark 와 같은 값 | landmark 와 같은 뜻. 따로 관리 |
| `models/yunet_cnn/config.py` | `MODEL_VERSION` | `v0.0.7` | CNN 파일 버전. 파일은 `app/models/{버전}/` (`docs/model_versions.md`) |
| `models/yunet_cnn/config.py` | `SMILE_PATH`, `EYE_PATH`, `OCC_PATH` | `{버전}/smile_fp32.tflite`, `{버전}/eye_fp32.tflite`, `{버전}/occlusion_fp32.tflite` | smile, eye, occlusion CNN |
| `models/yunet_cnn/config.py` | `SMILE_SIZE`, `EYE_SIZE`, `OCC_SIZE`, `EYE_TOP`, `SMILE_BOTTOM` | 128, 160, 128, 0.6, 0.45 | CNN 입력 크기, eye 입력의 얼굴 위쪽 비율, smile 입력의 얼굴 아래쪽 비율 |
| `models/yunet_cnn/config.py` | `SMILE_TH` | 0.31 | 웃음 판정 기준. 웃음을 놓치지 않는 쪽 (10절 14번). v0.0.6 smile 에 맞춰 0.5 에서 바꿈 (10절 17번) |
| `models/yunet_cnn/config.py` | `EYE_TH` | 0.84 | 눈 뜸 판정 기준. 감은 눈 오판을 줄이는 쪽 (10절 13번). v0.0.6 eye 에 맞춰 0.8 에서 바꿈 (10절 17번). v0.0.7 도 0.84 (10절 18번) |
| `models/yunet_cnn/config.py` | `OCC_TH` | 0.86 | 보임 판정. 가림 점수가 이 값 이상이면 보이지 않음. WIDER dev 보이는 얼굴 오판 2% 기준 (10절 12번) |
| `models/yunet_cnn/config.py` | `NUM_THREADS` | 4 | LiteRT 추론 스레드 수 |
| `models/yunet_cnn/config.py` | `YAW_TH` | 0.6 | 보임 판정. 고개 돌림 yaw 가 이 값 이상이면 보이지 않음 (10절 15번) |
| `models/yunet_cnn/config.py` | `YUNET_PATH`, `SCORE_TH`, `HAND_MODEL_PATH`, `NUM_HANDS`, `HAND_UP_RATIO`, `HAND_MIN_FINGERS` | yunet_landmark 와 같은 값 | 같은 뜻. 따로 관리 |
| `models/fake/config.py` | `YUNET_PATH` | `app/models/face_detection_yunet_2023mar.onnx` | 얼굴 검출. 없으면 가운데 가짜 얼굴 1개 |
| `models/fake/config.py` | `SCORE_TH` | 0.7 | YuNet 얼굴 신뢰도 기준. OpenCV 기본값 0.9 는 멀어진 얼굴을 "작음" 안내 전에 놓침 (10절 11번) |

- `app/models/` 는 앱 실행에 필요한 모델 (YuNet, FaceLandmarker, HandLandmarker, yunet_cnn 현재 버전) 만 커밋. 이전 버전 CNN 은 팀 드라이브

## 7. 결과 사진 선택 규칙

- `ok`: 조건을 충족한 사진을 찍힌 순서로 나열한 목록 (m장). 사진마다 판정 결과 `Prediction` 을 함께 넘김
  - 점수 계산(어떤 사진이 더 좋은가)은 선택 규칙 안에서. 지금은 눈 뜸 점수 `Prediction.eye_score()`
- n: 결과 장수 (`MAX_RESULT`)
- m = 0 (조건 충족 사진 없음): 조건에 가장 가까운 사진으로 대신 (`select_closest`)
  - 후보: 얼굴이 1명 이상 검출된 사진 전부 (k장). k = 0 이면 결과 사진 없음
  - 가까운 정도 (`closeness`, 사진마다. 큰 쪽이 가까움)
    1. 조건을 충족한 얼굴 수 (보임, 웃음, 눈 뜸 모두 통과)
    2. 가장 모자란 값: 얼굴마다 웃음, 눈 점수의 기준 대비 차이 (`Face.scores` 의 점수 - 기준. 눈 감음 점수 blink 처럼 작을수록 통과인 항목은 기준 - 점수) 중 최솟값 (`face_margin`)
       - 보이지 않거나 웃음, 눈을 판정하지 못한 얼굴은 -1. 점수가 없는 모델 (fake) 은 통과 0, 미통과 -1
       - 척도가 다른 웃음, 눈 점수를 기준 대비 차이로 합쳐 비교 (thinkcat)
  - 선택: k <= n 이면 전부. k > n 이면 후보를 찍힌 순서로 n개 구간으로 나누고, 구간마다 가장 가까운 사진 1장 (같으면 먼저 찍힌 사진. 비슷한 연속 사진만 고르지 않도록)
  - 결과 화면 제목: "모두 조건에 맞는 사진은 없어요. 가장 가까운 사진을 보여드려요" (`View.results_closest`)
  - 근거 (2026-09-27 thinkcat): 웃느라 가늘어진 눈을 감음으로 보는 오판 (eye 기준 0.84, 학습 목표에서 웃는 눈 0.7) 등으로 조건 충족 사진이 없을 때도 사용자가 찍은 사진을 받을 수 있도록
- m <= n: `ok` 전부 선택
- m > n: `ok` 를 찍힌 순서대로 장수 기준 n개 구간으로 나누고, 구간마다 눈 뜸 점수가 가장 높은 사진 1장 (같으면 먼저 찍힌 사진)
  - i번째 구간 = `ok[floor(i*m/n) : floor((i+1)*m/n)]`
- 근거 (2026-09-26)
  - 눈을 감으려는 순간(반쯤 감은 눈)이 조건 충족으로 결과에 포함된 사례. eye CNN, blendshape 모두 절대 기준으로는 뜸
  - 웃을 때 눈 모양(^^, --)은 사람마다 달라 사진 한 장으로 "반쯤 감은 눈" 을 정의하기 어려움 (thinkcat) -> 같은 촬영 안에서 비교
  - 사진 점수는 눈만 사용. 웃음은 약한 미소도 허용하기로 해서 (10절 14번) 웃음 점수로 고르지 않음

```python
def select(ok: list[tuple[T, Prediction]], n: int) -> list[T]:
    # ok 는 meets_condition() 을 통과한 사진만 (눈 뜸 점수가 항상 있음)
    m = len(ok)
    if m <= n:
        return [item for item, _ in ok]
    bounds = [i * m // n for i in range(n + 1)]
    return [max(ok[a:b], key=lambda x: x[1].eye_score())[0] for a, b in zip(bounds, bounds[1:])]
```

## 8. 사진 저장 규칙

- CAPTURE 상태에서 찍은 사진: `photos/{YYYYmmdd_HHMMSS}/frame_{번호:03d}.jpg` 로 저장
- RESULT 상태를 벗어나면 결과로 보여준 사진만 남기고 나머지 삭제

## 9. 측정 항목과 목표

- 먼저 측정하고, 목표에 못 미치는 항목을 최적화 대상으로 삼음

| 항목 | 측정 방법 | 기록 (로그 이벤트) | 목표 |
|---|---|---|---|
| 카메라 FPS | 카메라 스레드의 초당 프레임 수 | `camera_fps` | 20 이상 |
| 화면 FPS | UI 스레드의 초당 그리기 횟수 (로컬 PC 에서 보이는 기준) | `screen_fps` (상태별) | 15 이상 |
| 판정 FPS | 판정 스레드의 초당 `predict` 호출 수 | `predict` (caller=agent) 의 시각으로 계산 | 5 이상 |
| 판정 지연 | `Prediction.elapsed_ms`, 인원 수(1, 2, 4, 6, 8명)별 | `predict` 의 `total_ms`, 단계별 `*_ms`, `faces` | 200 ms 이하 |
| CHECKING 대기 시간 | CAPTURE 종료부터 RESULT 표시까지 | `checking` 의 `total_ms` | 1초 이하 |
| 촬영 장수 | CAPTURE 한 번에 저장된 사진 수 | `capture` 의 `frames`, `fps` | 40장 이상 (20 FPS x 2초) |
| 상태 전환 반응 시간 | 손 들기 -> PREPARE, 모두 웃음 -> CAPTURE 까지 걸린 시간 | `state` (전환 시각만. 반응 시간 계산은 미구현) | 측정 후 결정 |
| 사진 저장 시간 | 사진 1장 저장에 걸린 시간 | `checking` 의 `save_ms` / `frames` | 측정 후 결정 |
| CPU, 메모리 사용량 | 상태별 평균과 최대. CPU 는 코어별로 기록 | 미구현 | 측정 후 결정 |

- `WARN_PEOPLE`: 판정 지연이 200 ms 를 넘는 최소 인원 수

### 측정 기록 (`metrics.py`)

- 형식: JSON Lines. 한 줄에 이벤트 1개 `{"t": epoch 초, "ev": 이벤트명, ...}`
- 파일: 실행마다 1개, `app/logs/{YYYYmmdd_HHMMSS}.jsonl` (커밋하지 않음)
- 앱 시작 시 로그 파일 경로를 터미널에 출력

| 이벤트 | 기록 시점 | 필드 |
|---|---|---|
| `run` | 실행 시작 | `mode` (시작 모드), `model`, `screen`, `camera` (요청 해상도), `camera_actual` (실제로 열린 해상도) |
| `mode` | d 키로 모드 전환 | `mode` (dev/prod) |
| `predict` | 추론 1회 | `caller` (agent/check), `state`, 요청 `smile`/`eye`/`hand`/`visibility`, 안내 대상 얼굴 수 `guided`와 이유별 `guided_not_visible`/`guided_too_small` (visibility 요청 시), `faces`, 얼굴 높이 목록 `face_h`, `hands`, `total_ms`, 단계별 `face_ms`/`hand_ms`/`smile_ms`/`eye_ms` |
| `capture` | 촬영 종료 | `frames`, `duration_s`, `fps` |
| `checking` | 판정 종료 | `frames`, `ok`, `picked`, `fallback` (조건 충족 사진이 없어 가까운 사진으로 대신했는지), `save_ms`, `predict_ms`, `select_ms`, `total_ms` |
| `camera_fps` | 1초마다 | `fps` |
| `camera_lost` | 카메라 끊김 (시작할 때 카메라가 없을 때 포함) | `index` |
| `camera_restored` | 끊긴 뒤 다시 프레임을 받음 | - |
| `screen_fps` | 1초마다 | `fps`, `state` |
| `state` | 상태 전환 | `from`, `to` |

- 단계별 시간: `Prediction.timings` 를 모델 구현이 채움
  - landmark: `face` (FaceLandmarker), `hand` (HandLandmarker), `smile`, `eye` (룰 비교만. 실제 계산은 `face` 에 포함)
  - yunet_landmark: `face` (YuNet), `landmark` (잘라낸 얼굴 FaceLandmarker 합계), `hand`, `smile`, `eye` (룰 비교만)
  - yunet_cnn: `face` (YuNet, 얼굴 자르기 포함), `occlusion`, `smile`, `eye` (CNN 실행 합계), `hand`
  - fake: `face` (YuNet)

### 로그 분석 (`app/analysis/`)

- duckdb 로 로그를 읽어 요약 표 출력. 모든 이벤트는 `events` 테이블 하나 (없는 필드는 NULL, `filename` 컬럼으로 실행 구분)

```bash
cd app
.venv/bin/python -m analysis                                   # logs/*.jsonl 전체 요약
.venv/bin/python -m analysis logs/20260924_190200.jsonl        # 파일 지정
.venv/bin/python -m analysis --sql "select ev, count(*) from events group by ev"
```

- 요약 표: 요청 조합별 추론 시간 (avg, p50, p95, max), 단계별 평균, 얼굴 수별, 촬영, 판정, 카메라 FPS, 화면 FPS (상태별), 실행 목록, 상태 전환 횟수
- 기록이 없는 표는 "(기록 없음)"

## 10. 구현 결정에 필요한 사항

1. 손 들기 판정 방법: landmark 모델에 HandLandmarker 적용 (4절). 판정 정확도는 실제 촬영으로 확인 필요
2. CHECKING 대기 시간이 1초를 넘을 때의 최적화 방법 (측정 후 결정)
   - 적용: `predict(frame, *, smile, eye, hand)` 로 필요한 결과만 요청 (4절). CHECKING 은 hand 미요청
   - 맥 측정 (landmark, 노이즈 프레임 30회 평균): 손 포함 26.1 ms, 손 제외 2.7 ms
   - 라즈베리파이 측정 후 1초 초과 시 추가 최적화
3. 화면 전송 방식 (현재 X 포워딩)
4. 판정을 별도 프로세스로 분리할지 (측정 후 결정)
   - 스레드로 먼저 구현하고 9절 항목 측정
   - 판정을 켰을 때 카메라 FPS 나 화면 FPS 가 목표 아래로 떨어지고, CPU 사용률이 코어 하나에만 몰려 있으면 GIL 경합으로 판단
   - 이 경우 판정 스레드를 `multiprocessing` 프로세스로 분리. 프레임은 공유 메모리로 넘겨 복사 비용 절감
5. 최소 얼굴 높이, 보임 판정 방법 결정 (측정 기록, 2026-09-26)
   - WIDER FACE 라벨 데이터 dev 분할 (사진 515장, 얼굴 1,905개), 가로 640, YuNet score >= 0.7
   - 얼굴 높이는 WIDER 정답 박스 기준. 앱은 YuNet 박스 높이를 씀
     - 두 높이는 거의 같음: YuNet / 정답 높이 중앙값 0.98~1.02 (얼굴 크기 구간별), 정답 45~55px 얼굴의 YuNet 높이 중앙값 48.7px
     - YuNet 박스에 50 을 그대로 쓰면 정답 50px 미만 얼굴의 93% (431/462) 를 작음으로 잡고, 50px 이상 얼굴의 6% (49/875) 를 작음으로 잘못 잡음 (대부분 기준 근처)
     - 기준 근처에서 판정이 흔들리는 것은 안내 표시 안정화(1절)가 흡수. 50 을 그대로 씀
     - 측정 스크립트: `evaluation/wider/box_height_check.py`
   - 최소 얼굴 높이: 50px
     - 검출, 웃음 판정은 30px 이상이면 큰 차이 없음. 눈 판정이 기준을 정함
     - blendshape 눈 판정에서 뜬 눈을 감음으로 오판하는 비율: 30~40px 27%, 40~50px 22%, 50~60px 13%, 60~80px 9%, 80px 이상 6%
     - 카메라 640x480, 가로 화각 약 62도 가정 시 50px 는 약 2.3m, 화면 가로 폭 약 2.8m
   - 보임 판정: YuNet 고개 돌림 yaw >= 0.5
     - 검출된 보이지 않는 얼굴 400개, 보이는 얼굴 1,337개 기준
     - yaw >= 0.5: 보이지 않는 얼굴 중 66% 를 잡음, 보이는 얼굴을 잘못 잡음 3%. 옆모습만 보면 AUC 0.963
     - FaceLandmarker 미검출 방식: 65% / 3% 로 비슷하나 얼굴당 약 14ms 추가 (yaw 는 추가 모델 실행 없이 얼굴당 약 5us)
     - 둘을 함께 쓰면 84% / 6% (다른 사람, 물건에 가린 얼굴까지 잡음). 비용 대비 제외
   - 측정 스크립트: `evaluation/wider/requirements_check.py`
6. 카메라 연결 상태 지속 확인 (구현, 2026-09-27)
   - 실행 중 카메라 연결을 계속 확인 (`camera.py`)
   - 결정 (thinkcat)
     - 끊김 판단: 새 프레임이 `CAMERA_LOST_SEC` (2초) 이상 오지 않음. 시작 직후 첫 프레임을 기다리는 동안은 끊김이 아님
     - 화면 안내: 카메라 화면 자리에 "카메라 연결을 확인해 주세요". 끊기면 최신 프레임을 비워 멈춘 화면으로 판정하지 않음
     - 재연결: 끊긴 동안 `CAMERA_RETRY_SEC` (1초) 마다 장치를 닫았다 다시 엶. 새 프레임이 오면 원래 화면으로
     - 촬영 중 끊김 (PREPARE, CAPTURE): 촬영을 취소하고 PREVIEW 로. 찍던 프레임은 버림
     - 시작할 때 카메라가 없어도 앱을 멈추지 않고 같은 안내와 재연결
     - 기록: `camera_lost`, `camera_restored` 이벤트 (9절)
   - 테스트: 카메라를 여는 함수와 시계를 주입해 카메라 없이 끊김, 재연결 흉내
7. landmark 얼굴 검출을 2단계로 바꾼 근거 (측정 기록, 2026-09-25)
   - WIDER FACE val 단체 사진 후보 114장 (얼굴 502개), 가로 640 으로 축소
   - 전체 화면 FaceLandmarker: 얼굴 86개 (17%) 검출. 얼굴 높이 80px 미만에서는 거의 못 찾음
   - YuNet: 339개 (68%)
   - IoU 매칭이 아닌 사진별 min(검출 수, 정답 수) 합으로 비교한 대략적인 값
8. 선글라스류로 가려진 눈의 처리 (모델 교체 시 결정)
   - 요구사항: 선글라스류(선글라스, 색이 짙은 고글 등)로 눈이 보이지 않으면 눈 조건 충족으로 처리. 투명한 안경은 눈을 그대로 판정
   - 인터페이스 표현 후보: `eyes_open=True` 로 돌려줌 / `Face.eyewear` 같은 별도 필드 추가
   - 측정 (WIDER dev, 선글라스류 얼굴 38개): 눈 뜸 판정 비율 blendshape 91%, eye CNN v0.0.1 58%
   - eye CNN 은 선글라스 얼굴을 "뜸" 으로 학습해야 함
   - 결정 (yunet_cnn): `eyes_open=True` 로 돌려줌. eye CNN v0.0.2 가 합성 선글라스로 학습해 선글라스류를 뜸으로 판정 (WIDER dev 36/36)
     - 기준 0.8 에서는 뜸 판정이 줄어듦: v0.0.3 WIDER dev 23/36, holdout 28/38 (13번)
9. 일부러 가린 얼굴 (한계, 추후 결정)
   - 머리 스타일, 안대 등 본인이 치우기 어렵거나 일부러 가린 얼굴도 "보이지 않음" 으로 안내됨 -> 촬영을 시작할 수 없음
   - 모델은 일부러 가렸는지, 실수로 가렸는지 구분할 수 없음
   - 현재: 예외 없음 (선글라스류만 예외)
   - 결정 (2026-09-27 thinkcat): 한쪽 눈만 가려진 얼굴은 보이는 얼굴로 봄. 보임 기준을 "입, 그리고 눈 한쪽 이상" 으로 변경 (biz_spec 비기능 5)
     - 이유 (thinkcat): 두 눈이 모두 가려지는 경우는 드물고, 한쪽 눈 가림(앞머리 등)은 꽤 있으며 본인 스타일일 수 있음
     - 확인: 한쪽 눈이 가려진 얼굴도 eye 모델 v0.0.3 이 대체로 뜸으로 판정 (COFW 한쪽 눈 5점 가림 74장 중 62% 가 0.8 이상. 가림이 거의 없는 COFW 얼굴은 78%)
     - 감수: 앞사람에게 얼굴 한쪽이 가려진 사람도 통과 -> 결과 사진에 반쯤 가려진 얼굴이 나올 수 있음
   - 라벨링 데이터의 `hidden_reason` 으로 발생 빈도 확인 후 결정
     - 전체 라벨 (보이지 않는 얼굴 1,079개): 머리카락 12, 마스크 1. 드묾
   - 후보
     - 가려진 눈은 이유와 상관없이 면제 (실수로 가린 눈도 통과)
     - 안내가 N초 계속되면 의도한 가림으로 보고 허용
     - 안내 중 손을 들면 "이대로 촬영" 으로 허용 (추가 동작 없음)
   - 예외를 두면 필요한 것
     - 허용된 얼굴을 PREVIEW, PREPARE, CAPTURE 사진 전체에서 같은 얼굴로 이어서 찾는 얼굴 추적 (현재는 프레임마다 독립 판정)
     - 허용된 얼굴은 보이는 부분만 판정 (PREPARE 의 모두 웃음, `meets_condition`)
     - `meets_condition` 이 허용된 얼굴 정보를 받도록 인터페이스 변경
10. 검출되지 않는 가린 얼굴 (한계)
   - 보이지 않는 얼굴의 약 25% 는 YuNet 이 얼굴로 찾지 못함 (WIDER dev 533개 중 133개). 다른 사람에 가린 얼굴 42%, 기타 가림 39% 가 미검출
   - 찾지 못한 얼굴은 안내도 없고 조건 충족 판정에서도 빠짐 -> 그 사람이 가려진 사진이 조건 충족으로 고려될 수 있음
   - 모델로 해결하기 어려워 한계로 둠

11. YuNet 얼굴 신뢰도 기준 `SCORE_TH`: 0.7 (측정 기록, 2026-09-26)
   - WIDER dev 사진 515장 (조건 충족 138장), 가로 640, yunet_landmark, yaw >= 0.5 는 보이지 않음
   - 사진 조건 충족 판정. 보이는 얼굴 중 판단 불가(-1) 라벨이 있는 사진 47장은 정답을 모름 -> 두 가지로 계산
     - 0.9: precision 57/79 (72%), recall 57/138 (41%) / -1 사진 제외 시 57/77 (74%), 57/138 (41%)
     - 0.7: precision 49/55 (89%), recall 49/138 (36%) / -1 사진 제외 시 같음 (49/55, 49/138)
   - precision 을 우선함
     - precision 이 낮으면 조건에 맞지 않는 사진(눈 감음, 안 웃음)을 결과로 보여줌 -> 앱의 약속을 어김
     - recall 이 낮으면 조건에 맞는 사진 일부를 놓침. 촬영 한 번에 약 40장을 찍고 결과는 최대 5장이라, 일부를 놓쳐도 결과 사진은 나올 수 있음
   - 0.9 에서 잘못 통과한 사진은 주로 얼굴을 못 찾아 판정에서 빠진 경우 (가린 얼굴, 작은 얼굴)
   - 측정 스크립트: `evaluation/wider/photo_compare.py`

12. 가려진 얼굴(손, 다른 사람, 물건)의 보임 판정
   - 결정 (yunet_cnn, 2026-09-27): occlusion CNN (v0.0.4) 을 yaw 규칙과 함께 사용. `OCC_TH` 0.59 (학습 노트북 기준값)
     - 가림 기준은 10절 9번의 변경 기준 (입이 안 보이거나 두 눈이 모두 안 보임). WIDER 가림 라벨도 이 기준으로 재검수
     - WIDER, YuNet 0.7 이상 검출된 가린 얼굴 (dev 45, holdout 46), 보이는 얼굴 (dev 1,376, holdout 1,440)

| 방식 | AUC (dev / holdout) | 가림 recall (dev / holdout) | 보이는 얼굴 오판 (dev / holdout) |
|---|---|---|---|
| YuNet 신뢰도 < 0.8 | 0.897 / 0.917 | 31% / 20% | 1.7% / 1.9% |
| occlusion CNN, `OCC_TH` 0.59 | 0.905 / 0.912 | 29% / 50% | 1.7% / 1.4% |
| CNN 또는 YuNet (각각 dev 보임 오판 1% 기준) | - | 36% / 52% | 2.0% / 2.4% |

     - CNN 단독을 고른 이유: 보이는 얼굴 오판이 같은 수준에서 holdout 가림을 두 배 넘게 잡음. 합친 규칙은 기준값이 둘이고 dev 에서 3명 차이라 확실한 개선으로 보기 어려움
     - 두 방식은 다른 얼굴을 잡음: CNN 은 또렷한 얼굴의 입을 가린 손, 앞사람, 물건 / YuNet 신뢰도는 헬멧, 가면, 흐리거나 어두운 얼굴 (얼굴처럼 보이는 정도라 가림과 직접 관련 없음)
     - 못 잡는 가림: 헬멧, 가면, 창살, 어둡고 흐린 작은 얼굴 (학습 데이터에 없음). 앱 촬영 장면에서는 드문 종류
     - yaw 규칙과 합친 보이는 얼굴 오판: dev 5.0% (yaw 만 3.4%). 촬영을 잘못 막는 가장 큰 원인은 yaw 규칙 -> `YAW_TH` 재검토 (15번)
     - 측정 스크립트: `evaluation/wider/occlusion_cnn_eval.py`, `occlusion_yunet.py`
   - 갱신 (2026-09-27): occlusion CNN v0.0.5 (학습 20260926-1728), `OCC_TH` 0.86
     - v0.0.4 는 본인 손으로 입을 가린 얼굴을 잘 못 잡음 (앱 촬영에서 확인한 증상) -> 학습 데이터에 얼굴과 색 차이가 없는 가리개가 없었음
     - 재학습 데이터: 피부색을 맞춘 입 가림, 두 눈 가림 (앞사람 머리, 손, 물건. 어두운 가리개 포함) 추가. 합성 선글라스는 노트북에서 학습 중 무작위로
     - 기준값: 노트북 기준값 0.569 는 합성 위주 val 에서 정한 값이라 WIDER 에서 보이는 얼굴 오판이 dev 4.7%, holdout 3.4%
       -> WIDER dev 에서 보이는 얼굴 오판 2% 가 되는 값으로 정함 (EYE_TH 와 같은 방식. holdout 은 확인용)

| 모델, 기준 | AUC (dev / holdout) | 가림 recall (dev / holdout) | 보이는 얼굴 오판 (dev / holdout) | 선글라스 얼굴 오판 (dev / holdout) |
|---|---|---|---|---|
| v0.0.4 (1611), 0.59 | 0.905 / 0.912 | 29% / 50% | 1.7% / 1.4% | 1/38 / 0/38 |
| v0.0.5 (1728), 0.90 | 0.929 / 0.969 | 31% / 54% | 1.5% / 1.0% | 1/38 / 0/38 |
| v0.0.5 (1728), 0.86 | 0.929 / 0.969 | 40% / 63% | 2.0% / 1.7% | 1/38 / 0/38 |
| v0.0.5 (1728), 0.76 | 0.929 / 0.969 | 51% / 72% | 3.1% / 2.3% | 1/38 / 0/38 |

     - 0.86 을 고른 이유: 보이는 얼굴 오판을 v0.0.4 와 비슷하게 두고 가림을 더 잡음
     - 한계: 두 눈 가림은 잘 못 잡고, 두 눈 가림과 선글라스를 잘 구분하지 못함 (앱 촬영에서 확인한 증상. 두 눈 가림 실제 학습 데이터는 COFW 3장뿐, 나머지는 합성)
       - 두 눈을 가린 얼굴은 eye 모델이 눈 뜸으로 보기 어려워 CHECKING 에서 걸러질 것으로 봄 (추정, 미확인). 빠지는 것은 PREVIEW 안내
   - 이전 기록: 보임 판정이 yaw 만일 때 가려진 얼굴은 거의 못 잡았음 (아래는 예전 가림 기준의 측정)
   - WIDER dev, YuNet 이 검출한 가려진 얼굴 88개 (손 7, 다른 사람 35, 기타 물건 44, 머리카락 2)
     - yaw >= 0.5 로 잡음: 7개 (8%)
     - FaceLandmarker 미검출: 48개 (55%). yunet_landmark 는 CHECKING 에서 이 얼굴의 웃음, 눈이 None -> 조건 미충족으로 걸러짐 (안내는 없음)
   - 후보
     - PREVIEW 에서도 FaceLandmarker 실행, 미검출이면 보이지 않음 (yaw 와 함께: 보이지 않는 얼굴 84% 잡음, 보이는 얼굴 6% 오판. 얼굴당 약 13ms 추가. CNN 모델에는 적용 불가)
     - 가림 전용 분류기 (모델과 무관하게 사용 가능. 학습 데이터 필요)

13. yunet_cnn 눈 뜸 기준 `EYE_TH`: 0.8 (측정 기록, 2026-09-26)
   - 학습 노트북 기준값 0.459 는 감은 눈을 뜸으로 판정하는 경우가 많음
     - WIDER 감은 눈 recall: dev 65%, holdout 47% (blendshape 82%, 79%)
     - 앱 촬영에서 투명 안경에 반사가 비친 감은 눈을 뜸(0.804)으로 판정해 결과 사진에 포함됨
   - 감은 눈을 뜸으로 판정하면 조건에 맞지 않는 사진이 결과로 나감 -> 뜬 눈을 감음으로 보는 오판보다 우선해서 줄임
   - 얼굴 단위 (가로 640. 기준은 dev 로 고르고 holdout 은 확인용)

| EYE_TH | dev 감은 눈 recall | dev 뜬 눈을 감음으로 | holdout 감은 눈 recall | holdout 뜬 눈을 감음으로 |
|---|---|---|---|---|
| 0.459 | 26/40 (65%) | 43/1209 (4%) | 25/53 (47%) | 64/1262 (5%) |
| 0.7 | 34/40 (85%) | 119/1209 (10%) | 39/53 (74%) | 158/1262 (13%) |
| 0.8 | 37/40 (92%) | 197/1209 (16%) | 44/53 (83%) | 226/1262 (18%) |

   - 사진 조건 충족 (smile 0.757, YuNet 0.7, yaw 규칙): dev precision 90% -> 94%, recall 41% -> 32% / holdout 94% -> 94%, 35% -> 25% (smile 기준은 이후 0.5 로 변경. 14번)
   - 선글라스류 뜸 판정이 줄어듦: dev 36/36 -> 25/36, holdout 38/38 -> 27/38
   - 근본 해결은 eye 재학습 (투명 안경 증강. 안경테, 반사를 "뜸" 의 단서로 쓰지 않도록)
   - eye v0.0.3 (재학습: 투명 안경 증강, 4단계 눈 뜸 정도 라벨, 기하 지표 교사) 에서 기준 0.8 유지 (2026-09-26)
     - 목표: 감은 눈 recall (= 뜬 눈 precision) 을 올림
     - 얼굴 단위 뜬 눈 precision 은 감은 눈이 드물어 (3% 정도) 기준을 바꿔도 99% 대에서 거의 움직이지 않음 -> 기준은 감은 눈 recall 로 고름
     - 기준을 0.8 보다 올리면 감은 눈 recall 은 1명씩 늘고, 뜬 눈을 감음으로 보는 경우는 5%p 이상씩 늘어남
     - 학습 노트북 기준값 0.817 (val 감은 눈 recall >= 0.95) 과도 거의 같음

| v0.0.3 EYE_TH | dev 감은 눈 recall | dev 뜬 눈을 감음으로 | holdout 감은 눈 recall | holdout 뜬 눈을 감음으로 |
|---|---|---|---|---|
| 0.7 | 36/40 (90%) | 173/1209 (14%) | 48/53 (91%) | 205/1262 (16%) |
| 0.75 | 37/40 (92%) | 208/1209 (17%) | 48/53 (91%) | 235/1262 (19%) |
| 0.8 | 38/40 (95%) | 239/1209 (20%) | 49/53 (92%) | 276/1262 (22%) |
| 0.85 | 39/40 (98%) | 299/1209 (25%) | 49/53 (92%) | 332/1262 (26%) |
| 0.9 | 40/40 (100%) | 388/1209 (32%) | 50/53 (94%) | 419/1262 (33%) |

   - v0.0.2 (같은 0.8) 대비: 감은 눈 recall dev 92% -> 95%, holdout 83% -> 92% / 뜬 눈을 감음으로 dev 16% -> 20%, holdout 18% -> 22%
   - 선글라스류 뜸 판정: dev 25/36 -> 23/36, holdout 27/38 -> 28/38 (거의 같음)
   - v0.0.6 (eye 에 COFW 실제 선글라스 추가) 에서 0.84 로 바꿈 (2026-09-27). 같은 방식 (dev 감은 눈 recall 95%) 으로 고른 값 (17번)

14. yunet_cnn 웃음 기준 `SMILE_TH`: 0.5 (측정 기록, 2026-09-26)
   - 판단 (thinkcat): 웃었는데 안 웃음으로 판정하는 경우를 줄이는 쪽을 우선. 안 웃은 얼굴을 웃음으로 판정하는 경우는 어느 정도 허용
     - 이유: 본인은 웃었다고 생각하지만 남이 보면 안 웃은 것처럼 보이는 아주 약한 미소가 있음
   - 눈과의 차이: 감은 눈은 드물고 잠깐이라 뜸으로 잘못 보면 결과 사진에 그대로 나감 (13번). 웃음은 약한 미소를 놓치면 촬영 시작(PREPARE)과 결과 선택에서 계속 떨어짐
   - 얼굴 단위 (가로 640. 기준은 dev 로 고르고 holdout 은 확인용)

| SMILE_TH | dev 웃음 recall | dev 안 웃음을 웃음으로 | holdout 웃음 recall | holdout 안 웃음을 웃음으로 |
|---|---|---|---|---|
| 0.757 (학습 노트북 기준값) | 626/965 (65%) | 10/353 (3%) | 636/989 (64%) | 19/390 (5%) |
| 0.6 | 727/965 (75%) | 24/353 (7%) | 727/989 (74%) | 40/390 (10%) |
| 0.5 | 773/965 (80%) | 43/353 (12%) | 771/989 (78%) | 56/390 (14%) |
| 0.4 | 811/965 (84%) | 60/353 (17%) | 821/989 (83%) | 85/390 (22%) |

   - 사진 조건 충족 (EYE_TH 0.8, YuNet 0.7, yaw 규칙): dev precision 94% -> 89%, recall 32% -> 43% / holdout 94% -> 85%, 25% -> 31%
     - precision 하락분은 라벨상 안 웃은 사람이 있는데 통과한 사진. 약한 미소를 허용한 결과로 일부는 문제가 아님
   - 0.4 는 웃음 recall 이 4%p 더 오르지만 안 웃음을 웃음으로 보는 경우가 17~22% 로 늘어 제외
   - v0.0.6 (smile 얼굴 아래쪽 입력) 에서 0.31 로 바꿈 (2026-09-27). 안 웃음을 웃음으로 보는 비율을 0.5 때와 같게 맞춘 값 (17번)

15. yunet_cnn 고개 돌림 기준 `YAW_TH`: 0.6 (측정 기록, 2026-09-27)
   - 보임 기준 변경(9번) 뒤 WIDER 라벨 재검수, 가림 CNN 추가(12번) 후 다시 봄. 0.5 는 예전 기준(두 눈과 입이 모두 보여야 보임) 때 정한 값
   - 0.5 에서 보이는 얼굴의 3.4% (dev), 4.2% (holdout) 가 보이지 않음으로 판정됨 -> PREVIEW 에서 촬영이 막힘. 가림 CNN 오판(2%) 보다 큼
     - 걸린 얼굴은 작은 얼굴이 아님 (높이 중앙값 약 70px, 보이는 얼굴 전체 59px). 고개를 조금 돌렸지만 눈과 입이 보이는 얼굴
   - WIDER, YuNet 0.7 이상 검출 얼굴 (기준은 dev 로 고르고 holdout 은 확인용)

| YAW_TH | 보이는 얼굴 오판 (dev / holdout) | 옆모습 잡음 (dev / holdout) |
|---|---|---|
| 0.5 | 3.4% / 4.2% | 82% / 75% |
| 0.6 | 2.0% / 2.6% | 73% / 69% |
| 0.7 | 1.4% / 1.9% | 65% / 60% |

   - 결정 (2026-09-27 thinkcat): 0.6. 보이는 얼굴 오판을 가림 CNN 과 비슷한 2% 수준으로
   - 가림 CNN (`OCC_TH` 0.86) 과 합친 보임 판정: 보이는 얼굴 오판 dev 5.3% -> 4.0%, holdout 5.8% -> 4.2% / 가림 dev 44% -> 42%, holdout 63% / 옆모습 dev 86% -> 78%, holdout 78% -> 74%
   - 놓친 옆모습은 CHECKING 에서 웃음, 눈으로 판정됨 (대개 조건 미충족으로 걸러질 것으로 봄)
   - yunet_landmark, landmark 의 `YAW_TH` 는 0.5 그대로 (따로 측정하지 않음)

16. 추론 시간 줄이기 검토 (기록, 2026-09-27. 진행하지 않음 thinkcat)
   - 단계별 시간 (yunet_cnn, WIDER dev 사진, 가로 640, Mac. `evaluation/wider/compare_models_stage.py`)
     - PREVIEW 사진당 평균 약 90ms 중 손 검출(HandLandmarker) 63~70ms (약 70%), 가림 CNN 약 11ms
     - PREPARE 약 46ms (smile CNN 약 20ms), CHECKING 약 59ms (가림, smile, eye CNN 합 약 40ms)
   - 후보 1. 모델 결합: smile, eye, occlusion CNN 을 백본 하나 + 출력 3개로
     - 이득: CHECKING 에서 얼굴당 CNN 약 13ms -> 약 5ms. PREVIEW (가림만), PREPARE (smile 만) 는 CNN 1개라 이득 없음
     - 문제: 입력이 모두 다름. eye 얼굴 위쪽 60% 160px, smile 얼굴 아래쪽 45% 128px (v0.0.6), occlusion 얼굴 전체 128px. 얼굴 전체로 넣은 eye 는 WIDER dev AUC 0.952 -> 0.915 (학습 20260926-1307 실험)
     - 문제: 학습 데이터가 과제마다 다름 (GENKI, eye, occlusion) -> 부분 라벨 다중 과제 학습 필요. 모델 하나만 고쳐 재학습하기 어려워짐
   - 후보 2. PREVIEW 손 검출 줄이기 (학습 불필요, 설정으로 가능)
     - 손 검출을 판정 2~3회에 한 번만 하고 사이에는 직전 결과 사용 (손 들기는 여러 프레임에 걸친 동작)
     - 손 검출 입력 해상도 낮추기 (정확도 확인 필요)
   - 속도 측정 환경: colima (Linux VM, 4코어, 8GB) 는 같은 코드도 측정마다 2~3배 흔들려 비교에 쓰기 어려움. 속도 판단은 라즈베리파이 실기기에서 반복 측정

17. 렌즈 반사 선글라스 오판 (알려진 한계, 2026-09-27)
   - 증상 (앱 촬영, 선글라스 착용, 웃지 않음. 증상 확인용이고 기준값 근거 아님)
     - 렌즈에 밝은 반사(창문, 조명)가 비치면 smile CNN 이 웃음으로, eye CNN 이 감음으로 판정
     - 얼굴이 클수록 (약 120px 이상) 반사가 또렷해져 심함. 반사 없는 어두운 렌즈는 대체로 맞게 판정
     - 사진 93장 중 얼굴 200px 이상 27장: smile 0.5 이상 22장, eye 0.8 미만 27장 (v0.0.5)
   - WIDER 선글라스 얼굴 (dev 38, holdout 38): v0.0.5 기준으로 eye 감음 판정 24/76, 안 웃은 얼굴을 웃음으로 dev 5/16, holdout 5/18
   - 시도: 합성 선글라스에 렌즈 종류 추가 후 재학습 (dark 0.4, reflect 0.35, mirror 0.1, light 0.15. light 는 눈이 비쳐 라벨 유지)
     - 비교: WIDER 가로 640, 보이는 얼굴. 기준은 dev 에서 감은 눈 recall 95% (eye), 안 웃음을 웃음으로 v0.0.5 와 같은 비율 (smile)
     - 스크립트: `evaluation/wider/eye_retrain_compare.py`, `smile_retrain_compare.py`, `app_captures_retrain.py`

| 모델 | AUC (dev / holdout) | 선글라스 판정 (dev / holdout) | 앱 촬영 200px 이상 27장 |
|---|---|---|---|
| eye v0.0.5 (기준 0.8) | 0.947 / 0.933 | 뜸 66% / 71% | 뜸 0장 |
| eye 20260926-1907 (30 epoch 조기 종료) | 0.932 / 0.906 | 기준 0.932 에서 뜸 39% / 47% | 뜸 9장 (기준 0.8) |
| eye 20260926-1936 (56 epoch, 기준 0.85) | 0.945 / 0.924 | 뜸 71% / 79% | 뜸 6장 (기준 0.8) |
| smile v0.0.5 (기준 0.5) | 0.904 / 0.889 | 안 웃음을 웃음으로 31% / 28% | 웃음 22장 |
| smile 20260926-1921 (기준 0.638) | 0.898 / 0.892 | 안 웃음을 웃음으로 25% / 6% | 웃음 24장 (기준 0.5) |
| smile 20260926-1958 (얼굴 아래쪽 0.45, 기준 0.31) | 0.900 / 0.902 | 안 웃음을 웃음으로 13% / 0% | 웃음 8장 |
| eye 20260926-2040 (COFW 실제 선글라스 85장 추가, 기준 0.84) | 0.947 / 0.931 | 뜸 87% / 95% | 뜸 20장 / 32장 (기준 0.8, 촬영 추가 후) |

   - 1907 은 val AUC 가 잠깐 멈춘 사이 조기 종료 (학습률 코사인 감소 중간). 1936 은 같은 설정으로 끝까지 학습
   - 결과: WIDER 는 v0.0.5 와 거의 같고, 선글라스 판정 개선은 2~3명 차이 (감은 눈 43, 59명, 안 웃은 선글라스 16, 18명이라 흔들림 큼). 앱 촬영의 큰 얼굴 반사는 대부분 그대로
   - 결정 (2026-09-27 thinkcat): 1907, 1936, 1921 은 쓰지 않음. 재학습 효과가 크지 않음
   - smile 입력을 얼굴 아래쪽만으로 (렌즈를 입력에서 뺌): 학습 20260926-1958
     - 학습 노트북 `SMILE_BOTTOM` 0.45 (눈 중심 높이 약 0.37, 99% 가 0.42 이내. 합성 렌즈 아래 끝 약 0.52 -> 0.45 면 렌즈 제외)
     - 걱정했던 점: 눈가 단서가 빠짐 -> WIDER AUC 는 거의 같음 (입꼬리 blendshape 만 쓰는 yunet_landmark 도 0.936 / 0.933)
     - 기준 0.31: dev 에서 안 웃음을 웃음으로 보는 비율을 v0.0.5 (0.5) 와 같게 (11%). 웃음 recall dev 77% (v0.0.5 79%), holdout 77% (77%), holdout 안 웃음을 웃음으로 10% (15%)
     - 앱 촬영 93장 (모두 안 웃음) 중 웃음 판정 37장 -> 11장. 남은 것은 큰 얼굴에서 0.31~0.46
   - 결정 (2026-09-27 thinkcat): smile 은 1958 로 바꿈 (v0.0.6)
   - eye 에 실제 선글라스 추가: 학습 20260926-2040
     - 합성만으로는 실제 반사를 따라가지 못함 -> COFW 선글라스 얼굴 91장 검수 (thinkcat) 후 85장을 opened 로 추가 (실제 선글라스 13장 -> 98장)
     - 기준 0.84: dev 감은 눈 recall 95% (v0.0.5 의 0.8 과 같은 방식). 뜬 눈을 감음으로 dev 20%, holdout 20% (v0.0.5 21%, 23%). holdout 감은 눈 recall 55/59 (v0.0.5 56/59)
     - 앱 촬영 (114장, 기준 0.8) 뜸 판정 36장 -> 72장. 200px 이상 32장 중 0장 -> 20장. 120~200px 는 여전히 약함
   - 결정 (2026-09-27 thinkcat): eye 도 2040 으로 바꿔 v0.0.6 에 합침 (새 버전을 만들지 않음)
   - v0.0.6 사진 조건 충족 (앱 설정): dev precision 94%, recall 44% (v0.0.5 94%, 41%) / holdout 84%, 37% (v0.0.5 91%, 33%)
     - holdout 에서 새로 잘못 통과한 5장은 이전 eye 가 뜬 눈을 감음으로 잘못 봐서 막혀 있던 사진. 드러난 원인은 YuNet 미검출 2, smile 오판 1, 두 eye 모두 놓친 감은 눈 1, 옆모습 1 (`docs/model_versions.md` v0.0.6)

18. 웃느라 가늘어진 눈의 감음 오판 (측정 기록, 2026-09-27)
   - 증상: 웃어서 눈이 가늘어진 얼굴을 eye CNN 이 감음으로 판정 (표지 사진 두 명 v0.0.6 0.22, 0.48. 베이스라인 blendshape 는 한 명을 뜸으로 봄)
   - 원인 (추정): 학습 목표에서 웃는 눈 (단계 2) 이 0.7 인데 앱 기준 `EYE_TH` 가 0.84. 웃는 눈을 기준 아래로 내도록 학습한 셈. 웃는 눈 학습 데이터도 126장뿐
   - 시도 (학습, WIDER 가로 640 보이는 얼굴 AUC dev / holdout)

| 학습 | 바꾼 것 | AUC | 비고 |
|---|---|---|---|
| v0.0.6 eye (2040) | - | 0.947 / 0.931 | |
| 20260926-2324 | 웃는 눈 목표 1, 감기는 눈 0 | 0.946 / 0.931 | 변화 없음 |
| 20260927-0005 | GENKI 웃는 눈 293장 추가, 베이스라인 교사 (목표는 0.7 그대로) | 0.953 / 0.931 | 앱 촬영 반사 선글라스 약해짐 |
| 20260927-0023 | 위 둘 모두 | 0.956 / 0.941 | 채택 (v0.0.7) |

   - GENKI 추가: GENKI 웃음 사진 중 기하 눈 열림이 작은 300장을 검수 (thinkcat) 해 293장 (`training/data/eye/genki_smile/`)
   - 베이스라인 교사: 교사 점수를 기하 지표 교사에서 1 - blendshape 눈 감음 최대로 (thinkcat)
     - 참고: FaceLandmarker 기하 눈 열림 (위아래 눈꺼풀 거리 / 눈 가로) 의 WIDER AUC 는 blendshape 보다 낮음 (dev 0.918 / holdout 0.869, 두 눈 중 작은 쪽. blendshape 0.930 / 0.908). 작은 얼굴에서 특히 낮음 (`evaluation/wider/eye_geometry_auc.py`)
   - 결정 (2026-09-27 thinkcat): 0023 을 v0.0.7 로. `EYE_TH` 0.84 그대로
     - dev 에서 고른 0.814 는 뜬 눈 오판 dev 15.7% 로 가장 적지만 holdout 감은 눈 53/59. 0.84 는 54/59 (v0.0.6 55/59), 뜬 눈 오판 dev 17.5% (v0.0.6 20.2%), holdout 19.4% (20.2%)
   - 남은 한계: 눈이 많이 가늘어진 웃음은 여전히 감음 (표지 두 명 0.50, 0.61). 조건 충족 사진이 없을 때는 가까운 사진을 보여 줌 (7절)
   - 추후 과제 (2026-09-27 thinkcat): 웃는 눈 학습 데이터 추가, eye 학습 데이터 라벨 전체 재검수
19. 프레임 판정 스무딩 (구현, 2026-09-27)
   - 판정 약 5 FPS (1회 약 0.2초). 스무딩 전에는 손 들기, 모두 웃음이 판정 1회로 상태를 바꿈
   - 결정 (2026-09-27 thinkcat): 손 들기, 모두 웃음 모두 프레임 단위 연속 2회 (`HAND_ON_COUNT`, `SMILE_ON_COUNT`)
     - 손 들기: 잘못 시작하면 12초 촬영이 진행되어 비용이 큼. 약 0.4초 지연
     - 모두 웃음: 오판으로 넘어가면 웃지 않은 사람이 있는 사진만 찍힘
   - 적용하지 않음: CHECKING 눈 판정. 깜빡임 (0.1~0.4초, 촬영 20 FPS 에서 2~8장) 이 잡아야 할 대상이라 스무딩하면 지워짐
   - 한계: 프레임 단위 연속은 인원이 많을수록 모두 웃음 진입이 어려움 (모든 얼굴이 연속으로 통과해야 함)
     - 인원이 많을 때 진입이 늦으면 얼굴별 스무딩 (IoU 추적 + 최근 3회 중앙값) 검토
   - 흔들림 비율 (같은 표정 유지 중 판정이 바뀌는 비율) 은 WIDER (정지 사진) 로 잴 수 없음. 모델 평가 화면 연속 촬영으로 확인 필요

## 11. 테스트 (pytest)

- 카메라와 화면 없이 확인할 수 있는 로직을 pytest 로 테스트
- 속도와 FPS 는 9절에 따라 라즈베리파이에서 측정

| 대상 | 확인할 내용 |
|---|---|
| `Prediction.guidance`, `ready_to_start` | 작은 얼굴, 보이지 않는 얼굴 안내 이유, 둘 다 해당 시 보이지 않음 우선, 안내 대상이 있으면 시작 불가, 필요한 결과 미요청 시 ValueError |
| 안내 표시 안정화 | 안내 대상이 `GUIDE_ON_COUNT` 회 연속일 때만 표시 시작, `GUIDE_OFF_COUNT` 회 연속 없을 때만 끝, 표시 중에는 손을 들어도 시작 불가 |
| 전환 안정화 | 손 들기 `HAND_ON_COUNT` 회, 모두 웃음 `SMILE_ON_COUNT` 회 연속일 때만 전환, 한 번 끊기면 다시 셈 |
| `Prediction.meets_condition` | 얼굴 0명이면 False, 한 명이라도 웃지 않거나 눈을 감았으면 False, smile 과 eye 를 함께 요청하지 않았으면 ValueError |
| `Prediction.eye_score` | 얼굴 중 가장 낮은 점수, 얼굴이 없거나 점수 없는 얼굴이 있으면 None |
| 모델 (`models/`) | `create_model` 이름별 생성, 요청하지 않은 결과는 None, 보이지 않는 얼굴은 웃음과 눈이 None, 키보드 손 들기 유지 시간, yunet_cnn 의 기준값, 요청받은 CNN 만 실행, 가림 점수로 보임 판정, 얼굴 자르기 (eye 위쪽, smile 아래쪽) |
| landmark 룰 (`rules.py`) | 웃음, 눈 뜸 임계값 판정, 손 들기 (손가락 방향), 고개 돌림 yaw |
| 화면 (`ui`) | 상태별 렌더 크기, 촬영 중 버튼 없음, 버튼 클릭 판정, 얼굴 bbox, 손 bbox, dev 문구는 dev 에서만 표시, dev 얼굴 띠 (판정한 항목만, blink 는 1 - 점수, 가림) |
| 측정 기록 (`metrics`) | 파일 이름, JSON 한 줄 기록, start 전 무시, predict 이벤트 필드 |
| 로그 분석 (`analysis`) | 요청 조합별 추론 시간, 판정, 화면 FPS 보고서 값, 컬럼이 없으면 보고서 생략 |
| 결과 사진 선택 (`select`) | m=0, m<=n, m>n 각 경우의 선택 결과. m>n 일 때 구간마다 점수 최고 1장, 사진 점수는 눈을 가장 작게 뜬 사람 기준, 점수가 같으면 먼저 찍힌 사진, 중복 없음, 찍힌 순서 |
| 가까운 사진 선택 (`select_closest`, `closeness`, `face_margin`) | 기준 대비 차이 (blink 는 반대 방향), 판정 못 한 얼굴 -1, 점수 없는 모델, 충족 얼굴 수 우선, 구간마다 가장 가까운 사진, 얼굴 없는 사진 제외 |
| CHECKING 처리 (`check`) | 조건 충족 사진이 없으면 가까운 사진으로 대신하고 표시, 있으면 충족 사진만 |
| 결과 화면 | 가까운 사진일 때 제목 문구가 바뀜 |
| 상태 전환 | 1절 표의 전환 전부. 가짜 모델과 가짜 시계로 손 들기, 모두 웃음, 준비 시간 초과를 흉내 |
| 초기화면 버튼 | PREPARE, CAPTURE, CHECKING 상태에서는 동작하지 않음 |
| 카메라 (`camera`) | 프레임 보관, 시작할 때 카메라 없음, 끊기면 프레임 비우고 간격마다 다시 엶, 다시 연결, 촬영 누적 (가짜 캡처, 가짜 시계) |
| 카메라 끊김 | 촬영 중이면 PREVIEW 로 (취소 문구 없음), 그 외 상태는 그대로, 화면은 안내 문구 (버튼 그대로) |
| 사진 정리 (`storage`) | RESULT 를 벗어나면 결과 사진만 남고 나머지 삭제 (`tmp_path` 사용) |
| 판정 결과 받기 (`app.accepts`) | 현재 요청으로 만든 판정만 받음 (이전 상태, 모델 평가에서 바꾸기 전 항목의 판정은 버림) |
| 모델 평가 (`model_eval`) | 켠 항목만 요청, 모델은 한 번만 생성, 모르는 이름은 오류, 찍기는 jpg 와 json (scores 포함) 저장, 연속 촬영 간격과 폴더. dev 모드에서만 초기화면 버튼, 평가 화면 버튼 목록, 점수 표시 |

- 상태 전환은 시간에 의존하므로 시계 주입 (`now()` 함수를 인자로 받음). 테스트에서 실제로 준비 시간을 기다리지 않기 위함
- 테스트 파일은 `app/tests/` 아래에 모듈별로 둠
- 테스트는 `# given`, `# when`, `# then` 으로 단계 구분. docstring 첫 줄은 `pytest -v` 표시 이름 (`tests/conftest.py`)
