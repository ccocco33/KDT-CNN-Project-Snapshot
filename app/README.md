# src

앱 UI, 앱 상태 관리, 카메라 및 마이크 입력, 모델 추론 호출.

## 준비: 모델 파일

- 앱 실행에 필요한 모델은 저장소에 있음 (`app/models/`). 따로 받을 것 없음
  - 공통: `face_detection_yunet_2023mar.onnx` (YuNet, OpenCV Zoo), `face_landmarker.task`, `hand_landmarker.task` (MediaPipe)
  - yunet_cnn 의 CNN 현재 버전 v0.0.7: `models/v0.0.7/{smile,eye,occlusion}_{fp32,dynamic,int8}.tflite`, `*_inference_config.json` (앱은 fp32 사용)
- 이전 버전 (`models/v0.0.1` ~ `v0.0.6`) 은 저장소에 두지 않음. 필요하면 팀 드라이브에서 받아 `models/{버전}/` 에 둠 (`docs/model_versions.md`)
- 한글 폰트도 저장소에 있음 (`fonts/NanumGothic-Regular.ttf`, SIL OFL `fonts/OFL.txt`). 한글 폰트가 없는 라즈베리파이에서 글자가 깨지지 않도록 앱이 먼저 사용
- 공통 모델 원본 주소 (다시 받을 때)

```bash
cd app
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## 로컬 실행

- Python 3.11

```bash
cd app
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m photo_app                        # 앱 실행 (prod 모드. d 키로 dev 전환)
PHOTO_APP_MODE=dev .venv/bin/python -m photo_app     # dev 모드로 시작
.venv/bin/pytest                                     # 테스트 (-v: docstring 표시 이름)
.venv/bin/pytest --cov=photo_app --cov=analysis --cov-report=term-missing   # 테스트 + 커버리지
.venv/bin/python -m photo_app.wireframe              # 와이어프레임 PNG, 모은 grid.png -> docs/wireframe/
.venv/bin/python -m analysis                         # 로그 분석 (logs/*.jsonl)
```

- 사용할 모델: `photo_app/config.py` 의 `MODEL` (`landmark`, `yunet_landmark`, `yunet_cnn`, `fake`)
- 실행 모드: 기본 prod. 앱에서 `d` 키로 dev, prod 전환 (초기화면 왼쪽 위에 안내)
  - dev: 얼굴 박스와 판정 점수 (웃음, 눈 뜸 O/X), 손 bbox, 모델 이름 표시, 초기화면 모델 평가 버튼, 키보드 흉내 사용
  - prod: 모두 끔
- 키보드 (`d`, `q`, `ESC` 외에는 dev 모드에서만)

| 키 | 동작 |
|---|---|
| `d` | dev, prod 전환 (모든 모드) |
| `h` | 손 들기 흉내 (1초) |
| `s` | 웃음 토글 (fake 모델만) |
| `e` | 눈 뜸 토글 (fake 모델만) |
| `o` | 보이지 않음 토글 (fake 모델만) |
| `q`, `ESC` | 종료 (모든 모드) |

- 맥에서 처음 실행하면 터미널의 카메라 권한 요청이 뜸

## Docker 실행

- 라즈베리파이(Debian bookworm, Python 3.11, arm64)와 같은 환경
- 자원 제한: `compose.yaml` 의 `PI_CPUS` (기본 4), `PI_MEM` (기본 8g)
  - 제한은 코어 수와 메모리만. 코어 1개의 속도는 호스트 CPU 그대로
  - Docker Desktop 은 VM 자원보다 큰 값을 줄 수 없음. 기본값(4코어, 8GB)으로 돌리려면 Docker Desktop 설정 > Resources 에서 VM 자원을 늘리거나, 환경변수로 낮춤
  - 예: VM 이 2코어, 2GB 면 `PI_CPUS=2 PI_MEM=2g`
  - 매번 붙이지 않으려면 `app/.env` 에 기기 값을 적어 둠 (compose 가 자동으로 읽음, 커밋하지 않음)

```
PI_CPUS=2
PI_MEM=2g
```

### 테스트, 로그 분석 (맥, Linux)

```bash
cd app
docker compose build test
docker compose run --rm test                        # 테스트
docker compose run --rm test python -m analysis     # 로그 분석 (app/logs 마운트)
```

### 앱 (Linux 호스트 전용)

- 맥의 Docker 는 카메라를 컨테이너에 넘길 수 없어 앱 실행 불가. 맥에서는 로컬 실행 사용
- 필요: 카메라 `/dev/video0`, X 서버

```bash
cd app
xhost +local:docker                      # 컨테이너의 X 서버 접근 허용
docker compose run --rm app              # prod 모드 (d 키로 dev 전환)
PHOTO_APP_MODE=dev docker compose run --rm app
```

## 버전 고정

- `mediapipe==0.10.18`
  - 라즈베리파이(Linux aarch64)와 맥 모두 배포판이 있는 버전
  - 1.0.1 은 맥에서 FaceLandmarker 생성 시 중단, 0.10.35 는 Linux aarch64 배포판 없음
  - numpy 2 미만을 요구하므로 opencv-contrib-python 은 4.x 로 설치됨
