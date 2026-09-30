"""설정값"""
import os
from pathlib import Path

# 시작 실행 모드. 환경변수 PHOTO_APP_MODE 로 변경. 앱에서 d 키로 전환 (초기화면에 안내)
# - dev: 얼굴 bbox 와 판정 점수, 손 bbox, 모델 이름 표시, 모델 평가 버튼, 키보드 흉내(h, s, e, o) 사용
# - prod: 모두 끔
MODE = os.environ.get("PHOTO_APP_MODE", "prod")
if MODE not in ("dev", "prod"):
    raise ValueError(f"PHOTO_APP_MODE 는 dev 또는 prod: {MODE}")
DEV = MODE == "dev"   # 시작 값. 실행 중 모드는 app.py 가 관리

APP_DIR = Path(__file__).resolve().parents[1]   # app/

APP_NAME = "모두의 동네 사진사 (가칭)"
WINDOW_NAME = "modongsa"

SCREEN_W = 640           # 화면 너비
SCREEN_H = 480           # 화면 높이
SCREEN_FPS = 30          # 화면 갱신 상한

CAMERA_INDEX = 0         # cv2.VideoCapture 장치 번호
CAMERA_W = 640           # 카메라 요청 해상도 (장치가 지원하지 않으면 다른 값으로 열림)
CAMERA_H = 480
CAMERA_FPS = 30
CAMERA_FOURCC = "MJPG"   # 카메라 요청 픽셀 포맷. 기본 YUYV 는 USB 대역폭 때문에 640x480 에서 약 13 FPS (라즈베리파이 측정), MJPG 는 30 FPS
CAMERA_LOST_SEC = 2.0    # 새 프레임이 이 시간(초) 이상 오지 않으면 카메라 끊김으로 봄
CAMERA_RETRY_SEC = 1.0   # 끊긴 동안 카메라를 다시 여는 간격(초)

PREPARE_SEC = 10         # 준비 제한 시간
CAPTURE_SEC = 2          # 촬영 제한 시간
CANCEL_MSG_SEC = 3       # 취소 문구 표시 시간
MAX_RESULT = 5           # 결과로 보여주고 남길 최대 사진 수
MIN_FACE_H = 50          # 최소 얼굴 높이 (카메라 프레임 px). 미만이면 PREVIEW 에서 안내 (제품 요구사항)
GUIDE_ON_COUNT = 3       # 안내 대상이 이 횟수만큼 연속 판정되면 안내 표시 시작 (판정 5 FPS 에서 약 0.6초)
EVAL_BURST_SEC = 0.5     # 모델 평가 화면(dev) 연속 촬영 간격 (초)
DEV_CAPTURE_PREDICT_SEC = 0.2   # dev 모드 CAPTURE 중 표시용 판정 간격 (초, 약 5 FPS). prod 는 CAPTURE 중 판정 안 함
GUIDE_OFF_COUNT = 2      # 안내 대상 없음이 이 횟수만큼 연속 판정되면 안내 표시 끝
HAND_ON_COUNT = 2        # 손 들기가 이 횟수만큼 연속 판정되면 PREPARE 로 (판정 5 FPS 에서 약 0.4초)
SMILE_ON_COUNT = 2       # 모두 웃음이 이 횟수만큼 연속 판정되면 CAPTURE 로
WARN_PEOPLE = None       # 경고를 띄우는 인원 수 (제한 아님). 측정 후 결정

# 눈 감음 허용 (selection.EyeHold): 같은 얼굴이 촬영 시각 기준 EYE_HOLD_SEC 이상 연속 눈 감음이면 그 프레임부터 눈 뜸으로 봄
EYE_HOLD = False             # CHECKING (결과 고르기) 에 적용할지. 모델 평가 화면(dev)은 eye_hold 버튼으로 따로 켬
EYE_HOLD_SEC = 0.30          # 연속 눈 감음이 이 시간(초) 이상이면 허용 (그 전 프레임은 소급하지 않음)
EYE_HOLD_MIN_SAMPLES = 3     # 연속 눈 감음 최소 관측 수. 시간 조건과 둘 다 만족해야 허용
EYE_HOLD_MAX_GAP_SEC = 0.15  # 이웃한 프레임 촬영 시각 차이가 이보다 크면 누적 초기화
EYE_HOLD_IOU = 0.30          # 이전 프레임 얼굴과 같은 사람으로 이을 최소 박스 IoU
EYE_HOLD_AMBIGUITY = 0.10    # 가장 좋은 두 짝의 IoU 차이가 이보다 작으면 잇지 않음 (애매함)
SAVE_DIR = APP_DIR / "photos"   # 사진 저장 위치
LOG_DIR = APP_DIR / "logs"      # 시간 기록 로그 위치

# 모델
# - 모델별 설정(임계값, 모델 파일 경로 등)은 photo_app/models/<이름>/config.py
MODEL = "yunet_cnn"              # 사용할 모델: "fake" (키보드 흉내), "landmark" (FaceLandmarker + 룰), "yunet_landmark" (YuNet + 잘라낸 얼굴 FaceLandmarker + 룰), "yunet_cnn" (YuNet + CNN)
MODEL_DIR = APP_DIR / "models"  # 모델 파일(가중치) 위치. 커밋하지 않음
NUM_FACES = 10                  # 최대 검출 인원 (제품 요구사항)

# 한글 폰트 후보. 앞에서부터 존재하는 파일 사용
FONT_PATHS = (
    APP_DIR / "fonts" / "NanumGothic-Regular.ttf",          # 저장소에 포함 (OFL). 라즈베리파이 등 한글 폰트가 없는 환경용
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",      # Debian fonts-nanum
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",           # macOS (로컬 개발용)
)
