"""yunet_cnn 모델 설정
- 이 모델이 쓰는 값은 모두 여기에 둠. 다른 모델의 config 를 가져다 쓰지 않음
- CNN 파일은 app/models/{MODEL_VERSION}/ (버전별 파일, 성능은 docs/model_versions.md)
"""
from ... import config

MODEL_VERSION = "v0.0.7"
YUNET_PATH = config.MODEL_DIR / "face_detection_yunet_2023mar.onnx"   # 얼굴 검출
SMILE_PATH = config.MODEL_DIR / MODEL_VERSION / "smile_fp32.tflite"   # 웃음 CNN. 입력 얼굴 아래쪽 128x128 RGB 0~255, 출력 1 = 웃음
EYE_PATH = config.MODEL_DIR / MODEL_VERSION / "eye_fp32.tflite"       # 눈 CNN. 입력 얼굴 위쪽 160x160 RGB 0~255, 출력 1 = 뜸 (선글라스류 포함)
OCC_PATH = config.MODEL_DIR / MODEL_VERSION / "occlusion_fp32.tflite"  # 가림 CNN. 입력 얼굴 128x128 RGB 0~255, 출력 1 = 가림 (입 또는 두 눈)
HAND_PATH = config.MODEL_DIR / "hand_yolo.tflite"     # 손바닥 YOLO11n. 입력 320x320 RGB 0~1, 출력 손바닥 박스, 점수

YUNET_INPUT_WIDTH = 400  # YuNet 입력 가로 크기. 프레임을 이 크기로 줄여 검출 (라즈베리파이 YuNet 640 66 ms -> 400 23 ms)
                         # WIDER 1023장 보이는 얼굴 검출률 (정답 높이 50px 이상): 640 96.6%, 480 95.7%, 400 95.3%, 320 93.7%
SCORE_TH = 0.7           # YuNet 얼굴 신뢰도 기준. 미만이면 버림
YAW_TH = 0.6             # 보임 판정: 고개 돌림 yaw 가 이 값 이상이면 보이지 않음 (WIDER dev 보이는 얼굴 오판 2.0%, 옆모습 73% 잡음)
SMILE_SIZE = 128         # smile CNN 입력 크기
EYE_SIZE = 160           # eye CNN 입력 크기
OCC_SIZE = 128           # occlusion CNN 입력 크기
EYE_TOP = 0.6            # eye CNN 입력: 얼굴 정사각형의 위쪽 비율 (이마 ~ 코 중간)
SMILE_BOTTOM = 0.45      # smile CNN 입력: 얼굴 정사각형의 아래쪽 비율 (코끝 ~ 턱). 선글라스 렌즈(반사) 제외
SMILE_TH = 0.31          # 웃음 판정: smile 점수 >= SMILE_TH. 안 웃음을 웃음으로 보는 비율을 이전 모델 기준 0.5 와 같게 (WIDER dev 11%, 웃음 recall 77%. 학습 노트북 기준값 0.318)
EYE_TH = 0.84            # 눈 뜸 판정: eye 점수 >= EYE_TH
OCC_TH = 0.86            # 보임 판정: 가림 점수 >= OCC_TH 이면 보이지 않음 (WIDER dev 보이는 얼굴 오판 2% 기준. 가림 recall 40%. 학습 노트북 기준값 0.569 는 WIDER 오판 4.7%)
NUM_THREADS = 4          # LiteRT 추론 스레드 수
NUM_HANDS = 4            # 최대 검출 손 수
HAND_SCORE_TH = 0.5      # 손 들기 판정: 손바닥 YOLO 점수 >= HAND_SCORE_TH
HAND_IOU_TH = 0.45       # 손 YOLO NMS IoU 기준
