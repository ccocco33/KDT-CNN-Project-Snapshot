"""landmark 모델 설정"""
from ... import config

FACE_MODEL_PATH = config.MODEL_DIR / "face_landmarker.task"   # FaceLandmarker
HAND_MODEL_PATH = config.MODEL_DIR / "hand_yolo.tflite"   # 손바닥 YOLO (LiteRT)

NUM_HANDS = 4            # 최대 검출 손 수
SMILE_TH = 0.5           # 웃음 판정: mouthSmile 평균 >= SMILE_TH
BLINK_TH = 0.5           # 눈 뜸 판정: eyeBlink 최대 < BLINK_TH
YAW_TH = 0.5             # 보임 판정: 고개 돌림 yaw 가 이 값 이상이면 보이지 않음 (yunet_landmark 측정값. landmark 기준 미측정)
HAND_SCORE_TH = 0.5      # 손 들기 판정: 손바닥 YOLO 점수 >= HAND_SCORE_TH
HAND_IOU_TH = 0.45       # 손 YOLO NMS IoU 기준
NUM_THREADS = 4          # 손 YOLO LiteRT 추론 스레드 수
