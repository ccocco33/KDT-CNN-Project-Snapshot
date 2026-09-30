"""yunet_landmark 모델 설정
- 이 모델이 쓰는 값은 모두 여기에 둠. 다른 모델(landmark)의 config 를 가져다 쓰지 않음
"""
from ... import config

YUNET_PATH = config.MODEL_DIR / "face_detection_yunet_2023mar.onnx"   # 얼굴 검출 (1단계)
FACE_MODEL_PATH = config.MODEL_DIR / "face_landmarker.task"          # 잘라낸 얼굴 FaceLandmarker (2단계)
HAND_MODEL_PATH = config.MODEL_DIR / "hand_yolo.tflite"    # 손바닥 YOLO (LiteRT)

SCORE_TH = 0.7           # YuNet 얼굴 신뢰도 기준. 미만이면 버림 (WIDER dev 에서 0.9 보다 사진 조건 충족 precision 이 높음)
CROP_MARGIN = 0.5        # 얼굴을 잘라낼 때 여백 (얼굴 크기 비율)
YAW_TH = 0.5             # 보임 판정: 고개 돌림 yaw 가 이 값 이상이면 보이지 않음 (WIDER dev: 보이지 않는 얼굴 66% 잡음, 보이는 얼굴 3% 오판)
NUM_HANDS = 4            # 최대 검출 손 수
SMILE_TH = 0.5           # 웃음 판정: mouthSmile 평균 >= SMILE_TH
BLINK_TH = 0.5           # 눈 뜸 판정: eyeBlink 최대 < BLINK_TH
HAND_SCORE_TH = 0.5      # 손 들기 판정: 손바닥 YOLO 점수 >= HAND_SCORE_TH
HAND_IOU_TH = 0.45       # 손 YOLO NMS IoU 기준
NUM_THREADS = 4          # 손 YOLO LiteRT 추론 스레드 수
