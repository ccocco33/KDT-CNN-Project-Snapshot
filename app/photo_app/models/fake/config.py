"""fake 모델 설정"""
from ... import config

YUNET_PATH = config.MODEL_DIR / "face_detection_yunet_2023mar.onnx"   # 얼굴 검출. 없으면 가운데 가짜 얼굴 1개
SCORE_TH = 0.7           # YuNet 얼굴 신뢰도 기준 (OpenCV 기본값 0.9 는 작은 얼굴을 "작음" 안내 전에 놓침)
