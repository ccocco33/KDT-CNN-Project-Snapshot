"""landmark 룰 판정
- 입력: FaceLandmarker blendshape {이름: 점수 0~1}, 얼굴 랜드마크 좌표
- 웃음: 양쪽 mouthSmile 평균 >= th
- 눈 뜸: 양쪽 eyeBlink 중 큰 값 < th (한쪽만 감아도 감은 것으로 봄)
- 손 들기 판정은 util/hand.py (yunet_cnn 도 같이 씀)
- 고개 돌림 yaw: 코가 두 눈 가운데에서 벗어난 정도
- 임계값은 모델마다 자기 config 값을 인자로 넘김 (여러 모델이 같이 쓰는 코드)
"""
from __future__ import annotations

import math


def smile_score(bs: dict[str, float]) -> float:
    return (bs["mouthSmileLeft"] + bs["mouthSmileRight"]) / 2


def blink_score(bs: dict[str, float]) -> float:
    return max(bs["eyeBlinkLeft"], bs["eyeBlinkRight"])


def is_smiling(bs: dict[str, float], th: float) -> bool:
    return smile_score(bs) >= th


def is_eyes_open(bs: dict[str, float], th: float) -> bool:
    return blink_score(bs) < th


def yaw(right_eye: tuple[float, float], left_eye: tuple[float, float], nose: tuple[float, float]) -> float:
    """고개 돌림 정도
    - |코 x - 두 눈 가운데 x| / 두 눈 사이 거리. 정면 0, 옆으로 돌릴수록 커짐
    - 좌표는 픽셀 (x, y). 두 눈이 겹치면 거리 1 로 봄
    - 반환은 파이썬 float (numpy 좌표가 들어와도 비교 결과가 numpy.bool_ 이 되지 않도록. `visible is False` 판정용)
    """
    dist = math.hypot(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1]) or 1.0
    return float(abs(nose[0] - (right_eye[0] + left_eye[0]) / 2) / dist)
