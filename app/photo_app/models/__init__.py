"""모델 구현 모음
- create_model(name): 이름으로 모델 생성
- 구현별 의존성(mediapipe 등)은 선택된 모델만 import
"""
from __future__ import annotations

from ..model import Model

NAMES = ("fake", "landmark", "yunet_landmark", "yunet_cnn")


def create_model(name: str) -> Model:
    if name == "fake":
        from .fake import FakeModel
        return FakeModel()
    if name == "landmark":
        from .landmark import LandmarkModel
        return LandmarkModel()
    if name == "yunet_landmark":
        from .yunet_landmark import YunetLandmarkModel
        return YunetLandmarkModel()
    if name == "yunet_cnn":
        from .yunet_cnn import YunetCnnModel
        return YunetCnnModel()
    raise ValueError(f"알 수 없는 모델: {name} (가능: {', '.join(NAMES)})")
