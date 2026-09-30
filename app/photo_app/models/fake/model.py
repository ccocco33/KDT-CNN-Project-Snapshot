"""가짜 모델
- 얼굴: OpenCV YuNet 검출기. 모델 파일이 없으면 화면 가운데 가짜 얼굴 1개
- 웃음, 눈, 손 들기, 보임: 키보드 입력으로 흉내 (모든 얼굴에 같은 값)
  - s: 웃음 토글
  - e: 눈 뜸 토글
  - h: 손 들기
  - o: 보이지 않음 토글 (보이지 않으면 웃음, 눈 뜸 None)
"""
from __future__ import annotations

import threading
import time

import cv2

from . import config
from ...model import Face, Model, Prediction, requested
from ..util.keyboard import KeyboardHand


class FakeModel(Model):
    def __init__(self, yunet_path=config.YUNET_PATH, score_th: float = config.SCORE_TH):
        self.smile = False
        self.eyes_open = True
        self.visible = True
        self._hand = KeyboardHand()
        self._lock = threading.Lock()   # 판정 스레드와 메인 스레드(CHECKING)가 같이 호출
        self._detector = None
        if yunet_path.exists():
            self._detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), score_threshold=score_th)

    def handle_key(self, key: str) -> None:
        if key == "s":
            self.smile = not self.smile
        elif key == "e":
            self.eyes_open = not self.eyes_open
        elif key == "o":
            self.visible = not self.visible
        self._hand.handle_key(key)

    def predict(self, frame, *, smile: bool, eye: bool, hand: bool, visibility: bool) -> Prediction:
        start = time.perf_counter()
        with self._lock:
            boxes = self._detect(frame)
        face_ms = (time.perf_counter() - start) * 1000
        vis = self.visible if visibility else None
        judged = vis is not False   # 보이지 않는 얼굴은 웃음, 눈 판정 안 함
        eye_score = float(self.eyes_open) if eye and judged else None   # 뜸 1.0, 감음 0.0
        faces = [Face(b, self.smile if smile and judged else None, self.eyes_open if eye and judged else None, vis, eye_score)
                 for b in boxes]
        hands = [] if hand else None                          # 키보드 흉내라 손 bbox 없음
        raised = self._hand.raised() if hand else None
        total_ms = (time.perf_counter() - start) * 1000
        return Prediction(faces, hands, raised, total_ms, requested(smile, eye, hand, visibility), {"face": face_ms})

    def _detect(self, frame) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        if self._detector is None:
            return [(w // 2 - 60, h // 2 - 80, 120, 160)]
        self._detector.setInputSize((w, h))
        _, found = self._detector.detect(frame)
        if found is None:
            return []
        return [tuple(int(v) for v in f[:4]) for f in found]
