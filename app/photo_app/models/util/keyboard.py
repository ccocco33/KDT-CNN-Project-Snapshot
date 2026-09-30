"""키보드로 손 들기 흉내
- h 누른 뒤 HOLD_SEC 동안 손 든 상태
- 손 들기 모델 전까지 사용
"""
import time


class KeyboardHand:
    HOLD_SEC = 1.0

    def __init__(self, now=time.monotonic):
        self._now = now
        self._until = 0.0

    def handle_key(self, key: str) -> None:
        if key == "h":
            self._until = self._now() + self.HOLD_SEC

    def raised(self) -> bool:
        return self._now() < self._until
