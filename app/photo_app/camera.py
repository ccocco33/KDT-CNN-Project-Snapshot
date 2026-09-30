"""카메라 스레드
- 최신 프레임 1장 보관
- 촬영 중(start_recording ~ stop_recording)에는 프레임 전부 누적
- 연결 상태 확인
  - 새 프레임이 CAMERA_LOST_SEC 이상 오지 않으면 끊김 (lost = True). 최신 프레임을 비움 (멈춘 화면으로 판정하지 않도록)
    - 시작 직후 첫 프레임을 기다리는 동안은 끊김이 아님 (안내가 깜빡이지 않도록)
  - 끊긴 동안 CAMERA_RETRY_SEC 마다 장치를 닫았다 다시 엶. 새 프레임이 오면 연결됨
  - 시작할 때 카메라가 없어도 앱을 멈추지 않고 같은 방식으로 다시 시도
  - 기록: camera_lost (끊김), camera_restored (다시 연결)
- 카메라를 여는 함수(opener), 시계(now) 주입: 테스트에서 카메라 없이 끊김, 재연결 흉내
"""
from __future__ import annotations

import threading
import time

import cv2

from . import config, metrics


class Camera:
    def __init__(self, index: int = config.CAMERA_INDEX, opener=cv2.VideoCapture, now=time.monotonic,
                 lost_sec: float = config.CAMERA_LOST_SEC, retry_sec: float = config.CAMERA_RETRY_SEC):
        self._index = index
        self._opener = opener
        self._now = now
        self._lost_sec = lost_sec
        self._retry_sec = retry_sec
        self._lock = threading.Lock()
        self._frame = None
        self._ts = 0.0
        self._recording: list | None = None   # 촬영 중이면 [(ts, frame), ...]
        self._cap = None
        self.lost = False                       # 끊김 여부 (앱이 안내, 촬영 취소에 씀)
        self._last_frame = now()                # 마지막으로 프레임을 받은 시각 (끊김 판단)
        self._next_open = 0.0                   # 다음 열기 시도 시각
        self._fps_count, self._fps_window = 0, now()
        self._open()
        self._running = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join(timeout=1)
        self._release()

    @property
    def size(self) -> tuple[int, int]:
        """실제로 열린 해상도 (너비, 높이). 요청한 CAMERA_W, CAMERA_H 와 다를 수 있음. 열려 있지 않으면 (0, 0)"""
        if self._cap is None:
            return 0, 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def latest(self):
        """(최신 프레임, 타임스탬프). 아직 프레임이 없거나 끊겼으면 (None, 0.0)"""
        with self._lock:
            return self._frame, self._ts

    def start_recording(self):
        with self._lock:
            self._recording = []

    def stop_recording(self) -> list:
        """누적한 [(ts, frame), ...] 반환 후 누적 종료"""
        with self._lock:
            frames, self._recording = self._recording or [], None
        return frames

    def _open(self) -> None:
        """장치 열기. 실패하면 CAMERA_RETRY_SEC 뒤에 다시 시도"""
        self._release()
        cap = self._opener(self._index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))   # 해상도보다 먼저 (포맷별 지원 해상도가 다름)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_H)
            cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 오래된 프레임이 쌓이지 않게 버퍼 1장 (지연 줄임)
            self._cap = cap
        else:
            cap.release()
        self._next_open = self._now() + self._retry_sec

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _run(self):
        while self._running:
            if not self._step():
                time.sleep(0.01)

    def _step(self) -> bool:
        """카메라 1회 처리. 새 프레임을 받았으면 True"""
        now = self._now()
        ok, frame = self._cap.read() if self._cap is not None else (False, None)
        if ok:
            self._on_frame(frame, now)
            return True
        if now - self._last_frame >= self._lost_sec:
            self._on_lost()
            if now >= self._next_open:
                self._open()
        return False

    def _on_frame(self, frame, ts: float) -> None:
        self._last_frame = ts
        if self.lost:
            self.lost = False
            metrics.log("camera_restored")
        self._fps_count += 1
        if ts - self._fps_window >= 1.0:   # 1초마다 FPS 기록
            metrics.log("camera_fps", fps=round(self._fps_count / (ts - self._fps_window), 2))
            self._fps_count, self._fps_window = 0, ts
        with self._lock:
            self._frame, self._ts = frame, ts
            if self._recording is not None:
                self._recording.append((ts, frame))

    def _on_lost(self) -> None:
        if self.lost:
            return
        self.lost = True
        with self._lock:
            self._frame = None
        metrics.log("camera_lost", index=self._index)
