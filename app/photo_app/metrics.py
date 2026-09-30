"""시간 기록 로그
- JSON Lines: 한 줄에 이벤트 1개 {"t": epoch 초, "ev": 이벤트명, ...}
- 실행마다 파일 1개: {LOG_DIR}/{YYYYmmdd_HHMMSS}.jsonl
- 여러 스레드에서 호출 가능
- start() 전에는 기록하지 않음 (테스트, 와이어프레임)
- 이벤트
  - predict: 추론 1회. caller(agent/check), state, 요청(smile/eye/hand/visibility), faces, 얼굴 높이 목록 face_h,
    안내 대상 수 guided, 이유별 guided_not_visible, guided_too_small (visibility 요청 시), hands, total_ms, 단계별 *_ms
  - capture: 촬영 1회. frames, duration_s, fps
  - checking: 판정 1회. frames, ok, picked, fallback, save_ms, predict_ms, select_ms, total_ms,
    eye_hold_allowed (EYE_HOLD 이면 눈 감음 허용 규칙으로 눈 뜸으로 본 얼굴 x 프레임 수, 꺼져 있으면 null)
  - camera_fps: 1초마다 카메라 FPS
  - screen_fps: 1초마다 화면 FPS. state, 화면 1장 단계별 평균 render_ms, imshow_ms, waitkey_ms,
    waitkey_extra_ms (waitKey 가 요청한 대기보다 더 걸린 시간 = X 화면 전송으로 막힌 시간), waitkey_extra_max_ms
  - run: 실행 시작. mode, model, screen, camera (요청 해상도), camera_actual (실제로 열린 해상도)
  - state: 상태 전환. from, to
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

from . import config
from .model import NOT_VISIBLE, TOO_SMALL

_lock = threading.Lock()
_file = None


def start(log_dir: Path = config.LOG_DIR, now=datetime.now) -> Path:
    global _file
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with _lock:
        _file = open(path, "a", encoding="utf-8", buffering=1)   # 줄 단위 flush
    return path


def stop() -> None:
    global _file
    with _lock:
        if _file is not None:
            _file.close()
            _file = None


def log(ev: str, **fields) -> None:
    line = json.dumps({"t": time.time(), "ev": ev, **fields}, ensure_ascii=False)
    with _lock:
        if _file is not None:
            _file.write(line + "\n")


def log_prediction(pred, caller: str, state: str) -> None:
    """Prediction -> predict 이벤트"""
    reasons = [r for _, r in pred.guidance(config.MIN_FACE_H)] if "visibility" in pred.requested else None
    log(
        "predict",
        caller=caller,
        state=state,
        smile="smile" in pred.requested,
        eye="eye" in pred.requested,
        hand="hand" in pred.requested,
        visibility="visibility" in pred.requested,
        guided=None if reasons is None else len(reasons),
        guided_not_visible=None if reasons is None else reasons.count(NOT_VISIBLE),
        guided_too_small=None if reasons is None else reasons.count(TOO_SMALL),
        faces=len(pred.faces),
        face_h=[f.bbox[3] for f in pred.faces],
        hands=None if pred.hands is None else len(pred.hands),
        total_ms=round(pred.elapsed_ms, 3),
        **{f"{k}_ms": round(v, 3) for k, v in pred.timings.items()},
    )
