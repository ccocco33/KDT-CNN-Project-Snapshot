"""화면
- 상태별 화면 그리기: render(state, view) -> BGR 이미지 (SCREEN_W x SCREEN_H)
- 버튼 위치 계산, 클릭 좌표 -> 버튼 id
- 한글 문구는 PIL 로 그림 (OpenCV putText 는 한글 미지원)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config
from .model import NOT_VISIBLE, TOO_SMALL, Face, Hand
from .models import NAMES as MODEL_NAMES
from .states import State

# 색 (BGR)
BG = (40, 40, 40)
WHITE = (255, 255, 255)
GRAY = (170, 170, 170)
RED = (60, 60, 230)
YELLOW = (0, 210, 255)
CYAN = (255, 200, 0)
ORANGE = (0, 140, 255)
GREEN = (80, 220, 80)
BUTTON = (90, 90, 90)
BUTTON_ON = (60, 150, 60)   # 모델 평가 화면에서 선택된 버튼
BLACK = (0, 0, 0)

CAMERA_STATES = (State.PREVIEW, State.PREPARE, State.CAPTURE, State.MODEL_EVAL)
EVAL_ITEMS = ("smile", "eye", "hand", "visibility", "eye_hold")   # 모델 평가 화면 항목 버튼 (model_eval.ITEMS + EXTRAS 순서)
GUIDE_TEXT = {                       # PREVIEW 안내 이유별 문구
    NOT_VISIBLE: "얼굴이 보이도록 해주세요",
    TOO_SMALL: "조금 더 가까이 와주세요",
}


@dataclass
class Button:
    id: str
    label: str
    rect: tuple[int, int, int, int]   # x, y, w, h

    def contains(self, x: int, y: int) -> bool:
        bx, by, bw, bh = self.rect
        return bx <= x < bx + bw and by <= y < by + bh


@dataclass
class View:
    """화면에 그릴 데이터"""
    frame: np.ndarray | None = None                          # 카메라 프레임 (BGR)
    faces: list[Face] = field(default_factory=list)          # bbox 는 frame 좌표 기준
    remaining_sec: float = 0.0                               # 남은 준비/촬영 시간
    show_cancel_msg: bool = False                            # 준비 실패 직후 문구 표시 여부
    results: list[np.ndarray] = field(default_factory=list)  # 결과 사진 (BGR)
    results_closest: bool = False                            # 결과가 조건에 가장 가까운 사진 (조건 충족 사진이 없을 때)
    camera_lost: bool = False                                # 카메라 끊김: 카메라 화면 대신 안내
    hands: list[Hand] = field(default_factory=list)          # bbox 는 frame 좌표 기준
    guides: list[tuple[Face, str]] = field(default_factory=list)   # PREVIEW 안내 대상 [(얼굴, 이유)]. bbox 는 frame 좌표 기준
    dev: bool = False                                        # dev 모드: 얼굴 bbox, 손 bbox, dev_label 표시
    dev_label: str = ""                                      # dev 모드 오른쪽 아래 문구 (예: dev | model: landmark)
    eval_model: str = ""                                     # 모델 평가: 선택한 모델
    eval_items: frozenset[str] = frozenset()                 # 모델 평가: 켠 판정 항목
    eval_timings: dict[str, float] = field(default_factory=dict)   # 모델 평가: 판정 시간 {total, 단계: ms}
    eval_msg: str = ""                                       # 모델 평가: 알림 (찍기 저장 경로 등)
    eval_burst: bool = False                                 # 모델 평가: 연속 촬영 중


def buttons(state: State, dev: bool = False) -> list[Button]:
    """상태별 버튼 목록
    - PREPARE, CAPTURE, CHECKING: 버튼 없음
    - dev 모드: HOME 에 모델 평가(eval) 버튼
    - MODEL_EVAL: 모델 선택 (model:이름), 판정 항목 (item:이름), 찍기 (shot), 연속 촬영 켜기/끄기 (burst), 처음으로 (home)
    """
    w, h = config.SCREEN_W, config.SCREEN_H
    home = Button("home", "처음으로", (w - 110, 10, 100, 40))
    if state is State.HOME:
        start = [Button("start", "촬영 시작", (w // 2 - 100, h // 2 + 20, 200, 60))]
        return start + ([Button("eval", "모델 평가", (w // 2 - 100, h // 2 + 95, 200, 40))] if dev else [])
    if state is State.MODEL_EVAL:
        bw = (w - 130) // len(MODEL_NAMES)
        models = [Button(f"model:{n}", n, (10 + i * bw, 10, bw - 6, 34)) for i, n in enumerate(MODEL_NAMES)]
        items = [Button(f"item:{n}", n, (10 + i * 100, 50, 94, 28)) for i, n in enumerate(EVAL_ITEMS)]
        return models + items + [home, Button("burst", "연속", (w - 110, h - 106, 100, 40)),
                                 Button("shot", "찍기", (w - 110, h - 60, 100, 40))]
    if state is State.PREVIEW:
        return [home]
    if state is State.RESULT:
        return [
            Button("preview", "다시 찍기", (w // 2 - 210, h - 60, 200, 45)),
            Button("home", "처음으로", (w // 2 + 10, h - 60, 200, 45)),
        ]
    return []


def button_at(state: State, x: int, y: int, dev: bool = False) -> str | None:
    """클릭 좌표의 버튼 id. 버튼 밖이면 None"""
    for b in buttons(state, dev):
        if b.contains(x, y):
            return b.id
    return None


CAMERA_LOST_MSG = "카메라 연결을 확인해 주세요"


def render(state: State, view: View) -> np.ndarray:
    """
    설정한 해상도대로 렌더링
    CAMERA_STATES 경우: frame을 리사이징. 카메라가 끊겼으면 상태별 화면 대신 안내 문구 (버튼은 그대로)
    아닌 경우         : frame 크기의 numpy 배열 생성.
    """
    w, h = config.SCREEN_W, config.SCREEN_H
    if state in CAMERA_STATES:
        img, scale = _camera_background(view.frame, w, h)
    else:
        img, scale = np.full((h, w, 3), BG, np.uint8), (1.0, 1.0)
    texts: list[tuple] = []   # (문구, (x, y), 크기, 색, anchor). 도형을 다 그린 뒤 한 번에 그림

    if state in CAMERA_STATES and view.camera_lost:
        texts.append((CAMERA_LOST_MSG, (w // 2, h // 2), 24, WHITE, "mm"))
    else:
        if view.dev and state in CAMERA_STATES and state is not State.MODEL_EVAL:
            _draw_faces(img, view, scale, texts)   # 상태별 박스(안내, 미소)가 위에 보이도록 먼저 그림
        _DRAW[state](img, view, scale, texts)
        if view.dev and state in CAMERA_STATES:
            _draw_hands(img, view, scale, texts)

    for b in buttons(state, view.dev):
        x, y, bw, bh = b.rect
        on = (b.id == f"model:{view.eval_model}" or (b.id.startswith("item:") and b.id[5:] in view.eval_items)
              or (b.id == "burst" and view.eval_burst))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), BUTTON_ON if on else BUTTON, -1)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), WHITE, 1)
        size = 14 if state is State.MODEL_EVAL and b.id.split(":")[0] in ("model", "item") else 18
        texts.append((b.label, (x + bw // 2, y + bh // 2), size, WHITE, "mm"))

    if view.dev and view.dev_label:
        texts.append((view.dev_label, (config.SCREEN_W - 8, config.SCREEN_H - 8), 12, YELLOW, "rd"))

    return _draw_texts(img, texts)


# ---- 상태별 그리기 ----

def _draw_home(img, view, scale, texts):
    w, h = config.SCREEN_W, config.SCREEN_H
    n = config.WARN_PEOPLE if config.WARN_PEOPLE is not None else "N"
    texts.append((config.APP_NAME, (w // 2, h // 2 - 70), 36, WHITE, "mm"))
    texts.append((f"{n}명 이상 촬영시 앱이 느려질 수 있습니다.", (w // 2, h - 30), 16, GRAY, "mm"))
    texts.append((f"d: 개발자 모드 {'끄기' if view.dev else '켜기'}", (10, 18), 13, GRAY, "lm"))   # 모드 전환 안내 (초기화면만)


def _draw_preview(img, view, scale, texts):
    w, h = config.SCREEN_W, config.SCREEN_H
    people = len(view.faces)
    warn = config.WARN_PEOPLE is not None and people >= config.WARN_PEOPLE
    _box(img, (10, 10, 130, 40), BLACK, 0.5)
    texts.append((f"사람 수: {people}", (20, 30), 20, YELLOW if warn else WHITE, "lm"))
    _box(img, (0, h - 44, w, 44), BLACK, 0.6)
    texts.append(("손가락을 펴고 들면 사진 촬영이 시작됩니다.", (w // 2, h - 22), 20, WHITE, "mm"))
    for f, reason in view.guides:
        _face_label(img, f.bbox, scale, GUIDE_TEXT[reason], ORANGE, texts)
    if view.show_cancel_msg:
        _box(img, (40, h // 2 - 50, w - 80, 100), BLACK, 0.75)
        texts.append(("웃지 않으신 분이 있어 촬영을 취소했어요.", (w // 2, h // 2 - 16), 20, WHITE, "mm"))
        texts.append(("준비되면 다시 손을 들어 촬영을 시작해주세요.", (w // 2, h // 2 + 16), 20, WHITE, "mm"))


def _draw_prepare(img, view, scale, texts):
    w = config.SCREEN_W
    for f in view.faces:
        if not f.smile:
            _face_label(img, f.bbox, scale, "미소 지어주세요", RED, texts)
    _box(img, (w // 2 - 80, 10, 160, 44), BLACK, 0.5)
    texts.append((f"준비 {view.remaining_sec:.0f}초", (w // 2, 32), 24, WHITE, "mm"))


def _draw_capture(img, view, scale, texts):
    w = config.SCREEN_W
    _box(img, (w // 2 - 90, 10, 180, 44), BLACK, 0.5)
    cv2.circle(img, (w // 2 - 65, 32), 8, RED, -1)
    texts.append((f"촬영 중 {view.remaining_sec:.1f}초", (w // 2 + 10, 32), 22, WHITE, "mm"))


def _draw_checking(img, view, scale, texts):
    w, h = config.SCREEN_W, config.SCREEN_H
    texts.append(("사진을 고르는 중", (w // 2, h // 2), 28, WHITE, "mm"))


CLOSEST_MSG = "모두 조건에 맞는 사진은 없어요. 가장 가까운 사진을 보여드려요"


def _draw_result(img, view, scale, texts):
    w, h = config.SCREEN_W, config.SCREEN_H
    if not view.results:
        texts.append(("조건에 맞는 사진이 없어요", (w // 2, h // 2 - 30), 26, WHITE, "mm"))
        return
    if view.results_closest:
        texts.append((CLOSEST_MSG, (w // 2, 22), 16, WHITE, "mm"))
    else:
        texts.append((f"조건을 충족한 사진 {len(view.results)}장", (w // 2, 22), 20, WHITE, "mm"))
    tw, th, gap = 190, 142, 12   # 썸네일 크기, 간격 (4:3)
    rows = [view.results[:3], view.results[3:5]]
    for r, row in enumerate(rows):
        x0 = (w - (len(row) * tw + (len(row) - 1) * gap)) // 2
        y = 44 + r * (th + gap)
        for i, photo in enumerate(row):
            x = x0 + i * (tw + gap)
            img[y:y + th, x:x + tw] = cv2.resize(photo, (tw, th))
            cv2.rectangle(img, (x, y), (x + tw, y + th), WHITE, 1)


def _draw_faces(img, view, scale, texts):
    """dev 모드 전용: 검출된 얼굴 bbox 와 판정 점수 (YuNet 등 모델이 찾은 얼굴 전부)
    - 박스 아래 띠: "웃음 0.95 O", "눈 뜸 0.97 O" (판정한 항목만). 보이지 않는 얼굴은 "얼굴 가림"
      - 웃음, 눈 점수가 없으면 가림 점수 "가림 0.42", 그것도 없으면 "face"
      - 안내(주황), 미소(빨강) 띠는 박스 위라 겹치지 않도록 아래에 둠. 화면 아래 공간이 없으면 박스 위
    - 색: 표시한 판정이 모두 통과면 초록, 아니면 빨강
    - CAPTURE 는 dev 모드에서만 약 5 FPS 로 판정해 표시 (app.DEV_CAPTURE_REQUEST)
    """
    sx, sy = scale
    for f in view.faces:
        x, y, fw, fh = f.bbox
        x1, y1, x2, y2 = int(x * sx), int(y * sy), int((x + fw) * sx), int((y + fh) * sy)
        lines, ok = _face_score_lines(f)
        color = GREEN if ok else RED
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        size = 13
        band_w = max(int(_font(size).getlength(line)) for line in lines) + 10
        band_h = 17 * len(lines) + 4
        top = y2 + 2 if y2 + 2 + band_h <= config.SCREEN_H else max(y1 - 2 - band_h, 0)
        left = min(max(x1, 0), config.SCREEN_W - band_w)
        cv2.rectangle(img, (left, top), (left + band_w, top + band_h), color, -1)
        for i, line in enumerate(lines):
            texts.append((line, (left + 5, top + 3 + 17 * i), size, WHITE, "la"))


def _face_score_lines(f) -> tuple[list[str], bool]:
    """dev 얼굴 띠 글자와 통과 여부. 눈 감음 점수 (blink, landmark 계열) 는 눈 뜸 = 1 - 점수로 표시"""
    if f.visible is False:
        return ["얼굴 가림"], False
    lines, ok = [], True
    if "smile" in f.scores:
        v, _, passed = f.scores["smile"]
        lines.append(f"웃음 {v:.2f} {'O' if passed else 'X'}")
        ok = ok and passed
    if "eye" in f.scores or "blink" in f.scores:
        v, _, passed = f.scores["eye"] if "eye" in f.scores else (1 - f.scores["blink"][0], None, f.scores["blink"][2])
        lines.append(f"눈 뜸 {v:.2f} {'O' if passed else 'X'}")
        ok = ok and passed
    if not lines:
        lines = [f"가림 {f.occlusion:.2f}" if f.occlusion is not None else "face"]
    return lines, ok


def eval_passed(f) -> bool:
    """모델 평가 화면 얼굴 통과 여부: 모든 판정 항목 통과. eye_hold 로 허용한 얼굴은 eye 가 X 여도 통과"""
    held = f.scores.get("eye_hold", (0, 0, False))[2]
    return all(passed or (k == "eye" and held) for k, (_, _, passed) in f.scores.items() if k != "eye_hold")


def _draw_model_eval(img, view, scale, texts):
    """모델 평가 화면
    - 얼굴 박스: 모든 판정 항목 통과면 초록, 하나라도 아니면 빨강. 박스 아래 "항목 점수/기준값 O|X" 한 줄씩
      - eye_hold 줄: "eye_hold 연속 눈 감음 초/기준 초 O|X". O (허용) 이면 eye 가 X 여도 통과로 봄
    - 왼쪽 아래: 판정 시간 (전체, 단계별 ms), 알림
    """
    sx, sy = scale
    for f in view.faces:
        x, y, fw, fh = f.bbox
        x1, y1, x2, y2 = int(x * sx), int(y * sy), int((x + fw) * sx), int((y + fh) * sy)
        ok = eval_passed(f)
        color = GREEN if ok else RED
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        lines = [f"{k} {s:.2f}/{th:g} {'O' if passed else 'X'}" for k, (s, th, passed) in f.scores.items()] or ["(점수 없음)"]
        panel_h = 16 * len(lines) + 4
        top = y2 + 2 if y2 + 2 + panel_h <= config.SCREEN_H else max(y1 - 2 - panel_h, 0)   # 아래 공간이 없으면 박스 위
        left = min(max(x1, 0), config.SCREEN_W - 150)
        _box(img, (left, top, 150, panel_h), BLACK, 0.6)
        for i, line in enumerate(lines):
            texts.append((line, (left + 4, top + 2 + 16 * i), 13, color, "la"))
    t = view.eval_timings
    if t:
        detail = ", ".join(f"{k} {v:.0f}" for k, v in t.items() if k != "total")
        info = f"{view.eval_model}  {t.get('total', 0):.0f} ms ({1000 / max(t.get('total', 1), 1):.1f} FPS)  {detail}"
    else:
        info = f"{view.eval_model}  판정 대기 중"
    _box(img, (0, config.SCREEN_H - 44, config.SCREEN_W - 120, 44), BLACK, 0.6)
    texts.append((info, (8, config.SCREEN_H - 30), 13, WHITE, "lm"))
    if view.eval_msg:
        texts.append((view.eval_msg, (8, config.SCREEN_H - 12), 13, YELLOW, "lm"))


def _draw_hands(img, view, scale, texts):
    """dev 모드 전용: 손 bbox
    - 든 손은 노란색, 아닌 손은 하늘색
    - 글자는 판정 이유 (Hand.pose. 없으면 hand up / hand)
    """
    sx, sy = scale
    for hand in view.hands:
        x, y, hw, hh = hand.bbox
        color, label = (YELLOW, "hand up") if hand.raised else (CYAN, "hand")
        label = hand.pose or label
        x1, y1 = int(x * sx), int(y * sy)
        cv2.rectangle(img, (x1, y1), (int((x + hw) * sx), int((y + hh) * sy)), color, 2)
        texts.append((label, (x1 + 3, y1 + 3), 14, color, "la"))


_DRAW = {
    State.HOME: _draw_home,
    State.PREVIEW: _draw_preview,
    State.PREPARE: _draw_prepare,
    State.CAPTURE: _draw_capture,
    State.CHECKING: _draw_checking,
    State.RESULT: _draw_result,
    State.MODEL_EVAL: _draw_model_eval,
}


# ---- 그리기 도구 ----

def _face_label(img, bbox, scale, label, color, texts, size=16):
    """얼굴 박스와 박스 위 문구 띠 (PREPARE 미소 안내, PREVIEW 안내)"""
    sx, sy = scale
    x, y, fw, fh = bbox
    x1, y1, x2, y2 = int(x * sx), int(y * sy), int((x + fw) * sx), int((y + fh) * sy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    lw = int(_font(size).getlength(label)) + 12
    ly = max(y1 - 26, 0)
    cv2.rectangle(img, (x1 - 1, ly), (x1 - 1 + lw, ly + 26), color, -1)   # 문구 배경 띠
    texts.append((label, (x1 + 5, ly + 13), size, WHITE, "lm"))


def _camera_background(frame, w, h):
    """카메라 프레임을 화면 크기로 맞춤
    - 반환: (이미지, (x 배율, y 배율)). 배율은 bbox 좌표 변환용
    - frame 이 없으면 회색 화면
    """
    if frame is None:
        return np.full((h, w, 3), GRAY, np.uint8), (1.0, 1.0)
    fh, fw = frame.shape[:2]
    return cv2.resize(frame, (w, h)), (w / fw, h / fh)


def _box(img, rect, color, alpha):
    """반투명 사각형. 화면 밖은 잘라내고, 남는 영역이 없으면 그리지 않음"""
    x, y, bw, bh = rect
    x1, y1 = max(x, 0), max(y, 0)
    roi = img[y1:y + bh, x1:x + bw]
    if roi.size == 0:
        return
    roi[:] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0)


@lru_cache(maxsize=None)
def _font(size: int):
    for path in config.FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_texts(img, texts):
    """모은 문구를 PIL 로 한 번에 그림
    - BGR -> RGB 변환은 한 번만
    """
    if not texts:
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, xy, size, bgr, anchor in texts:
        draw.text(xy, text, font=_font(size), fill=bgr[::-1], anchor=anchor)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
