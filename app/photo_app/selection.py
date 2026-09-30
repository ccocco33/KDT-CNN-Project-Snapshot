"""결과 사진 선택 규칙
- select: 조건 충족 사진에서 고름
- select_closest: 조건 충족 사진이 없을 때 조건에 가장 가까운 사진을 고름
- EyeHold: 같은 얼굴이 오래 눈을 감고 있으면 눈 뜸으로 봄 (촬영 프레임을 시간 순서로 해석)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TypeVar

from . import config
from .model import Prediction

T = TypeVar("T")


def select(ok: list[tuple[T, Prediction]], n: int) -> list[T]:
    """조건 충족 사진에서 최대 n장 선택
    - ok: [(사진, 판정 결과), ...] 찍힌 순서. meets_condition() 을 통과한 사진만 (눈 뜸 점수가 항상 있음)
    - m <= n: 전부
    - m > n: 찍힌 순서대로 장수 기준 n개 구간으로 나누고, 구간마다 눈 뜸 점수가 가장 높은 사진 1장 (같으면 먼저 찍힌 사진)
      - 눈 뜸 점수: Prediction.eye_score() (사진 속 눈을 가장 작게 뜬 사람 기준)
      - 촬영 시간 전체에 고르게 퍼진 사진 중, 눈을 감으려는 순간처럼 눈이 작아진 사진을 피함
    - 반환: 선택한 사진 (찍힌 순서)
    """
    m = len(ok)
    if m <= n:
        return [item for item, _ in ok]
    bounds = [i * m // n for i in range(n + 1)]
    return [max(ok[a:b], key=lambda x: x[1].eye_score())[0] for a, b in zip(bounds, bounds[1:])]


LOWER_IS_PASS = frozenset({"blink"})   # 점수가 기준보다 작아야 통과하는 판정 항목 (눈 감음)
JUDGED = ("smile", "eye", "blink")     # 가까운 정도에 쓰는 판정 항목 (웃음, 눈)


def face_margin(face) -> float:
    """얼굴의 가장 모자란 값: 웃음, 눈 점수의 기준 대비 차이 중 최솟값 (클수록 조건에 가까움)
    - 차이: 점수 - 기준 (LOWER_IS_PASS 항목은 기준 - 점수). 통과한 항목은 0 이상
    - 보이지 않는 얼굴, 웃음이나 눈을 판정하지 못한 얼굴: -1
    - 점수가 없는 모델 (fake): 통과 여부로 대신 (통과 0, 미통과 -1)
    """
    if face.visible is False or face.smile is None or face.eyes_open is None:
        return -1.0
    margins = [(th - s) if k in LOWER_IS_PASS else (s - th)
               for k, (s, th, _) in face.scores.items() if k in JUDGED]
    if not margins:
        return 0.0 if face.smile and face.eyes_open else -1.0
    return min(margins)


def closeness(pred: Prediction) -> tuple[int, float]:
    """사진이 조건에 가까운 정도 (큰 쪽이 가까움)
    - 1순위: 조건을 충족한 얼굴 수 (보임, 웃음, 눈 뜸 모두 통과)
    - 2순위: 얼굴들의 가장 모자란 값 (face_margin) 중 최솟값
    """
    passed = sum(1 for f in pred.faces if f.visible is not False and f.smile and f.eyes_open)
    return passed, min(face_margin(f) for f in pred.faces)


def select_closest(cands: list[tuple[T, Prediction]], n: int) -> list[T]:
    """조건 충족 사진이 없을 때 조건에 가장 가까운 사진 최대 n장
    - cands: [(사진, 판정 결과), ...] 찍힌 순서. 얼굴이 없는 사진은 뺌
    - k (얼굴 있는 사진 수) <= n: 전부
    - k > n: 찍힌 순서대로 n개 구간으로 나누고, 구간마다 가장 가까운 사진 1장 (closeness. 같으면 먼저 찍힌 사진)
      - select 와 같은 방식. 비슷한 연속 사진만 고르지 않도록
    - 반환: 선택한 사진 (찍힌 순서)
    """
    cands = [(item, pred) for item, pred in cands if pred.faces]
    k = len(cands)
    if k <= n:
        return [item for item, _ in cands]
    bounds = [i * k // n for i in range(n + 1)]
    return [max(cands[a:b], key=lambda x: closeness(x[1]))[0] for a, b in zip(bounds, bounds[1:])]


@dataclass
class _Track:
    """이어서 보는 얼굴 1명: 마지막 박스, 눈 감음 시작 시각, 연속 눈 감음 관측 수"""
    box: tuple[float, float, float, float]
    closed_since: float | None = None
    samples: int = 0


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _best(values: list[float], th: float, margin: float) -> int | None:
    """th 이상 중 가장 큰 값의 번호. 없거나 두 번째와의 차이가 margin 미만(애매함)이면 None"""
    ranked = sorted(((v, i) for i, v in enumerate(values) if v >= th), reverse=True)
    if not ranked or (len(ranked) > 1 and ranked[0][0] - ranked[1][0] < margin):
        return None
    return ranked[0][1]


class EyeHold:
    """같은 얼굴이 촬영 시각 기준 EYE_HOLD_SEC 이상 연속 눈 감음이면 그 프레임부터 눈 뜸으로 봄
    - 촬영 1번 (또는 모델 평가 화면 켠 동안) 마다 새로 만들어 프레임을 시간 순서로 apply
    - 같은 얼굴: 이전 프레임 얼굴과 박스 IoU 가 서로 가장 큰 짝 (IoU >= EYE_HOLD_IOU, 애매하면 잇지 않음). 얼굴 인식은 아님
    - 허용: 연속 눈 감음 시간 >= EYE_HOLD_SEC 이고 관측 수 >= EYE_HOLD_MIN_SAMPLES. 그 전 프레임은 소급하지 않음
    - 누적 초기화: 눈 뜸, 판정 불가, 안 보임, 얼굴 높이 < MIN_FACE_H, 짝 없음,
      이웃 프레임 간격 > EYE_HOLD_MAX_GAP_SEC, 촬영 시각이 늘지 않음, eye 를 요청하지 않은 판정
    - 반환: 허용한 얼굴만 eyes_open=True 로 바꾼 복사본. 원래 판정과 eye_score 는 그대로 (select 순위도 그대로)
      - 모든 얼굴의 scores["eye_hold"] = (연속 눈 감음 시간, EYE_HOLD_SEC, 허용 여부)
    """

    def __init__(self):
        self._tracks: list[_Track] = []
        self._last_ts: float | None = None
        self.allowed = 0   # 허용한 얼굴 x 프레임 수 (로그용)

    def apply(self, ts: float, pred: Prediction) -> Prediction:
        if self._last_ts is not None and not (0 < ts - self._last_ts <= config.EYE_HOLD_MAX_GAP_SEC):
            self._tracks = []
        self._last_ts = ts
        if "eye" not in pred.requested:
            self._tracks = []
            return pred
        boxes = [tuple(float(v) for v in f.bbox) for f in pred.faces]
        iou = [[_iou(t.box, b) for b in boxes] for t in self._tracks]
        row_best = [_best(row, config.EYE_HOLD_IOU, config.EYE_HOLD_AMBIGUITY) for row in iou]
        col_best = [_best([row[j] for row in iou], config.EYE_HOLD_IOU, config.EYE_HOLD_AMBIGUITY)
                    for j in range(len(boxes))]
        tracks, faces = [], []
        for j, (face, box) in enumerate(zip(pred.faces, boxes)):
            i = col_best[j]
            track = replace(self._tracks[i], box=box) if i is not None and row_best[i] == j else _Track(box)
            closed = face.eyes_open is not None and not face.eyes_open   # numpy bool 도 받음
            judged = (face.visible is not None and bool(face.visible) and face.eye_score is not None
                      and math.isfinite(face.eye_score) and box[3] >= config.MIN_FACE_H)
            held, allow = 0.0, False
            if closed and judged:
                if track.closed_since is None:
                    track.closed_since, track.samples = ts, 0
                track.samples += 1
                held = ts - track.closed_since
                allow = held + 1e-9 >= config.EYE_HOLD_SEC and track.samples >= config.EYE_HOLD_MIN_SAMPLES
            else:
                track.closed_since, track.samples = None, 0
            self.allowed += allow
            scores = {**face.scores, "eye_hold": (held, config.EYE_HOLD_SEC, allow)}
            faces.append(replace(face, eyes_open=True if allow else face.eyes_open, scores=scores))
            tracks.append(track)
        self._tracks = tracks
        return replace(pred, faces=faces)
