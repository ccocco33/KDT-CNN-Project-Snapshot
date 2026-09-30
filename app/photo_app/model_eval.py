"""모델 평가 화면 (dev 모드, MODEL_EVAL) 의 선택 상태와 찍기
- 모델 선택: models.NAMES 중 하나. 처음 고를 때 만들어 보관 (다시 고르면 다시 만들지 않음)
- 판정 항목 선택: smile, eye, hand, visibility 를 켜고 끔. 켠 항목만 predict 에 요청
- 해석 항목 선택: eye_hold (눈 감음 허용 규칙, selection.EyeHold) 를 켜고 끔. predict 에는 넘기지 않음. 처음에는 꺼짐
- 찍기: 판정한 프레임 + 판정 결과를 {root}/eval/{시각}.jpg, .json 으로 저장
  - json: 모델, 요청 항목, 얼굴별 bbox, 판정, scores (점수, 기준값, 통과), 손, 단계별 시간
- 연속 촬영: 켜 두면 EVAL_BURST_SEC 마다 저장. 켤 때마다 {root}/eval/burst_{시각}/ 폴더에 모음
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2

from . import config
from .model import Model, Prediction
from .models import NAMES, create_model

ITEMS = ("smile", "eye", "hand", "visibility")   # predict 요청 항목
EXTRAS = ("eye_hold",)                            # 판정 결과 해석 항목 (predict 에 넘기지 않음)


class ModelEval:
    def __init__(self, first: str, first_model: Model | None = None, create=create_model):
        self._create = create
        self._models: dict[str, Model] = {first: first_model} if first_model is not None else {}
        self.name = first
        self.items = set(ITEMS)   # 켠 판정 항목
        self.changed_at = 0.0     # 모델, 항목을 바꾼 시각 (이전 판정 결과를 버리는 기준)
        self.burst_dir: Path | None = None   # 연속 촬영 폴더. None 이면 연속 촬영 꺼짐
        self.burst_count = 0
        self._burst_next = 0.0

    @property
    def bursting(self) -> bool:
        return self.burst_dir is not None

    def start_burst(self, now: float, root: Path = config.SAVE_DIR, clock=datetime.now) -> None:
        self.burst_dir = Path(root) / "eval" / f"burst_{clock().strftime('%Y%m%d_%H%M%S')}"
        self.burst_count = 0
        self._burst_next = now

    def stop_burst(self) -> None:
        self.burst_dir = None

    def burst_due(self, now: float, interval: float = config.EVAL_BURST_SEC) -> bool:
        """연속 촬영 중이고 저장할 때가 되었는지. True 면 다음 저장 시각을 interval 뒤로"""
        if self.burst_dir is None or now < self._burst_next:
            return False
        self._burst_next = now + interval
        return True

    @property
    def model(self) -> Model:
        if self.name not in self._models:
            self._models[self.name] = self._create(self.name)
        return self._models[self.name]

    def select(self, name: str, now: float) -> None:
        if name not in NAMES:
            raise ValueError(f"알 수 없는 모델: {name}")
        self.name = name
        self.changed_at = now

    def toggle(self, item: str, now: float) -> None:
        if item not in ITEMS + EXTRAS:
            raise ValueError(f"알 수 없는 판정 항목: {item}")
        self.items ^= {item}
        self.changed_at = now

    def request(self) -> dict[str, bool]:
        return {item: item in self.items for item in ITEMS}

    def save_shot(self, frame, pred: Prediction | None, root: Path = config.SAVE_DIR, now=datetime.now,
                  folder: Path | None = None) -> Path:
        """프레임과 판정 결과 저장 -> jpg 경로. folder 가 없으면 {root}/eval"""
        out = Path(folder) if folder is not None else Path(root) / "eval"
        out.mkdir(parents=True, exist_ok=True)
        stem = now().strftime("%Y%m%d_%H%M%S_%f")
        path = out / f"{stem}.jpg"
        cv2.imwrite(str(path), frame)
        record = {"model": self.name, "request": sorted(self.items), "prediction": None}
        if pred is not None:
            record["prediction"] = {
                "elapsed_ms": round(pred.elapsed_ms, 3),
                "timings": {k: round(v, 3) for k, v in pred.timings.items()},
                "faces": [{"bbox": list(f.bbox), "visible": f.visible, "smile": f.smile, "eyes_open": f.eyes_open,
                           "scores": {k: [round(float(s), 4), th, bool(ok)] for k, (s, th, ok) in f.scores.items()}}
                          for f in pred.faces],
                "hands": None if pred.hands is None else [{"bbox": list(h.bbox), "raised": h.raised, "pose": h.pose}
                                                           for h in pred.hands],
                "hand_raised": pred.hand_raised,
            }
        (out / f"{stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        return path
