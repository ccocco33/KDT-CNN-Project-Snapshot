"""화면 상태와 전환
- 카메라, 화면과 분리한 순수 로직
- 시계(now) 주입: 테스트에서 시간 흉내용
- 입력: 버튼 클릭(click), 판정 결과(on_prediction), 시간 경과(tick), 사진 확인 완료(checked), 카메라 끊김(camera_lost)
- 카메라 끊김: PREPARE, CAPTURE 중이면 촬영을 취소하고 PREVIEW 로 (끊긴 동안 PREVIEW 에서 안내). 그 외 상태는 그대로
- 안내 표시 안정화 (PREVIEW): 판정 1회의 오판으로 안내가 깜빡이거나 촬영 시작이 막히지 않도록
  - 안내 대상이 GUIDE_ON_COUNT 회 연속이면 안내 표시 시작, 없음이 GUIDE_OFF_COUNT 회 연속이면 끝
  - 안내 표시 중에는 손을 들어도 PREPARE 로 가지 않음
  - 얼굴 추적 없이 프레임 단위로 셈. 안내 박스는 가장 최근 판정의 안내 대상 얼굴
- 전환 안정화: 판정 1회의 오판으로 촬영이 시작되거나 촬영 단계로 넘어가지 않도록
  - PREVIEW -> PREPARE: 손 들기 (시작 가능) 가 HAND_ON_COUNT 회 연속
  - PREPARE -> CAPTURE: 모두 웃음이 SMILE_ON_COUNT 회 연속
  - 얼굴 추적 없이 프레임 단위로 셈
- 모델 평가 (MODEL_EVAL, dev 모드 전용): HOME 의 eval 버튼으로 들어가고 home 버튼으로 돌아감
  - 모델, 판정 항목 선택과 찍기는 앱(app.py)이 처리. 판정 결과로 상태가 바뀌지 않음
"""
from __future__ import annotations

import time
from enum import Enum, auto

from . import config
from .model import Face, Prediction


class State(Enum):
    HOME = auto()
    PREVIEW = auto()
    PREPARE = auto()
    CAPTURE = auto()
    CHECKING = auto()
    RESULT = auto()
    MODEL_EVAL = auto()   # dev 모드 모델 평가 화면


class Machine:
    def __init__(self, now=time.monotonic):
        self._now = now
        self.state = State.HOME
        self._entered = now()
        self._cancel_until = None   # 취소 문구 표시 종료 시각
        self._streak = 0            # 전환 조건이 이어진 판정 횟수
        self._reset_guide()

    def click(self, button_id: str | None) -> None:
        s = self.state
        if s is State.HOME and button_id == "start":
            self._go(State.PREVIEW)
        elif s is State.HOME and button_id == "eval":
            self._go(State.MODEL_EVAL)
        elif s is State.MODEL_EVAL and button_id == "home":
            self._go(State.HOME)
        elif s is State.PREVIEW and button_id == "home":
            self._go(State.HOME)
        elif s is State.RESULT and button_id == "preview":
            self._go(State.PREVIEW)
        elif s is State.RESULT and button_id == "home":
            self._go(State.HOME)

    def on_prediction(self, pred: Prediction) -> None:
        s = self.state
        if s is State.PREVIEW:
            self._update_guide(pred.guidance(config.MIN_FACE_H))
            ready = not self.guiding and pred.ready_to_start(config.MIN_FACE_H)
            if self._count(ready) >= config.HAND_ON_COUNT:
                self._cancel_until = None
                self._go(State.PREPARE)
        elif s is State.PREPARE:
            if self._count(bool(pred.faces) and all(f.smile for f in pred.faces)) >= config.SMILE_ON_COUNT:
                self._go(State.CAPTURE)

    def tick(self) -> None:
        s = self.state
        if s is State.PREPARE and self._elapsed() >= config.PREPARE_SEC:
            self._cancel_until = self._now() + config.CANCEL_MSG_SEC
            self._go(State.PREVIEW)
        elif s is State.CAPTURE and self._elapsed() >= config.CAPTURE_SEC:
            self._go(State.CHECKING)

    def camera_lost(self) -> None:
        if self.state in (State.PREPARE, State.CAPTURE):
            self._cancel_until = None
            self._go(State.PREVIEW)

    def checked(self) -> None:
        if self.state is State.CHECKING:
            self._go(State.RESULT)

    def remaining(self) -> float:
        """PREPARE, CAPTURE 의 남은 시간(초). 그 외 상태는 0"""
        limit = {State.PREPARE: config.PREPARE_SEC, State.CAPTURE: config.CAPTURE_SEC}.get(self.state)
        return max(limit - self._elapsed(), 0.0) if limit else 0.0

    @property
    def guide_faces(self) -> list[tuple[Face, str]]:
        """화면에 안내할 얼굴과 이유. 안내 표시 중이 아니면 빈 목록"""
        return self._guides if self.guiding else []

    @property
    def show_cancel_msg(self) -> bool:
        return self.state is State.PREVIEW and self._cancel_until is not None and self._now() < self._cancel_until

    def _go(self, state: State) -> None:
        self.state = state
        self._entered = self._now()
        self._streak = 0
        self._reset_guide()

    def _reset_guide(self) -> None:
        self.guiding = False      # 안내 표시 중 여부
        self._guides = []         # 가장 최근 판정의 안내 대상 [(얼굴, 이유), ...]
        self._on_count = 0        # 안내 대상이 있는 판정의 연속 횟수
        self._off_count = 0       # 안내 대상이 없는 판정의 연속 횟수

    def _update_guide(self, guides: list[tuple[Face, str]]) -> None:
        self._guides = guides
        if guides:
            self._on_count, self._off_count = self._on_count + 1, 0
            if self._on_count >= config.GUIDE_ON_COUNT:
                self.guiding = True
        else:
            self._on_count, self._off_count = 0, self._off_count + 1
            if self._off_count >= config.GUIDE_OFF_COUNT:
                self.guiding = False

    def _count(self, ok: bool) -> int:
        """전환 조건 연속 횟수. 조건이 끊기면 0"""
        self._streak = self._streak + 1 if ok else 0
        return self._streak

    def _elapsed(self) -> float:
        return self._now() - self._entered
