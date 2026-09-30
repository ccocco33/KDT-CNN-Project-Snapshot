"""메인 루프
- 카메라 스레드, 판정 스레드, states, ui 연결
- 종료: q 또는 ESC, 창 닫기
- 모델 평가 (MODEL_EVAL, dev 모드): 고른 모델과 판정 항목으로 판정 스레드를 돌림 (model_eval.ModelEval)
  - 모델, 항목 버튼, 찍기는 여기서 처리. 나가면 판정 스레드의 모델을 앱 모델로 되돌림
"""
from __future__ import annotations

import threading
import time

import cv2

from . import config, metrics
from .camera import Camera
from .model import Model, Prediction, requested
from .model_eval import ModelEval
from .models import create_model
from .selection import EyeHold, select, select_closest
from .states import Machine, State
from .storage import Session
from .ui import View, button_at, render

# 판정 스레드가 도는 상태와 요청할 결과
AGENT_REQUESTS = {
    State.PREVIEW: dict(smile=False, eye=False, hand=True, visibility=True),
    State.PREPARE: dict(smile=True, eye=False, hand=False, visibility=False),
}
CHECK_REQUEST = dict(smile=True, eye=True, hand=False, visibility=True)   # 보이지 않는 얼굴은 조건 미충족
DEV_CAPTURE_REQUEST = CHECK_REQUEST   # dev 모드 CAPTURE: 화면에 판정 점수 표시용 (CHECKING 과 같은 항목)


def request_for(state: State, ev: ModelEval, dev: bool = False) -> dict[str, bool] | None:
    """상태별 판정 스레드 요청. MODEL_EVAL 은 평가 화면에서 켠 항목. 판정하지 않는 상태는 None
    - dev 모드 CAPTURE 는 표시용 판정 (DEV_CAPTURE_REQUEST). prod 는 촬영 FPS 를 위해 판정하지 않음
    """
    if state is State.MODEL_EVAL:
        return ev.request()
    if dev and state is State.CAPTURE:
        return DEV_CAPTURE_REQUEST
    return AGENT_REQUESTS.get(state)


def agent_interval(state: State) -> float:
    """판정 스레드의 최소 판정 간격(초). CAPTURE 는 촬영 FPS 를 덜 빼앗도록 DEV_CAPTURE_PREDICT_SEC"""
    return config.DEV_CAPTURE_PREDICT_SEC if state is State.CAPTURE else 0.0


def accepts(req: dict[str, bool] | None, pred: Prediction) -> bool:
    """판정 결과가 현재 요청으로 만든 것인지
    - 상태가 바뀌는 사이에 끝난 이전 상태의 판정(예: PREPARE -> PREVIEW 직후 도착한 PREPARE 판정)은 버림
    - 모델 평가에서 항목을 바꾼 직후 도착한 이전 항목의 판정도 버림
    """
    return req is not None and pred.requested == requested(**req)


class Agent:
    """
    - 판정을 담당하는 agent
    - 최신 프레임만 가져와 Model.predict 호출
    - request 가 있을 때만 동작
    - interval: 판정 시작 사이 최소 간격(초). 0 이면 쉬지 않고 판정
    """

    def __init__(self, camera: Camera, model: Model):
        self._camera = camera
        self.model = model    # 판정에 쓰는 모델 (모델 평가 화면에서 바뀜)
        self._lock = threading.Lock()
        self._latest = (None, None, None)   # (프레임 ts, Prediction, 판정한 프레임)
        self.request = None   # predict 요청 인자. None 이면 정지
        self.interval = 0.0   # 판정 시작 사이 최소 간격(초)
        self.state_name = ""  # 로그용 현재 상태 이름
        self._running = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join(timeout=1)

    def latest(self):
        with self._lock:
            return self._latest

    def _run(self):
        last_ts, last_start = None, 0.0
        while self._running:
            frame, ts = self._camera.latest()
            request = self.request
            if request is None or frame is None or ts == last_ts or time.monotonic() - last_start < self.interval:
                time.sleep(0.005)
                continue
            last_start = time.monotonic()
            pred = self.model.predict(frame, **request)
            metrics.log_prediction(pred, caller="agent", state=self.state_name)
            last_ts = ts
            with self._lock:
                self._latest = (ts, pred, frame)


def check(frames: list, model: Model, session: Session):
    """CHECKING 처리
    - 조건 충족 사진 중 MAX_RESULT 장 선택 (구간마다 눈 뜸 점수가 가장 높은 사진. selection.select)
    - 조건 충족 사진이 없으면 조건에 가장 가까운 사진 MAX_RESULT 장 (selection.select_closest)
    - 고른 사진만 저장. 파일 이름 번호는 촬영 순서 (frame_{번호}.jpg)
    - EYE_HOLD 이면 촬영 시각 순서로 눈 감음 허용 규칙 적용 (selection.EyeHold). 로그의 predict 이벤트는 원래 판정
    - 반환: ([(경로, 프레임), ...], 가까운 사진으로 대신했는지)
    """
    start = time.perf_counter()
    images = [f for _, f in frames]
    hold = EyeHold() if config.EYE_HOLD else None
    ok, cands = [], []
    for i, (ts, img) in enumerate(frames):
        pred = model.predict(img, **CHECK_REQUEST)
        metrics.log_prediction(pred, caller="check", state=State.CHECKING.name)
        if hold is not None:
            pred = hold.apply(ts, pred)
        cands.append(((i, img), pred))
        if pred.meets_condition():
            ok.append(((i, img), pred))
    t_predicted = time.perf_counter()
    fallback = not ok
    chosen = select_closest(cands, config.MAX_RESULT) if fallback else select(ok, config.MAX_RESULT)
    t_selected = time.perf_counter()
    picked_images = [img for _, img in chosen]
    paths = session.save(picked_images, numbers=[i for i, _ in chosen])
    end = time.perf_counter()
    metrics.log(
        "checking",
        frames=len(images), ok=len(ok), picked=len(chosen), fallback=fallback,
        eye_hold_allowed=None if hold is None else hold.allowed,
        predict_ms=round((t_predicted - start) * 1000, 3),
        select_ms=round((t_selected - t_predicted) * 1000, 3),
        save_ms=round((end - t_selected) * 1000, 3),
        total_ms=round((end - start) * 1000, 3),
    )
    return list(zip(paths, picked_images)), fallback


def log_capture(frames: list) -> None:
    """capture 이벤트: 촬영 장수, 첫 프레임 ~ 마지막 프레임 시간, FPS"""
    duration = frames[-1][0] - frames[0][0] if len(frames) > 1 else 0.0
    fps = (len(frames) - 1) / duration if duration > 0 else 0.0
    metrics.log("capture", frames=len(frames), duration_s=round(duration, 3), fps=round(fps, 2))


class ScreenStats:
    """화면 1장 단계별 시간 모음 -> 1초마다 screen_fps 이벤트
    - render: 화면 그리기, imshow: cv2.imshow 호출 (비동기. 이미지 복사만)
    - waitkey: cv2.waitKey 전체. X 로 화면 전송이 여기서 일어남 (X 포워딩이면 전송이 끝날 때까지 막힘)
    - waitkey_extra: waitKey 가 요청한 대기(wait_ms)보다 더 걸린 시간 = 화면 전송으로 막힌 시간
    """

    def __init__(self, start: float):
        self._window = start
        self._rows = []   # [(render, imshow, waitkey, waitkey_extra), ...] (ms)

    def add(self, render_ms: float, imshow_ms: float, waitkey_ms: float, wait_ms: int) -> None:
        self._rows.append((render_ms, imshow_ms, waitkey_ms, max(0.0, waitkey_ms - wait_ms)))

    def flush(self, now: float, state: str) -> None:
        """1초가 지났으면 기록하고 새로 모음"""
        if now - self._window < 1.0:
            return
        n = len(self._rows)
        avg = [round(sum(col) / n, 2) for col in zip(*self._rows)] if n else [None] * 4
        metrics.log(
            "screen_fps", fps=round(n / (now - self._window), 2), state=state,
            render_ms=avg[0], imshow_ms=avg[1], waitkey_ms=avg[2], waitkey_extra_ms=avg[3],
            waitkey_extra_max_ms=round(max(r[3] for r in self._rows), 2) if n else None,
        )
        self._rows, self._window = [], now


def main():
    camera = Camera()
    model = create_model(config.MODEL)
    agent = Agent(camera, model)
    machine = Machine()
    clicks = []
    session = None
    picked = []          # 결과 [(경로, 프레임), ...]
    closest = False      # 결과가 조건에 가장 가까운 사진인지 (조건 충족 사진이 없을 때)
    last_pred_ts = None
    pred = None
    pred_frame = None                     # pred 를 판정한 프레임 (모델 평가 찍기용)
    ev = ModelEval(config.MODEL, model)   # 모델 평가 화면 선택 상태
    eval_msg = ("", 0.0)                  # 모델 평가 화면 알림 (문구, 표시 끝 시각)
    eval_hold = EyeHold()                 # 모델 평가 화면 eye_hold: 판정마다 카메라 시각으로 적용. 항목, 모델을 바꾸면 새로

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            clicks.append((x, y))

    print("log ->", metrics.start())
    metrics.log(
        "run", mode=config.MODE, model=config.MODEL,
        screen=f"{config.SCREEN_W}x{config.SCREEN_H}", camera=f"{config.CAMERA_W}x{config.CAMERA_H}",
        camera_actual="{}x{}".format(*camera.size),
    )
    dev_label = f"dev | model: {config.MODEL}"
    dev = config.DEV   # 실행 중 모드 (d 키로 전환). dev: 얼굴 점수, 손 박스, 모델 이름, 모델 평가 버튼, 키보드 흉내
    screen = ScreenStats(time.monotonic())
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(config.WINDOW_NAME, on_mouse)
    camera.start()
    agent.start()
    try:
        while True:
            loop_start = time.monotonic()
            prev = machine.state

            # 입력 -> 상태 전환
            while clicks:
                x, y = clicks.pop(0)
                bid = button_at(machine.state, x, y, dev=dev)
                if machine.state is State.MODEL_EVAL and bid and bid.split(":")[0] in ("model", "item", "shot", "burst"):
                    kind, _, arg = bid.partition(":")
                    if kind == "shot":
                        if pred_frame is not None:
                            path = ev.save_shot(pred_frame, pred)
                            metrics.log("eval_shot", model=ev.name, path=str(path))
                            eval_msg = (f"저장: {path.name}", time.monotonic() + 2.0)
                        continue
                    if kind == "burst":
                        if ev.bursting:
                            eval_msg = (f"연속 촬영 끝: {ev.burst_count}장 ({ev.burst_dir.name})", time.monotonic() + 3.0)
                            ev.stop_burst()
                        else:
                            ev.start_burst(time.monotonic())
                        continue
                    if kind == "model":
                        ev.select(arg, time.monotonic())
                        agent.model = ev.model   # 처음 고르는 모델은 여기서 만듦 (잠시 멈출 수 있음)
                    else:
                        ev.toggle(arg, time.monotonic())
                    agent.request = ev.request()
                    pred = None
                    eval_hold = EyeHold()
                    continue
                machine.click(bid)
            
            # agent의 최신 프레임을 기준으로 측정
            ts, p, p_frame = agent.latest()
            fresh = machine.state is not State.MODEL_EVAL or (ts is not None and ts > ev.changed_at)
            if ts is not None and ts != last_pred_ts and fresh and accepts(request_for(machine.state, ev, dev), p):
                last_pred_ts, pred, pred_frame = ts, p, p_frame
                if machine.state is State.MODEL_EVAL and "eye_hold" in ev.items:
                    pred = eval_hold.apply(ts, pred)
                machine.on_prediction(pred)
                if machine.state is State.MODEL_EVAL and ev.burst_due(time.monotonic()):   # 연속 촬영: 새 판정이 왔을 때만
                    path = ev.save_shot(pred_frame, pred, folder=ev.burst_dir)
                    ev.burst_count += 1
                    metrics.log("eval_shot", model=ev.name, path=str(path), burst=True)
            if camera.lost:
                machine.camera_lost()
            machine.tick()

            # 상태가 바뀐 경우의 처리
            state = machine.state
            if state is not prev:
                metrics.log("state", **{"from": prev.name, "to": state.name})
                agent.state_name = state.name
                if state is State.MODEL_EVAL:
                    ev.changed_at = time.monotonic()
                    agent.model = ev.model
                    eval_hold = EyeHold()
                elif prev is State.MODEL_EVAL:
                    agent.model = model
                    ev.stop_burst()
                agent.request = request_for(state, ev, dev)
                agent.interval = agent_interval(state)
                if prev is State.RESULT and session is not None:
                    session.keep_only([p for p, _ in picked])
                    session, picked, closest = None, [], False
                if state is State.CAPTURE:
                    camera.start_recording()
                if prev is State.CAPTURE and state is not State.CHECKING:   # 촬영 취소 (카메라 끊김): 누적 버림
                    camera.stop_recording()
                if state is State.CHECKING:
                    frames = camera.stop_recording()
                    log_capture(frames)
                if request_for(state, ev, dev) is None or state is State.MODEL_EVAL or prev is State.MODEL_EVAL:
                    pred = None

            # 화면
            frame, _ = camera.latest()
            view = View(
                frame=frame,
                faces=pred.faces if pred else [],
                remaining_sec=machine.remaining(),
                show_cancel_msg=machine.show_cancel_msg,
                results=[img for _, img in picked],
                results_closest=closest,
                camera_lost=camera.lost,
                hands=pred.hands if pred and pred.hands else [],
                guides=machine.guide_faces,
                dev=dev,
                dev_label=dev_label,
                eval_model=ev.name,
                eval_items=frozenset(ev.items),
                eval_timings=({"total": pred.elapsed_ms, **pred.timings} if pred else {}),
                eval_msg=(f"연속 촬영 중: {ev.burst_count}장" if ev.bursting
                          else eval_msg[0] if time.monotonic() < eval_msg[1] else ""),
                eval_burst=ev.bursting,
            )
            t_render = time.perf_counter()
            img = render(state, view)
            t_imshow = time.perf_counter()
            cv2.imshow(config.WINDOW_NAME, img)
            t_shown = time.perf_counter()

            # CHECKING: "사진을 고르는 중" 화면을 띄운 뒤 처리
            if state is State.CHECKING:
                cv2.waitKey(1)
                session = Session()
                picked, closest = check(frames, model, session)
                machine.checked()

            # 키 입력, 종료
            wait_ms = max(1, int((1 / config.SCREEN_FPS - (time.monotonic() - loop_start)) * 1000))
            t_wait = time.perf_counter()
            key = cv2.waitKey(wait_ms) & 0xFF
            t_end = time.perf_counter()
            screen.add((t_imshow - t_render) * 1000, (t_shown - t_imshow) * 1000, (t_end - t_wait) * 1000, wait_ms)
            screen.flush(time.monotonic(), state.name)   # 1초마다 화면 FPS, 단계별 시간 기록
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                dev = not dev
                metrics.log("mode", mode="dev" if dev else "prod")
                agent.request = request_for(machine.state, ev, dev)   # CAPTURE 중 전환 시 표시용 판정 켜고 끔
                if agent.request is None:
                    pred = None
            handle_key = getattr(agent.model, "handle_key", None)   # 키보드 흉내용 (손 들기 등). dev 전용
            if dev and key != 0xFF and handle_key:
                handle_key(chr(key))
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if session is not None:   # 결과 화면에서 종료한 경우도 결과 사진만 남김
            session.keep_only([p for p, _ in picked])
        agent.stop()
        camera.stop()
        cv2.destroyAllWindows()
        metrics.stop()
