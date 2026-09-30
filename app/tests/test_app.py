import json
from datetime import datetime

import numpy as np
import pytest

from photo_app import config, metrics
from photo_app.app import AGENT_REQUESTS, DEV_CAPTURE_REQUEST, ScreenStats, accepts, agent_interval, check, request_for
from photo_app.model import Face, Prediction, requested
from photo_app.model_eval import ITEMS, ModelEval
from photo_app.states import State
from photo_app.storage import Session


def prediction(**req):
    return Prediction([], [], False, 0.0, requested(**req))


def make_eval():
    return ModelEval("fake", create=lambda name: f"model:{name}")


@pytest.mark.parametrize("state", [State.PREVIEW, State.PREPARE])
def test_accepts_prediction_made_for_current_state(state):
    """현재 상태의 요청으로 만든 판정은 받음"""
    # given
    pred = prediction(**AGENT_REQUESTS[state])

    # when
    ok = accepts(request_for(state, make_eval()), pred)

    # then
    assert ok


def test_rejects_prepare_prediction_after_returning_to_preview():
    """PREPARE 요청으로 만든 판정이 PREVIEW 로 돌아간 뒤 도착하면 버림"""
    # given
    pred = prediction(**AGENT_REQUESTS[State.PREPARE])

    # when
    ok = accepts(request_for(State.PREVIEW, make_eval()), pred)

    # then
    assert not ok


def test_rejects_prediction_in_state_without_agent():
    """판정 스레드가 쉬는 상태(CAPTURE)에서는 받지 않음"""
    # given
    pred = prediction(**AGENT_REQUESTS[State.PREPARE])

    # when
    ok = accepts(request_for(State.CAPTURE, make_eval()), pred)

    # then
    assert not ok


@pytest.mark.parametrize("dev, expected", [(True, DEV_CAPTURE_REQUEST), (False, None)])
def test_capture_request_only_in_dev(dev, expected):
    """CAPTURE 판정 요청은 dev 모드에서만 (화면 표시용)"""
    # given
    ev = make_eval()

    # when
    req = request_for(State.CAPTURE, ev, dev)

    # then
    assert req == expected


@pytest.mark.parametrize("state, expected", [(State.CAPTURE, config.DEV_CAPTURE_PREDICT_SEC), (State.PREVIEW, 0.0)])
def test_agent_interval_limits_capture(state, expected):
    """CAPTURE 만 판정 간격을 둠"""
    # when
    interval = agent_interval(state)

    # then
    assert interval == expected


def test_model_eval_request_follows_toggled_items():
    """모델 평가 요청은 켠 항목만. 항목을 끄면 그 항목 요청이 빠짐"""
    # given
    ev = make_eval()

    # when
    ev.toggle("hand", now=1.0)

    # then
    assert request_for(State.MODEL_EVAL, ev) == dict(smile=True, eye=True, hand=False, visibility=True)
    assert ev.changed_at == 1.0


def test_model_eval_rejects_prediction_of_previous_items():
    """모델 평가에서 항목을 바꾸기 전 요청으로 만든 판정은 버림"""
    # given
    ev = make_eval()
    old = prediction(**ev.request())
    ev.toggle("eye", now=1.0)

    # when
    ok = accepts(request_for(State.MODEL_EVAL, ev), old)

    # then
    assert not ok


def test_model_eval_creates_model_once():
    """모델은 처음 고를 때 만들고 다시 고르면 같은 것을 씀"""
    # given
    made = []
    ev = ModelEval("fake", create=lambda name: made.append(name) or f"model:{name}")

    # when
    ev.select("yunet_cnn", now=1.0)
    first = ev.model
    ev.select("fake", now=2.0)
    ev.select("yunet_cnn", now=3.0)
    second = ev.model

    # then
    assert first == second == "model:yunet_cnn"
    assert made.count("yunet_cnn") == 1


@pytest.mark.parametrize("bad", ["select", "toggle"])
def test_model_eval_rejects_unknown_names(bad):
    """모르는 모델, 판정 항목은 오류"""
    # given
    ev = make_eval()

    # when / then
    with pytest.raises(ValueError):
        getattr(ev, bad)("no_such", now=1.0)


def test_model_eval_save_shot_writes_image_and_scores(tmp_path):
    """찍기: 프레임 jpg 와 판정 결과 json 저장 (점수, 기준값, 통과 포함)"""
    # given
    ev = make_eval()
    face = Face((10, 20, 30, 40), True, None, True, scores={"smile": (0.62, 0.5, True)})
    pred = Prediction([face], None, None, 12.3, requested(**ev.request()), {"face": 5.0})
    frame = np.zeros((48, 64, 3), np.uint8)

    # when
    path = ev.save_shot(frame, pred, root=tmp_path, now=lambda: datetime(2026, 9, 27, 1, 2, 3))

    # then
    record = json.loads(path.with_suffix(".json").read_text())
    assert path.exists() and path.parent == tmp_path / "eval"
    assert record["model"] == "fake" and record["request"] == sorted(ITEMS)
    assert record["prediction"]["faces"][0]["scores"]["smile"] == [0.62, 0.5, True]


def test_model_eval_burst_saves_at_interval(tmp_path):
    """연속 촬영: 켠 직후 1장, 그 뒤 간격이 지나야 다음 장. 끄면 저장 안 함"""
    # given
    ev = make_eval()
    ev.start_burst(now=10.0, root=tmp_path, clock=lambda: datetime(2026, 9, 27, 1, 2, 3))

    # when
    due = [ev.burst_due(t, interval=0.5) for t in (10.0, 10.2, 10.5, 10.6)]
    ev.stop_burst()
    after_stop = ev.burst_due(20.0, interval=0.5)

    # then
    assert due == [True, False, True, False]
    assert not after_stop
    assert not ev.bursting


def test_model_eval_burst_saves_into_burst_folder(tmp_path):
    """연속 촬영 사진은 켤 때 만든 burst 폴더에 저장"""
    # given
    ev = make_eval()
    ev.start_burst(now=0.0, root=tmp_path, clock=lambda: datetime(2026, 9, 27, 1, 2, 3))

    # when
    path = ev.save_shot(np.zeros((8, 8, 3), np.uint8), None, folder=ev.burst_dir)

    # then
    assert path.parent == tmp_path / "eval" / "burst_20260927_010203"


class ScriptedModel:
    """프레임 번호마다 정한 판정 결과를 돌려주는 모델 (프레임 = 번호가 적힌 배열)"""

    def __init__(self, faces_by_frame):
        self.faces_by_frame = faces_by_frame

    def predict(self, frame, **req):
        return Prediction(self.faces_by_frame[int(frame[0, 0, 0])], None, None, 0.0, requested(**req))


def frames(n):
    return [(float(i), np.full((8, 8, 3), i, np.uint8)) for i in range(n)]


def eye_face(e):
    return Face((0, 0, 4, 4), True, e >= 0.84, True, e, None, {"smile": (0.9, 0.31, True), "eye": (e, 0.84, e >= 0.84)})


def test_check_uses_closest_when_no_photo_meets_condition(tmp_path):
    """조건 충족 사진이 없으면 가장 가까운 사진으로 대신하고 표시"""
    # given
    model = ScriptedModel([[eye_face(0.3)], [eye_face(0.7)], [eye_face(0.5)]])

    # when
    picked, closest = check(frames(3), model, Session(tmp_path))

    # then
    assert closest and len(picked) == 3


def test_check_uses_ok_photos_when_any_meets_condition(tmp_path):
    """조건 충족 사진이 있으면 그 사진만 (가까운 사진으로 대신하지 않음)"""
    # given
    model = ScriptedModel([[eye_face(0.3)], [eye_face(0.9)], [eye_face(0.5)]])

    # when
    picked, closest = check(frames(3), model, Session(tmp_path))

    # then
    assert not closest and [p.name for p, _ in picked] == ["frame_001.jpg"]


def test_check_saves_only_picked_photos(tmp_path):
    """고른 사진만 저장 (파일 이름 번호는 촬영 순서)"""
    # given
    session = Session(tmp_path)
    model = ScriptedModel([[eye_face(0.3)], [eye_face(0.9)], [eye_face(0.5)]])

    # when
    check(frames(3), model, session)

    # then
    assert [p.name for p in session.dir.iterdir()] == ["frame_001.jpg"]


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_check_logs_checking_event(tmp_path):
    """checking 이벤트: 장수, 조건 충족 수, 고른 수, 대신했는지, 단계별 시간"""
    # given
    path = metrics.start(tmp_path / "logs", now=lambda: datetime(2026, 9, 28, 1, 2, 3))
    model = ScriptedModel([[eye_face(0.3)], [eye_face(0.9)], [eye_face(0.5)]])

    # when
    check(frames(3), model, Session(tmp_path / "photos"))
    metrics.stop()

    # then
    e = [e for e in read_events(path) if e["ev"] == "checking"][0]
    assert (e["frames"], e["ok"], e["picked"], e["fallback"]) == (3, 1, 1, False)
    assert {"predict_ms", "select_ms", "save_ms", "total_ms"} <= set(e)


def test_screen_stats_logs_stage_averages_every_second(tmp_path):
    """1초가 지나면 화면 FPS 와 단계별 평균 시간을 screen_fps 로 기록. waitkey_extra 는 요청한 대기를 넘은 시간"""
    # given
    path = metrics.start(tmp_path, now=lambda: datetime(2026, 9, 28, 1, 2, 3))
    screen = ScreenStats(start=0.0)
    screen.add(render_ms=4.0, imshow_ms=1.0, waitkey_ms=30.0, wait_ms=20)
    screen.add(render_ms=6.0, imshow_ms=1.0, waitkey_ms=110.0, wait_ms=10)

    # when
    screen.flush(now=0.5, state="PREVIEW")
    screen.flush(now=1.0, state="PREVIEW")
    metrics.stop()

    # then
    events = read_events(path)
    assert len(events) == 1
    e = events[0]
    assert (e["ev"], e["fps"], e["state"]) == ("screen_fps", 2.0, "PREVIEW")
    assert (e["render_ms"], e["imshow_ms"], e["waitkey_ms"]) == (5.0, 1.0, 70.0)
    assert (e["waitkey_extra_ms"], e["waitkey_extra_max_ms"]) == (55.0, 100.0)


def test_screen_stats_extra_not_negative_on_early_key(tmp_path):
    """키 입력으로 waitKey 가 요청한 대기보다 일찍 끝나면 waitkey_extra 는 0"""
    # given
    path = metrics.start(tmp_path, now=lambda: datetime(2026, 9, 28, 1, 2, 3))
    screen = ScreenStats(start=0.0)

    # when
    screen.add(render_ms=4.0, imshow_ms=1.0, waitkey_ms=5.0, wait_ms=20)
    screen.flush(now=1.0, state="HOME")
    metrics.stop()

    # then
    assert read_events(path)[0]["waitkey_extra_ms"] == 0.0


def closed_face():
    """웃고 눈 감은 얼굴 (높이 100px, 눈 점수 0.3)"""
    return Face((0, 0, 80, 100), True, False, True, 0.3, None, {"smile": (0.9, 0.31, True), "eye": (0.3, 0.84, False)})


def burst(n, step=0.1):
    """촬영 시각 step 초 간격 프레임 n장 (프레임 = 번호가 적힌 배열)"""
    return [(i * step, np.full((8, 8, 3), i, np.uint8)) for i in range(n)]


def test_check_eye_hold_allows_long_closed_eyes(tmp_path, monkeypatch):
    """EYE_HOLD 이면 0.3초 이상 눈 감은 프레임부터 조건 충족으로 고르고, 허용 수를 checking 에 기록"""
    # given
    monkeypatch.setattr(config, "EYE_HOLD", True)
    path = metrics.start(tmp_path / "logs", now=lambda: datetime(2026, 9, 28, 1, 2, 3))
    model = ScriptedModel([[closed_face()]] * 5)

    # when
    picked, closest = check(burst(5), model, Session(tmp_path / "photos"))
    metrics.stop()

    # then
    assert not closest and [p.name for p, _ in picked] == ["frame_003.jpg", "frame_004.jpg"]
    assert [e for e in read_events(path) if e["ev"] == "checking"][0]["eye_hold_allowed"] == 2


def test_check_without_eye_hold_keeps_closed_eyes_out(tmp_path, monkeypatch):
    """EYE_HOLD 가 꺼져 있으면 오래 감아도 조건 미충족 (가까운 사진으로 대신)"""
    # given
    monkeypatch.setattr(config, "EYE_HOLD", False)
    model = ScriptedModel([[closed_face()]] * 5)

    # when
    _, closest = check(burst(5), model, Session(tmp_path))

    # then
    assert closest


def test_model_eval_eye_hold_is_not_requested():
    """eye_hold 는 켜고 끌 수 있지만 predict 요청에는 넣지 않음. 처음에는 꺼짐"""
    # given
    ev = make_eval()
    before = "eye_hold" in ev.items

    # when
    ev.toggle("eye_hold", now=1.0)

    # then
    assert not before and "eye_hold" in ev.items
    assert set(ev.request()) == set(ITEMS)
