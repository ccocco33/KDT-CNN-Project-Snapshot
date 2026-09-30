import json
from datetime import datetime

import pytest

from photo_app import metrics
from photo_app.model import Face, Prediction, requested

NOW = lambda: datetime(2026, 9, 24, 12, 0, 0)


@pytest.fixture
def log_path(tmp_path):
    path = metrics.start(tmp_path, now=NOW)
    yield path
    metrics.stop()


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_log_file_name(log_path, tmp_path):
    """실행마다 시각 이름의 jsonl 파일 생성"""
    # given: log_path

    # when
    name = log_path.name

    # then
    assert log_path == tmp_path / "20260924_120000.jsonl"


def test_log_writes_one_json_per_line(log_path):
    """이벤트 1개가 JSON 한 줄로 기록"""
    # given
    metrics.log("capture", frames=40, fps=20.0)

    # when
    rows = read(log_path)

    # then
    assert len(rows) == 1
    assert rows[0]["ev"] == "capture" and rows[0]["frames"] == 40 and "t" in rows[0]


def test_log_ignored_before_start(tmp_path):
    """start 전에는 기록하지 않음"""
    # given
    metrics.stop()

    # when
    metrics.log("capture", frames=1)

    # then
    assert list(tmp_path.iterdir()) == []


def test_log_prediction_fields(log_path):
    """predict 이벤트에 요청, 인원, 단계별 시간 기록"""
    # given
    pred = Prediction([Face((0, 0, 1, 1), True, None)], None, None, 12.5,
                      requested(True, False, False, False), {"face": 10.0, "smile": 0.01})

    # when
    metrics.log_prediction(pred, caller="agent", state="PREPARE")

    # then
    row = read(log_path)[0]
    assert row["ev"] == "predict" and row["caller"] == "agent" and row["state"] == "PREPARE"
    assert (row["smile"], row["eye"], row["hand"], row["visibility"]) == (True, False, False, False)
    assert row["faces"] == 1 and row["hands"] is None
    assert row["total_ms"] == 12.5 and row["face_ms"] == 10.0 and row["smile_ms"] == 0.01


def test_log_prediction_guided(log_path):
    """visibility 를 요청한 판정은 안내 대상 수와 이유별 수 기록, 아니면 None. 얼굴 높이는 항상 기록"""
    # given
    faces = [Face((0, 0, 60, 60), None, None, True), Face((0, 0, 60, 60), None, None, False), Face((0, 0, 20, 20), None, None, True)]
    with_vis = Prediction(faces, [], False, 1.0, requested(False, False, True, True))
    without_vis = Prediction(faces, None, None, 1.0, requested(True, False, False, False))

    # when
    metrics.log_prediction(with_vis, caller="agent", state="PREVIEW")
    metrics.log_prediction(without_vis, caller="agent", state="PREPARE")

    # then
    rows = read(log_path)
    assert (rows[0]["guided"], rows[0]["guided_not_visible"], rows[0]["guided_too_small"]) == (2, 1, 1)
    assert rows[0]["face_h"] == [60, 60, 20]
    assert (rows[1]["guided"], rows[1]["guided_not_visible"], rows[1]["guided_too_small"]) == (None, None, None)
