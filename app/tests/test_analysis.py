from datetime import datetime

import pytest

from analysis.logs import connect, run_reports
from photo_app import metrics
from photo_app.model import Face, Prediction, requested

NOW = lambda: datetime(2026, 9, 24, 12, 0, 0)


@pytest.fixture
def log_path(tmp_path):
    path = metrics.start(tmp_path, now=NOW)
    face = Face((0, 0, 1, 1), True, True)
    for ms in (10.0, 20.0, 30.0):
        metrics.log_prediction(Prediction([face], [], False, ms, requested(False, False, True, True), {"face": 5.0, "hand": ms - 5}),
                               caller="agent", state="PREVIEW")
    metrics.log_prediction(Prediction([face], None, None, 4.0, requested(True, True, False, True), {"face": 4.0}),
                           caller="check", state="CHECKING")
    metrics.log("capture", frames=40, duration_s=1.95, fps=20.0)
    metrics.log("screen_fps", fps=28.0, state="PREVIEW")
    metrics.log("screen_fps", fps=30.0, state="PREVIEW", render_ms=5.0, imshow_ms=0.5, waitkey_ms=60.0,
                waitkey_extra_ms=40.0, waitkey_extra_max_ms=90.0)   # 이전 로그(단계별 시간 없음)와 섞여도 동작
    metrics.log("checking", frames=40, ok=10, picked=5, save_ms=100.0, predict_ms=160.0, select_ms=0.01, total_ms=260.01)
    metrics.stop()
    return path


def test_predict_report_groups_by_request(log_path):
    """추론 시간 보고서는 호출 위치, 요청 조합별로 묶음"""
    # given
    con = connect(str(log_path))

    # when
    rows = dict(run_reports(con))["추론 시간: 요청 조합별 (ms)"].fetchall()

    # then
    agent = [r for r in rows if r[0] == "agent"][0]
    assert agent[1:6] == (False, False, True, True, 3)   # smile, eye, hand, visibility, n
    assert agent[6] == 20.0                              # avg


def test_checking_report(log_path):
    """판정 보고서에 저장, 추론, 전체 시간"""
    # given
    con = connect(str(log_path))

    # when
    row = dict(run_reports(con))["판정 (checking, ms)"].fetchone()

    # then
    assert row[0] == 1                 # n
    assert row[4:6] == (100.0, 160.0)  # save, predict


def test_report_skipped_without_columns(log_path):
    """해당 이벤트가 없어 컬럼이 없으면 보고서 결과 None"""
    # given
    con = connect(str(log_path))

    # when
    results = dict(run_reports(con))

    # then
    assert results["실행 목록"] is None
    assert results["상태 전환 횟수"] is None


def test_screen_fps_report_by_state(log_path):
    """화면 FPS 보고서는 상태별 평균"""
    # given
    con = connect(str(log_path))

    # when
    rows = dict(run_reports(con))["화면 FPS: 상태별"].fetchall()

    # then
    assert rows == [("PREVIEW", 2, 29.0, 28.0, 30.0)]


def test_screen_time_report_skips_logs_without_stage_times(log_path):
    """화면 1장 시간 보고서는 단계별 시간이 있는 screen_fps 만 상태별로 묶음"""
    # given
    con = connect(str(log_path))

    # when
    rows = dict(run_reports(con))["화면 1장 시간: 상태별 (ms)"].fetchall()

    # then
    assert rows == [("PREVIEW", 1, 5.0, 0.5, 60.0, 40.0, 90.0)]
