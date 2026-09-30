"""시간 기록 로그 분석
- 입력: photo_app 이 남긴 JSON Lines (logs/*.jsonl)
- duckdb 테이블 이름: events (모든 이벤트를 한 테이블로. 없는 필드는 NULL, filename 컬럼으로 실행 구분)
- 요약 보고서: REPORTS 의 (제목, 필요한 컬럼, SQL). SQL 은 문자열 또는 로그 컬럼을 받아 문자열을 만드는 함수
"""
from __future__ import annotations

from pathlib import Path

import duckdb

DEFAULT_GLOB = str(Path(__file__).resolve().parents[1] / "logs" / "*.jsonl")

STAGES = ("face", "landmark", "hand", "smile", "eye")   # 모델 단계별 시간 컬럼: {단계}_ms


def _stage_sql(cols: set[str]) -> str:
    """로그에 있는 단계 컬럼만으로 단계별 평균 SQL 생성"""
    stages = ", ".join(f"round(avg({s}_ms), 4) as {s}" for s in STAGES if f"{s}_ms" in cols)
    return f"""
        select caller, count(*) as n, {stages}, round(avg(total_ms), 2) as total
        from events where ev = 'predict'
        group by all order by caller
        """


# (제목, 필요한 컬럼, SQL). 필요한 컬럼이 로그에 없으면 건너뜀
def _request_sql(cols: set[str]) -> str:
    """요청 조합별 추론 시간 SQL. visibility 컬럼은 로그에 있을 때만 (이전 로그 호환)"""
    keys = "smile, eye, hand" + (", visibility" if "visibility" in cols else "")
    return f"""
        select caller, {keys}, count(*) as n,
               round(avg(total_ms), 2) as avg,
               round(quantile_cont(total_ms, 0.5), 2) as p50,
               round(quantile_cont(total_ms, 0.95), 2) as p95,
               round(max(total_ms), 2) as max
        from events where ev = 'predict'
        group by all order by caller, {keys}
        """


REPORTS = [
    (
        "추론 시간: 요청 조합별 (ms)",
        {"smile", "eye", "hand", "total_ms"},
        _request_sql,
    ),
    (
        "추론 시간: 단계별 평균 (ms)",
        {"face_ms"},
        _stage_sql,
    ),
    (
        "추론 시간: 얼굴 수별 (ms)",
        {"faces", "total_ms"},
        """
        select faces, count(*) as n,
               round(avg(total_ms), 2) as avg,
               round(quantile_cont(total_ms, 0.95), 2) as p95
        from events where ev = 'predict'
        group by all order by faces
        """,
    ),
    (
        "촬영 (capture)",
        {"frames", "duration_s", "fps"},
        """
        select count(*) as n,
               round(avg(frames), 1) as avg_frames, min(frames) as min_frames,
               round(avg(duration_s), 3) as avg_duration_s,
               round(avg(fps), 2) as avg_fps
        from events where ev = 'capture'
        """,
    ),
    (
        "판정 (checking, ms)",
        {"save_ms", "predict_ms", "select_ms"},
        """
        select count(*) as n,
               round(avg(frames), 1) as frames, round(avg(ok), 1) as ok, round(avg(picked), 1) as picked,
               round(avg(save_ms), 2) as save, round(avg(predict_ms), 2) as predict,
               round(avg(select_ms), 3) as "select",
               round(avg(total_ms), 2) as total, round(max(total_ms), 2) as max_total
        from events where ev = 'checking'
        """,
    ),
    (
        "카메라 FPS",
        {"fps"},
        """
        select count(*) as n, round(avg(fps), 2) as avg, round(min(fps), 2) as min, round(max(fps), 2) as max
        from events where ev = 'camera_fps'
        """,
    ),
    (
        "화면 FPS: 상태별",
        {"fps", "state"},
        """
        select state, count(*) as n, round(avg(fps), 2) as avg, round(min(fps), 2) as min, round(max(fps), 2) as max
        from events where ev = 'screen_fps'
        group by all order by state
        """,
    ),
    (
        "화면 1장 시간: 상태별 (ms)",
        {"render_ms", "imshow_ms", "waitkey_ms", "waitkey_extra_ms"},
        """
        select state, count(*) as n,
               round(avg(render_ms), 2) as render, round(avg(imshow_ms), 2) as imshow,
               round(avg(waitkey_ms), 2) as waitkey, round(avg(waitkey_extra_ms), 2) as waitkey_extra,
               round(max(waitkey_extra_max_ms), 2) as waitkey_extra_max
        from events where ev = 'screen_fps' and render_ms is not null
        group by all order by state
        """,
    ),
    (
        "실행 목록",
        {"mode", "model"},
        """
        select filename, mode, model, screen, camera
        from events where ev = 'run'
        order by filename
        """,
    ),
    (
        "상태 전환 횟수",
        {"from", "to"},
        """
        select "from", "to", count(*) as n
        from events where ev = 'state'
        group by all order by n desc
        """,
    ),
]


def connect(files: list[str] | str = DEFAULT_GLOB) -> duckdb.DuckDBPyConnection:
    """로그 파일(또는 glob) -> events 뷰가 있는 duckdb 연결"""
    con = duckdb.connect()
    files = [files] if isinstance(files, str) else list(files)
    listed = ", ".join("'" + f.replace("'", "''") + "'" for f in files)   # create view 는 파라미터 바인딩 불가
    con.execute(
        f"create view events as select * from read_json_auto([{listed}], "
        "format='newline_delimited', union_by_name=true, filename=true)"
    )
    return con


def columns(con: duckdb.DuckDBPyConnection) -> set[str]:
    return {row[0] for row in con.execute("describe events").fetchall()}


def run_reports(con: duckdb.DuckDBPyConnection) -> list[tuple[str, duckdb.DuckDBPyRelation | None]]:
    """[(제목, 결과)]. 필요한 컬럼이 없는 보고서는 결과 None"""
    cols = columns(con)
    results = []
    for title, need, sql in REPORTS:
        if not need <= cols:
            results.append((title, None))
            continue
        results.append((title, con.sql(sql(cols) if callable(sql) else sql)))
    return results
