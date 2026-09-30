from datetime import datetime

import numpy as np

from photo_app.storage import Session

NOW = lambda: datetime(2026, 9, 24, 12, 0, 0)


def frames(n):
    return [np.zeros((10, 10, 3), np.uint8) for _ in range(n)]


def test_save_all_frames_to_session_dir(tmp_path):
    """촬영한 사진 전부를 세션 폴더에 저장"""
    # given
    session = Session(tmp_path, now=NOW)

    # when
    paths = session.save(frames(3))

    # then
    assert session.dir == tmp_path / "20260924_120000"
    assert [p.name for p in paths] == ["frame_000.jpg", "frame_001.jpg", "frame_002.jpg"]
    assert all(p.exists() for p in paths)


def test_save_uses_given_numbers(tmp_path):
    """번호를 주면 그 번호로 파일 이름을 지음 (고른 사진의 촬영 순서)"""
    # given
    session = Session(tmp_path, now=NOW)

    # when
    paths = session.save(frames(2), numbers=[4, 17])

    # then
    assert [p.name for p in paths] == ["frame_004.jpg", "frame_017.jpg"]


def test_keep_only_result_photos(tmp_path):
    """결과 사진만 남기고 나머지 삭제"""
    # given
    session = Session(tmp_path, now=NOW)
    paths = session.save(frames(5))

    # when
    session.keep_only([paths[0], paths[4]])

    # then
    assert sorted(session.dir.iterdir()) == [paths[0], paths[4]]


def test_keep_nothing_removes_session_dir(tmp_path):
    """결과 사진이 없으면 세션 폴더까지 삭제"""
    # given
    session = Session(tmp_path, now=NOW)
    session.save(frames(3))

    # when
    session.keep_only([])

    # then
    assert not session.dir.exists()
