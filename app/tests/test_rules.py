import numpy as np
import pytest

from photo_app.models.landmark import rules


def bs(smile_l=0.0, smile_r=0.0, blink_l=0.0, blink_r=0.0):
    return {"mouthSmileLeft": smile_l, "mouthSmileRight": smile_r,
            "eyeBlinkLeft": blink_l, "eyeBlinkRight": blink_r}


def test_smile_when_both_corners_up():
    """양쪽 입꼬리 점수 평균이 임계값 이상이면 웃음"""
    # given
    scores = bs(smile_l=0.8, smile_r=0.6)

    # when
    smiling = rules.is_smiling(scores, th=0.5)

    # then
    assert smiling


def test_no_smile_when_one_corner_up():
    """한쪽 입꼬리만 올라가 평균이 임계값 미만이면 웃음 아님"""
    # given
    scores = bs(smile_l=0.8, smile_r=0.1)

    # when
    smiling = rules.is_smiling(scores, th=0.5)

    # then
    assert not smiling


def test_smile_threshold_boundary():
    """평균이 임계값과 같으면 웃음"""
    # given
    scores = bs(smile_l=0.5, smile_r=0.5)

    # when
    smiling = rules.is_smiling(scores, th=0.5)

    # then
    assert smiling


def test_eyes_open_when_no_blink():
    """양쪽 눈 깜빡임 점수가 낮으면 눈 뜸"""
    # given
    scores = bs(blink_l=0.1, blink_r=0.2)

    # when
    open_ = rules.is_eyes_open(scores, th=0.5)

    # then
    assert open_


@pytest.mark.parametrize("blink_l, blink_r", [(0.9, 0.1), (0.1, 0.9)])
def test_eyes_closed_when_one_eye_blinks(blink_l, blink_r):
    """한쪽 눈만 감아도 눈 감음"""
    # given
    scores = bs(blink_l=blink_l, blink_r=blink_r)

    # when
    open_ = rules.is_eyes_open(scores, th=0.5)

    # then
    assert not open_


def test_yaw_frontal_face():
    """코가 두 눈 가운데에 있으면 yaw 0"""
    # given
    right_eye, left_eye, nose = (40, 50), (60, 50), (50, 60)

    # when
    value = rules.yaw(right_eye, left_eye, nose)

    # then
    assert value == pytest.approx(0.0)


def test_yaw_turned_face():
    """코가 두 눈 가운데에서 눈 사이 거리의 절반만큼 벗어나면 yaw 0.5 (방향 무관)"""
    # given
    right_eye, left_eye = (40, 50), (60, 50)

    # when
    values = [rules.yaw(right_eye, left_eye, (x, 60)) for x in (60, 40)]

    # then
    assert values == pytest.approx([0.5, 0.5])


def test_yaw_returns_python_float_for_numpy_input():
    """numpy 좌표(YuNet 출력)가 들어와도 파이썬 float 반환 (비교 결과가 bool)"""
    # given
    right_eye, left_eye, nose = np.float32([40, 50]), np.float32([60, 50]), np.float32([60, 60])

    # when
    value = rules.yaw(right_eye, left_eye, nose)

    # then
    assert type(value) is float
    assert (value < 0.1) is False
