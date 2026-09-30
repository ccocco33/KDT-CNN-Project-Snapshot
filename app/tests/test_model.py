import pytest

from photo_app.model import NOT_VISIBLE, TOO_SMALL, Face, Prediction, requested

BOX = (0, 0, 10, 10)


def pred(*faces, smile=True, eye=True):
    return Prediction(list(faces), None, None, 0.0, requested(smile, eye, False, False))


def test_meets_condition_no_face():
    """얼굴이 없으면 조건 미충족"""
    # given
    p = pred()

    # when
    ok = p.meets_condition()

    # then
    assert not ok


def test_meets_condition_everyone_smiles_with_eyes_open():
    """모두 웃고 눈을 뜨면 조건 충족"""
    # given
    p = pred(Face(BOX, True, True), Face(BOX, True, True))

    # when
    ok = p.meets_condition()

    # then
    assert ok


def test_meets_condition_one_not_smiling():
    """한 명이라도 웃지 않으면 조건 미충족"""
    # given
    p = pred(Face(BOX, True, True), Face(BOX, False, True))

    # when
    ok = p.meets_condition()

    # then
    assert not ok


def test_meets_condition_one_eyes_closed():
    """한 명이라도 눈을 감으면 조건 미충족"""
    # given
    p = pred(Face(BOX, True, True), Face(BOX, True, False))

    # when
    ok = p.meets_condition()

    # then
    assert not ok


@pytest.mark.parametrize("smile, eye", [(True, False), (False, True), (False, False)])
def test_meets_condition_requires_smile_and_eye(smile, eye):
    """smile, eye 를 함께 요청하지 않은 결과면 오류"""
    # given
    p = pred(smile=smile, eye=eye)

    # when, then
    with pytest.raises(ValueError, match="smile=True, eye=True"):
        p.meets_condition()


def test_requested():
    """요청 인자 중 True 인 것만 기록"""
    # given
    flags = dict(smile=True, eye=False, hand=True, visibility=True)

    # when
    names = requested(**flags)

    # then
    assert names == {"smile", "hand", "visibility"}


MIN_H = 50
BIG = (0, 0, 60, 60)      # 높이 60 >= MIN_H
SMALL = (0, 0, 40, 40)    # 높이 40 < MIN_H


def preview(*faces, hand=True, visibility=True):
    return Prediction(list(faces), [], hand, 0.0, requested(False, False, True, visibility))


def test_guidance_reasons():
    """보이지 않는 얼굴은 not_visible, 작은 얼굴은 too_small, 나머지는 안내 없음"""
    # given
    hidden = Face(BIG, None, None, False)
    small = Face(SMALL, None, None, True)
    ok = Face(BIG, None, None, True)
    p = preview(hidden, small, ok)

    # when
    guides = p.guidance(MIN_H)

    # then
    assert guides == [(hidden, NOT_VISIBLE), (small, TOO_SMALL)]


def test_guidance_not_visible_first():
    """작고 보이지 않는 얼굴은 not_visible 만"""
    # given
    face = Face(SMALL, None, None, False)

    # when
    guides = preview(face).guidance(MIN_H)

    # then
    assert guides == [(face, NOT_VISIBLE)]


def test_guidance_requires_visibility():
    """visibility 미요청 결과면 guidance 오류"""
    # given
    p = preview(Face(BIG, None, None), visibility=False)

    # when, then
    with pytest.raises(ValueError):
        p.guidance(MIN_H)


def test_ready_to_start():
    """손 들기, 얼굴 1명 이상, 안내 대상 없음이면 시작 가능"""
    # given
    p = preview(Face(BIG, None, None, True))

    # when
    ready = p.ready_to_start(MIN_H)

    # then
    assert ready


@pytest.mark.parametrize("faces, hand", [
    ([Face(BIG, None, None, True)], False),                                  # 손 안 듦
    ([], True),                                                              # 얼굴 0명
    ([Face(BIG, None, None, True), Face(SMALL, None, None, True)], True),    # 작은 얼굴
    ([Face(BIG, None, None, True), Face(BIG, None, None, False)], True),     # 보이지 않는 얼굴
])
def test_not_ready_to_start(faces, hand):
    """손을 안 들었거나, 얼굴이 없거나, 안내 대상이 있으면 시작 불가"""
    # given
    p = preview(*faces, hand=hand)

    # when
    ready = p.ready_to_start(MIN_H)

    # then
    assert not ready


@pytest.mark.parametrize("hand, visibility", [(False, True), (True, False)])
def test_ready_to_start_requires_hand_and_visibility(hand, visibility):
    """hand, visibility 를 함께 요청하지 않은 결과면 ready_to_start 오류"""
    # given
    p = Prediction([Face(BIG, None, None, True)], [], True, 0.0, requested(False, False, hand, visibility))

    # when, then
    with pytest.raises(ValueError):
        p.ready_to_start(MIN_H)


def test_meets_condition_not_visible_face():
    """보이지 않는 얼굴(웃음, 눈 None)이 있으면 조건 미충족"""
    # given
    p = pred(Face(BOX, True, True, True), Face(BOX, None, None, False))

    # when
    ok = p.meets_condition()

    # then
    assert not ok


def test_eye_score_is_lowest_face():
    """사진의 눈 뜸 점수는 얼굴 중 가장 낮은 점수"""
    # given
    p = pred(Face(BOX, True, True, None, 0.9), Face(BOX, True, True, None, 0.6))

    # when
    score = p.eye_score()

    # then
    assert score == 0.6


@pytest.mark.parametrize("faces", [[], [Face(BOX, True, True, None, 0.9), Face(BOX, True, True)]])
def test_eye_score_none(faces):
    """얼굴이 없거나 점수가 없는 얼굴이 있으면 None"""
    # given
    p = pred(*faces)

    # when
    score = p.eye_score()

    # then
    assert score is None
