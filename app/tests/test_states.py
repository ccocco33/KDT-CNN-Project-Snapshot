import pytest

from photo_app import config
from photo_app.model import NOT_VISIBLE, TOO_SMALL, Face, Prediction, requested
from photo_app.states import Machine, State

BOX = (0, 0, 60, 60)         # 높이 >= MIN_FACE_H
SMALL_BOX = (0, 0, 20, 20)   # 높이 < MIN_FACE_H


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def pred(faces=1, smile=False, hand=False):
    return Prediction([Face(BOX, smile, True, True) for _ in range(faces)], [], hand, 0.0, requested(smile, False, True, True))


def guided(hand=False, visible=True, box=BOX):
    """안내 대상 얼굴 1명(보이지 않음 또는 작음)이 있는 PREVIEW 판정"""
    faces = [Face(BOX, None, None, True), Face(box, None, None, visible)]
    return Prediction(faces, [], hand, 0.0, requested(False, False, True, True))


def repeat(m, p, n):
    """같은 판정을 n회 연속으로 넣음"""
    for _ in range(n):
        m.on_prediction(p)


def machine_at(state):
    """state 까지 정상 경로로 이동한 Machine"""
    clock = Clock()
    m = Machine(now=clock)
    path = {
        State.HOME: [],
        State.PREVIEW: [lambda: m.click("start")],
        State.PREPARE: [lambda: m.click("start"), lambda: repeat(m, pred(hand=True), config.HAND_ON_COUNT)],
        State.CAPTURE: [lambda: m.click("start"), lambda: repeat(m, pred(hand=True), config.HAND_ON_COUNT),
                        lambda: repeat(m, pred(smile=True), config.SMILE_ON_COUNT)],
    }
    for step in path[state]:
        step()
    assert m.state is state
    return m, clock


def test_start_button_moves_to_preview():
    """초기화면에서 촬영 시작 버튼 클릭 시 미리보기로 이동"""
    # given
    m, _ = machine_at(State.HOME)

    # when
    m.click("start")

    # then
    assert m.state is State.PREVIEW


def test_home_button_in_preview():
    """미리보기에서 처음으로 버튼 클릭 시 초기화면으로 이동"""
    # given
    m, _ = machine_at(State.PREVIEW)

    # when
    m.click("home")

    # then
    assert m.state is State.HOME


def test_hand_raised_moves_to_prepare():
    """미리보기에서 손 들기가 HAND_ON_COUNT 회 연속이면 준비 단계로 이동"""
    # given
    m, _ = machine_at(State.PREVIEW)

    # when
    repeat(m, pred(hand=True), config.HAND_ON_COUNT)

    # then
    assert m.state is State.PREPARE


def test_hand_raised_once_stays_preview():
    """손 들기가 한 번 끊기면 처음부터 다시 셈"""
    # given
    m, _ = machine_at(State.PREVIEW)
    repeat(m, pred(hand=True), config.HAND_ON_COUNT - 1)
    m.on_prediction(pred())

    # when
    repeat(m, pred(hand=True), config.HAND_ON_COUNT - 1)

    # then
    assert m.state is State.PREVIEW


def test_hand_raised_without_face_is_ignored():
    """얼굴이 없으면 손을 들어도 미리보기 유지"""
    # given
    m, _ = machine_at(State.PREVIEW)

    # when
    repeat(m, pred(faces=0, hand=True), config.HAND_ON_COUNT)

    # then
    assert m.state is State.PREVIEW


def test_all_smile_moves_to_capture():
    """준비 단계에서 모두 웃음이 SMILE_ON_COUNT 회 연속이면 촬영 단계로 이동"""
    # given
    m, _ = machine_at(State.PREPARE)

    # when
    repeat(m, pred(faces=3, smile=True), config.SMILE_ON_COUNT)

    # then
    assert m.state is State.CAPTURE


def test_all_smile_once_stays_prepare():
    """모두 웃음이 한 번 끊기면 처음부터 다시 셈"""
    # given
    m, _ = machine_at(State.PREPARE)
    repeat(m, pred(faces=3, smile=True), config.SMILE_ON_COUNT - 1)
    m.on_prediction(pred(faces=3))

    # when
    repeat(m, pred(faces=3, smile=True), config.SMILE_ON_COUNT - 1)

    # then
    assert m.state is State.PREPARE


def test_not_all_smile_stays_prepare():
    """한 명이라도 웃지 않으면 준비 단계 유지"""
    # given
    m, _ = machine_at(State.PREPARE)
    faces = [Face(BOX, True, True), Face(BOX, False, True)]

    # when
    repeat(m, Prediction(faces, None, None, 0.0), config.SMILE_ON_COUNT)

    # then
    assert m.state is State.PREPARE


def test_zero_face_is_not_all_smile():
    """얼굴이 0명이면 모두 웃음으로 판정하지 않음"""
    # given
    m, _ = machine_at(State.PREPARE)

    # when
    repeat(m, pred(faces=0, smile=True), config.SMILE_ON_COUNT)

    # then
    assert m.state is State.PREPARE


def test_prepare_timeout_moves_to_preview_with_cancel_msg():
    """준비 시간 초과 시 취소 문구와 함께 미리보기로 이동"""
    # given
    m, clock = machine_at(State.PREPARE)

    # when
    clock.t += config.PREPARE_SEC
    m.tick()

    # then
    assert m.state is State.PREVIEW
    assert m.show_cancel_msg


def test_cancel_msg_disappears():
    """취소 문구는 표시 시간이 지나면 사라짐"""
    # given
    m, clock = machine_at(State.PREPARE)
    clock.t += config.PREPARE_SEC
    m.tick()

    # when
    clock.t += config.CANCEL_MSG_SEC

    # then
    assert not m.show_cancel_msg


def test_capture_timeout_moves_to_checking():
    """촬영 시간 종료 시 사진 확인 단계로 이동"""
    # given
    m, clock = machine_at(State.CAPTURE)

    # when
    clock.t += config.CAPTURE_SEC
    m.tick()

    # then
    assert m.state is State.CHECKING


def test_checked_moves_to_result():
    """사진 확인이 끝나면 결과 화면으로 이동"""
    # given
    m, clock = machine_at(State.CAPTURE)
    clock.t += config.CAPTURE_SEC
    m.tick()

    # when
    m.checked()

    # then
    assert m.state is State.RESULT


@pytest.mark.parametrize("button, expected", [("preview", State.PREVIEW), ("home", State.HOME)])
def test_result_buttons(button, expected):
    """결과 화면의 다시 찍기, 처음으로 버튼"""
    # given
    m, clock = machine_at(State.CAPTURE)
    clock.t += config.CAPTURE_SEC
    m.tick()
    m.checked()

    # when
    m.click(button)

    # then
    assert m.state is expected


@pytest.mark.parametrize("state", [State.PREPARE, State.CAPTURE])
def test_home_button_ignored_while_shooting(state):
    """촬영 중에는 처음으로 버튼이 동작하지 않음"""
    # given
    m, _ = machine_at(state)

    # when
    m.click("home")

    # then
    assert m.state is state


def test_remaining_time():
    """준비 단계의 남은 시간"""
    # given
    m, clock = machine_at(State.PREPARE)

    # when
    clock.t += 2

    # then
    assert m.remaining() == pytest.approx(config.PREPARE_SEC - 2)


def test_guide_shown_after_consecutive_targets():
    """안내 대상이 GUIDE_ON_COUNT 회 연속이면 안내 표시 시작"""
    # given
    m, _ = machine_at(State.PREVIEW)
    for _ in range(config.GUIDE_ON_COUNT - 1):
        m.on_prediction(guided(visible=False))
    assert not m.guiding

    # when
    m.on_prediction(guided(visible=False))

    # then
    assert m.guiding
    assert [reason for _, reason in m.guide_faces] == [NOT_VISIBLE]


def test_guide_reason_too_small():
    """작은 얼굴의 안내 이유는 too_small"""
    # given
    m, _ = machine_at(State.PREVIEW)

    # when
    for _ in range(config.GUIDE_ON_COUNT):
        m.on_prediction(guided(box=SMALL_BOX))

    # then
    assert [reason for _, reason in m.guide_faces] == [TOO_SMALL]


def test_single_target_does_not_block_start():
    """안내 대상이 한 번만 나오면 안내하지 않고, 다음 판정부터 손을 들면 시작"""
    # given
    m, _ = machine_at(State.PREVIEW)
    m.on_prediction(guided(visible=False))

    # when
    repeat(m, pred(hand=True), config.HAND_ON_COUNT)

    # then
    assert not m.guide_faces
    assert m.state is State.PREPARE


def test_hand_raised_with_target_in_same_frame_is_ignored():
    """안내 표시 전이라도 그 판정에 안내 대상이 있으면 시작 불가"""
    # given
    m, _ = machine_at(State.PREVIEW)

    # when
    repeat(m, guided(hand=True, visible=False), config.HAND_ON_COUNT)

    # then
    assert m.state is State.PREVIEW


def test_hand_raised_while_guiding_is_ignored():
    """안내 표시 중에는 안내 대상이 사라진 판정에서 손을 들어도 시작 불가"""
    # given
    m, _ = machine_at(State.PREVIEW)
    for _ in range(config.GUIDE_ON_COUNT):
        m.on_prediction(guided(visible=False))

    # when
    m.on_prediction(pred(hand=True))

    # then
    assert m.guiding
    assert m.state is State.PREVIEW


def test_guide_ends_after_consecutive_clear():
    """안내 대상 없음이 GUIDE_OFF_COUNT 회 연속이면 안내 표시 끝, 이후 손을 들면 시작"""
    # given
    m, _ = machine_at(State.PREVIEW)
    for _ in range(config.GUIDE_ON_COUNT):
        m.on_prediction(guided(visible=False))
    for _ in range(config.GUIDE_OFF_COUNT):
        m.on_prediction(pred())
    assert not m.guiding

    # when
    repeat(m, pred(hand=True), config.HAND_ON_COUNT)

    # then
    assert m.state is State.PREPARE


def test_guide_resets_when_leaving_preview():
    """미리보기를 벗어나면 안내 표시 초기화"""
    # given
    m, _ = machine_at(State.PREVIEW)
    for _ in range(config.GUIDE_ON_COUNT):
        m.on_prediction(guided(visible=False))

    # when
    m.click("home")
    m.click("start")

    # then
    assert not m.guiding and m.guide_faces == []


def test_home_eval_button_goes_to_model_eval_and_back():
    """초기화면 eval 버튼 -> 모델 평가, 처음으로 -> 초기화면"""
    # given
    m = Machine()

    # when
    m.click("eval")
    entered = m.state
    m.click("home")

    # then
    assert (entered, m.state) == (State.MODEL_EVAL, State.HOME)


@pytest.mark.parametrize("state", [State.PREPARE, State.CAPTURE])
def test_camera_lost_cancels_shooting(state):
    """촬영 중 카메라가 끊기면 촬영을 취소하고 PREVIEW 로 (취소 문구는 띄우지 않음)"""
    # given
    m, _ = machine_at(state)

    # when
    m.camera_lost()

    # then
    assert m.state is State.PREVIEW and not m.show_cancel_msg


@pytest.mark.parametrize("state", [State.HOME, State.PREVIEW])
def test_camera_lost_keeps_other_states(state):
    """촬영 중이 아니면 상태를 바꾸지 않음"""
    # given
    m, _ = machine_at(state)

    # when
    m.camera_lost()

    # then
    assert m.state is state
