from pathlib import Path

import numpy as np
import pytest

from photo_app import config
from photo_app.model import NOT_VISIBLE, TOO_SMALL, Face, Hand
from photo_app.states import State
from photo_app.ui import View, _face_score_lines, button_at, buttons, eval_passed, render

SIZE = (config.SCREEN_H, config.SCREEN_W, 3)


@pytest.mark.parametrize("state", list(State))
def test_render_size_without_frame(state):
    """
    CAMERA를 사용하지 않는 경우에 설정한 해상도대로 렌더링.
    """

    # given
    view = View()

    # when
    img = render(state, view)

    # then
    assert img.shape == SIZE


@pytest.mark.parametrize("state", [State.PREVIEW, State.PREPARE, State.CAPTURE])
def test_render_resizes_camera_frame(state):
    """
    CAMERA를 사용하는 경우에 frame이 설정한 해상도대로 리사이징 되어 렌더링
    """

    # given
    frame = np.zeros((720, 1280, 3), np.uint8)
    faces = [Face((100, 100, 200, 200), smile=False, eyes_open=True)]
    view = View(frame=frame, faces=faces)

    # when
    img = render(state, view)

    # then
    assert img.shape == SIZE


def test_render_result_with_photos():
    """
    결과 화면이 설정한 해상도 대로 렌더링
    """
    # given
    photos = [np.zeros((480, 640, 3), np.uint8)] * 5
    view = View(results=photos)

    # when
    img = render(State.RESULT, view)

    # then
    assert img.shape == SIZE


@pytest.mark.parametrize("state", [State.PREPARE, State.CAPTURE, State.CHECKING])
def test_no_buttons_while_shooting(state):
    """
    촬영 중에는 버튼이 0개.
    (버튼 자체가 추가될 수는 있음. 현재는 없음.)
    """

    # given: state

    # when
    bs = buttons(state)

    # then
    assert bs == []


@pytest.mark.parametrize("state", [State.PREVIEW, State.RESULT])
def test_home_button_in_preview_and_result(state):
    """
    미리보기와 결과화면에서 초기화면 버튼 존재.
    """
    # given: state

    # when
    ids = [b.id for b in buttons(state)]

    # then
    assert "home" in ids


def test_button_at_inside_start_button():
    """
    시작하기 버튼 클릭
    """

    # given
    x, y, w, h = buttons(State.HOME)[0].rect

    # when
    hit = button_at(State.HOME, x + w // 2, y + h // 2)

    # then
    assert hit == "start"


def test_button_at_outside_buttons():
    """
    시작하기 버튼 위치 밖에서 클릭
    """

    # given
    x, y = 0, 0

    # when
    hit = button_at(State.HOME, x, y)

    # then
    assert hit is None


@pytest.mark.parametrize("dev", [True, False])
def test_hand_box_drawn_only_in_dev(dev):
    """손 bbox 는 dev 모드에서만 표시"""
    # given
    frame = np.zeros((480, 640, 3), np.uint8)
    hands = [Hand((200, 200, 80, 80), raised=False)]
    with_hands = View(frame=frame, hands=hands, dev=dev)
    without_hands = View(frame=frame, dev=dev)

    # when
    diff = np.any(render(State.PREVIEW, with_hands) != render(State.PREVIEW, without_hands))

    # then
    assert diff == dev


@pytest.mark.parametrize("dev", [True, False])
def test_face_box_drawn_only_in_dev(dev):
    """검출 얼굴 bbox 는 dev 모드에서만 표시"""
    # given
    frame = np.zeros((480, 640, 3), np.uint8)
    faces = [Face((200, 200, 80, 80), None, None)]
    with_faces = View(frame=frame, faces=faces, dev=dev)
    without_faces = View(frame=frame, dev=dev)

    # when
    diff = np.any(render(State.CAPTURE, with_faces) != render(State.CAPTURE, without_faces))   # 얼굴 수 문구가 없는 화면

    # then
    assert diff == dev


def test_hand_label_shows_pose():
    """dev 모드 손 박스 글자는 판정 이유 (pose) 가 있으면 그것"""
    # given
    frame = np.zeros((480, 640, 3), np.uint8)
    fist = View(frame=frame, hands=[Hand((200, 200, 80, 80), raised=False, pose="fist")], dev=True)
    plain = View(frame=frame, hands=[Hand((200, 200, 80, 80), raised=False)], dev=True)

    # when
    diff = np.any(render(State.PREVIEW, fist) != render(State.PREVIEW, plain))

    # then
    assert diff


def test_face_label_shows_occlusion_score():
    """dev 모드 얼굴 박스 글자는 가림 점수가 있으면 그 점수"""
    # given
    frame = np.zeros((480, 640, 3), np.uint8)
    scored = View(frame=frame, faces=[Face((200, 200, 80, 80), None, None, occlusion=0.42)], dev=True)
    plain = View(frame=frame, faces=[Face((200, 200, 80, 80), None, None)], dev=True)

    # when
    diff = np.any(render(State.CAPTURE, scored) != render(State.CAPTURE, plain))

    # then
    assert diff


def test_face_label_shows_smile_eye_scores():
    """dev 모드 얼굴 박스 글자는 웃음, 눈 점수가 있으면 그 점수"""
    # given
    frame = np.zeros((480, 640, 3), np.uint8)
    face = Face((200, 200, 80, 80), True, True, True, scores={"smile": (0.95, 0.31, True), "eye": (0.97, 0.84, True)})
    scored = View(frame=frame, faces=[face], dev=True)
    plain = View(frame=frame, faces=[Face((200, 200, 80, 80), True, True, True)], dev=True)

    # when
    diff = np.any(render(State.PREPARE, scored) != render(State.PREPARE, plain))

    # then
    assert diff


@pytest.mark.parametrize("scores, visible, expected", [
    ({"smile": (0.95, 0.31, True), "eye": (0.97, 0.84, True)}, True, (["웃음 0.95 O", "눈 뜸 0.97 O"], True)),
    ({"smile": (0.2, 0.31, False)}, True, (["웃음 0.20 X"], False)),
    ({"blink": (0.1, 0.5, True)}, True, (["눈 뜸 0.90 O"], True)),
    ({}, False, (["얼굴 가림"], False)),
    ({}, True, (["face"], True)),
])
def test_face_score_lines(scores, visible, expected):
    """dev 얼굴 띠: 판정한 항목만, blink 는 눈 뜸 = 1 - 점수, 보이지 않으면 얼굴 가림"""
    # given
    face = Face((0, 0, 10, 10), None, None, visible, scores=scores)

    # when
    result = _face_score_lines(face)

    # then
    assert result == expected


@pytest.mark.parametrize("dev", [True, False])
def test_dev_label_drawn_only_in_dev(dev):
    """dev 문구는 dev 모드에서만 표시"""
    # given
    with_label = View(dev=dev, dev_label="dev | model: landmark")
    without_label = View(dev=dev)

    # when
    diff = np.any(render(State.HOME, with_label) != render(State.HOME, without_label))

    # then
    assert diff == dev


@pytest.mark.parametrize("reason", [NOT_VISIBLE, TOO_SMALL])
def test_preview_draws_guide_box(reason):
    """미리보기에서 안내 대상 얼굴이 있으면 박스와 문구를 그림"""
    # given
    frame = np.zeros((480, 640, 3), np.uint8)
    face = Face((200, 200, 100, 100), None, None, reason != NOT_VISIBLE)
    with_guide = View(frame=frame, faces=[face], guides=[(face, reason)])
    without_guide = View(frame=frame, faces=[face])

    # when
    a = render(State.PREVIEW, with_guide)
    b = render(State.PREVIEW, without_guide)

    # then
    assert not np.array_equal(a[150:310, 190:420], b[150:310, 190:420])


@pytest.mark.parametrize("dev", [True, False])
def test_eval_button_only_in_dev(dev):
    """초기화면의 모델 평가 버튼은 dev 모드에서만"""
    # given
    ids = [b.id for b in buttons(State.HOME, dev)]

    # when
    has_eval = "eval" in ids

    # then
    assert has_eval == dev


def test_model_eval_buttons_select_model_and_item():
    """모델 평가 화면 버튼: 모델, 판정 항목, 찍기, 처음으로"""
    # given
    ids = {b.id for b in buttons(State.MODEL_EVAL, True)}

    # when
    expected = {"model:fake", "model:landmark", "model:yunet_landmark", "model:yunet_cnn",
                "item:smile", "item:eye", "item:hand", "item:visibility", "item:eye_hold", "shot", "burst", "home"}

    # then
    assert ids == expected


def test_model_eval_draws_face_scores():
    """모델 평가 화면은 얼굴 점수 줄을 그림"""
    # given
    frame = np.zeros((480, 640, 3), np.uint8)
    face = Face((200, 150, 100, 100), True, True, True, scores={"smile": (0.62, 0.5, True)})
    with_scores = View(frame=frame, faces=[face], dev=True, eval_model="yunet_cnn")
    without = View(frame=frame, faces=[Face((200, 150, 100, 100), True, True, True)], dev=True, eval_model="yunet_cnn")

    # when
    a, b = render(State.MODEL_EVAL, with_scores), render(State.MODEL_EVAL, without)

    # then
    assert not np.array_equal(a[250:300, 200:360], b[250:300, 200:360])


@pytest.mark.parametrize("bbox", [(200, 380, 200, 100), (500, 400, 200, 120), (-30, -20, 100, 100)])
def test_model_eval_draws_scores_for_faces_near_edges(bbox):
    """화면 가장자리(아래, 오른쪽, 위 왼쪽 밖)에 걸친 얼굴도 점수 줄을 그리다 멈추지 않음"""
    # given
    face = Face(bbox, True, True, True, scores={"yaw": (0.1, 0.6, True), "occ": (0.1, 0.86, True),
                                               "smile": (0.7, 0.5, True), "eye": (0.9, 0.8, True)})
    view = View(frame=np.zeros((480, 640, 3), np.uint8), faces=[face], dev=True, eval_model="yunet_cnn")

    # when
    img = render(State.MODEL_EVAL, view)

    # then
    assert img.shape == (480, 640, 3)


def test_result_title_differs_when_closest():
    """조건에 가장 가까운 사진을 보여 줄 때 결과 제목 문구가 바뀜"""
    # given
    photos = [np.zeros((480, 640, 3), np.uint8)] * 3

    # when
    ok = render(State.RESULT, View(results=photos))
    closest = render(State.RESULT, View(results=photos, results_closest=True))

    # then
    assert not np.array_equal(ok[:40], closest[:40])


def test_camera_lost_shows_message_and_keeps_buttons():
    """카메라가 끊기면 카메라 화면 대신 안내 문구, 버튼은 그대로"""
    # given
    frame = np.full((480, 640, 3), 128, np.uint8)

    # when
    normal = render(State.PREVIEW, View(frame=frame))
    lost = render(State.PREVIEW, View(frame=frame, camera_lost=True))

    # then
    x, y, bw, bh = next(b.rect for b in buttons(State.PREVIEW) if b.id == "home")
    assert not np.array_equal(normal[200:280, 150:490], lost[200:280, 150:490])
    assert np.array_equal(normal[y:y + bh, x:x + bw], lost[y:y + bh, x:x + bw])


def test_bundled_korean_font_is_used_first():
    """저장소에 포함한 한글 폰트가 첫 후보이고 파일이 있음 (한글 폰트가 없는 환경에서 글자 깨짐 방지)"""
    # when
    first = Path(config.FONT_PATHS[0])

    # then
    assert first.parent.name == "fonts" and first.exists()


@pytest.mark.parametrize("eye_ok, held, expected", [
    (True, False, True),    # 눈 뜸
    (False, True, True),    # 눈 감음이지만 eye_hold 허용
    (False, False, False),  # 눈 감음, 허용 안 됨
])
def test_eval_passed_counts_eye_hold(eye_ok, held, expected):
    """모델 평가 화면 통과: eye_hold 로 허용한 얼굴은 eye 가 X 여도 통과, eye_hold X 만으로는 실패하지 않음"""
    # given
    face = Face((0, 0, 80, 100), True, eye_ok or held, True, 0.5,
                scores={"smile": (0.9, 0.31, True), "eye": (0.5, 0.84, eye_ok), "eye_hold": (0.4 if held else 0.0, 0.3, held)})

    # when
    ok = eval_passed(face)

    # then
    assert ok is expected
