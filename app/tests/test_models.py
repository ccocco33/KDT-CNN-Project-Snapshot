from pathlib import Path

import numpy as np
import pytest

from photo_app.models.fake import config as fake_config
from photo_app.models.landmark import config as landmark_config
from photo_app.models.yunet_landmark import config as yunet_landmark_config
from photo_app.models.yunet_cnn import config as yunet_cnn_config
from photo_app.models.yunet_cnn.model import lower, square_crop, upper
from photo_app.models import create_model
from photo_app.models.fake import FakeModel
from photo_app.models.util.keyboard import KeyboardHand


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_create_fake_model():
    """이름 fake 로 가짜 모델 생성"""
    # given
    name = "fake"

    # when
    model = create_model(name)

    # then
    assert isinstance(model, FakeModel)


def test_create_unknown_model():
    """알 수 없는 이름이면 오류"""
    # given
    name = "unknown"

    # when, then
    with pytest.raises(ValueError):
        create_model(name)


def test_keyboard_hand_raised_after_h():
    """h 를 누르면 손 든 상태"""
    # given
    hand = KeyboardHand(now=Clock())

    # when
    hand.handle_key("h")

    # then
    assert hand.raised()


def test_keyboard_hand_expires():
    """손 든 상태는 유지 시간이 지나면 풀림"""
    # given
    clock = Clock()
    hand = KeyboardHand(now=clock)
    hand.handle_key("h")

    # when
    clock.t += KeyboardHand.HOLD_SEC

    # then
    assert not hand.raised()


@pytest.mark.skipif(not (landmark_config.FACE_MODEL_PATH.exists() and landmark_config.HAND_MODEL_PATH.exists()), reason="landmark 모델 파일 없음")
def test_landmark_model_no_face_on_blank_frame():
    """얼굴이 없는 프레임에서 얼굴 0명"""
    # given
    model = create_model("landmark")
    frame = np.zeros((480, 640, 3), np.uint8)

    # when
    pred = model.predict(frame, smile=True, eye=True, hand=True, visibility=True)

    # then
    assert pred.faces == []


def test_fake_model_skips_hand_when_hand_false():
    """hand 미요청이면 키보드로 손을 들어도 손 결과 None"""
    # given
    model = create_model("fake")
    model.handle_key("h")
    frame = np.zeros((480, 640, 3), np.uint8)

    # when
    pred = model.predict(frame, smile=True, eye=True, hand=False, visibility=True)

    # then
    assert pred.hand_raised is None
    assert pred.hands is None


@pytest.mark.skipif(not (landmark_config.FACE_MODEL_PATH.exists() and landmark_config.HAND_MODEL_PATH.exists()), reason="landmark 모델 파일 없음")
def test_landmark_model_skips_hand_when_hand_false():
    """hand 미요청이면 손 결과 None"""
    # given
    model = create_model("landmark")
    model.handle_key("h")
    frame = np.zeros((480, 640, 3), np.uint8)

    # when
    pred = model.predict(frame, smile=True, eye=True, hand=False, visibility=True)

    # then
    assert pred.hand_raised is None
    assert pred.hands is None


YUNET_LANDMARK_FILES = (yunet_landmark_config.FACE_MODEL_PATH, yunet_landmark_config.HAND_MODEL_PATH, yunet_landmark_config.YUNET_PATH)


@pytest.mark.skipif(not all(p.exists() for p in YUNET_LANDMARK_FILES), reason="yunet_landmark 모델 파일 없음")
def test_yunet_landmark_model_no_face_on_blank_frame():
    """얼굴이 없는 프레임에서 얼굴 0명"""
    # given
    model = create_model("yunet_landmark")
    frame = np.zeros((480, 640, 3), np.uint8)

    # when
    pred = model.predict(frame, smile=True, eye=True, hand=True, visibility=True)

    # then
    assert pred.faces == []


@pytest.mark.skipif(not all(p.exists() for p in YUNET_LANDMARK_FILES), reason="yunet_landmark 모델 파일 없음")
def test_yunet_landmark_model_skips_landmark_when_smile_eye_false():
    """smile, eye 미요청이면 잘라낸 얼굴 판정 생략"""
    # given
    model = create_model("yunet_landmark")
    frame = np.zeros((480, 640, 3), np.uint8)

    # when
    pred = model.predict(frame, smile=False, eye=False, hand=True, visibility=True)

    # then
    assert "landmark" not in pred.timings


@pytest.mark.skipif(not all(p.exists() for p in YUNET_LANDMARK_FILES), reason="yunet_landmark 모델 파일 없음")
def test_yunet_landmark_model_uses_own_config(monkeypatch):
    """yunet_landmark 는 landmark 가 아닌 자기 config 의 임계값 사용"""
    # given
    monkeypatch.setattr(yunet_landmark_config, "SMILE_TH", 0.31)
    monkeypatch.setattr(yunet_landmark_config, "BLINK_TH", 0.32)
    monkeypatch.setattr(yunet_landmark_config, "HAND_SCORE_TH", 0.33)
    monkeypatch.setattr(yunet_landmark_config, "HAND_IOU_TH", 0.34)

    # when
    model = create_model("yunet_landmark")

    # then
    assert (model._smile_th, model._blink_th, model._hands.score_th, model._hands.iou_th) == (0.31, 0.32, 0.33, 0.34)


@pytest.mark.parametrize("smile, eye", [(True, False), (False, True)])
def test_fake_model_returns_none_for_unrequested(smile, eye):
    """요청하지 않은 웃음, 눈 결과는 None"""
    # given
    model = FakeModel(yunet_path=Path("no_such_file"))   # 가운데 가짜 얼굴 1개
    frame = np.zeros((480, 640, 3), np.uint8)

    # when
    pred = model.predict(frame, smile=smile, eye=eye, hand=False, visibility=True)

    # then
    face = pred.faces[0]
    assert (face.smile is None) == (not smile)
    assert (face.eyes_open is None) == (not eye)


def test_fake_model_visibility_none_when_unrequested():
    """visibility 미요청이면 보임 결과 None"""
    # given
    model = FakeModel(yunet_path=Path("no_such_file"))
    frame = np.zeros((480, 640, 3), np.uint8)

    # when
    pred = model.predict(frame, smile=True, eye=True, hand=False, visibility=False)

    # then
    assert pred.faces[0].visible is None


def test_fake_model_hidden_face_skips_smile_eye():
    """o 로 보이지 않음이면 visible False, 웃음과 눈은 None"""
    # given
    model = FakeModel(yunet_path=Path("no_such_file"))
    model.handle_key("o")
    frame = np.zeros((480, 640, 3), np.uint8)

    # when
    pred = model.predict(frame, smile=True, eye=True, hand=False, visibility=True)

    # then
    face = pred.faces[0]
    assert face.visible is False
    assert face.smile is None and face.eyes_open is None


@pytest.mark.skipif(not fake_config.YUNET_PATH.exists(), reason="YuNet 모델 파일 없음")
def test_fake_model_uses_own_score_threshold():
    """fake 모델의 YuNet 신뢰도 기준은 OpenCV 기본값이 아닌 자기 config 값"""
    # given
    expected = fake_config.SCORE_TH

    # when
    model = FakeModel()

    # then
    assert model._detector.getScoreThreshold() == pytest.approx(expected)


YUNET_CNN_FILES = (yunet_cnn_config.YUNET_PATH, yunet_cnn_config.SMILE_PATH, yunet_cnn_config.EYE_PATH, yunet_cnn_config.OCC_PATH,
                   yunet_cnn_config.HAND_PATH)
needs_yunet_cnn = pytest.mark.skipif(not all(p.exists() for p in YUNET_CNN_FILES), reason="yunet_cnn 모델 파일 없음")
FRAME = np.zeros((480, 640, 3), np.uint8)


def stub_faces(model, yaw=0.0, occ=0.0):
    """YuNet 대신 가로 100 x 세로 100 얼굴 1개를 돌려주도록 바꿈. 가림 점수는 occ"""
    box = (100, 100, 100, 100)
    model._detect_faces = lambda frame: [(box, box, yaw)]
    model._occ = lambda crop: occ


def fail(_):
    raise AssertionError("실행되면 안 되는 CNN")


@needs_yunet_cnn
def test_yunet_cnn_model_no_face_on_blank_frame():
    """얼굴이 없는 프레임에서 얼굴 0명"""
    # given
    model = create_model("yunet_cnn")

    # when
    pred = model.predict(FRAME, smile=True, eye=True, hand=True, visibility=True)

    # then
    assert pred.faces == []


class FakeYunet:
    """cv2.FaceDetectorYN 흉내. 받은 입력 크기를 기록하고 found 를 그대로 돌려줌"""

    def __init__(self, found):
        self.found, self.size = found, None

    def setInputSize(self, size):
        self.size = size

    def detect(self, img):
        return 1, self.found


@needs_yunet_cnn
def test_yunet_cnn_detects_on_reduced_input(monkeypatch):
    """YuNet 입력은 가로 YUNET_INPUT_WIDTH 로 줄이고, 박스는 프레임 좌표로 되돌림"""
    # given
    model = create_model("yunet_cnn")
    monkeypatch.setattr(yunet_cnn_config, "YUNET_INPUT_WIDTH", 400)
    found = np.array([[40, 30, 20, 15] + [10, 5] * 5 + [0.9]], np.float32)   # 400x300 입력 기준
    model._detector = FakeYunet(found)

    # when
    faces = model._detect_faces(FRAME)

    # then
    assert model._detector.size == (400, 300)
    assert faces[0][0] == (64, 48, 32, 24)


@needs_yunet_cnn
def test_yunet_cnn_model_uses_own_thresholds(monkeypatch):
    """웃음, 눈 뜸은 yunet_cnn config 의 기준값으로 판정"""
    # given
    model = create_model("yunet_cnn")
    stub_faces(model)
    model._smile = lambda crop: 0.6
    model._eye = lambda crop: 0.4
    monkeypatch.setattr(yunet_cnn_config, "SMILE_TH", 0.5)
    monkeypatch.setattr(yunet_cnn_config, "EYE_TH", 0.5)

    # when
    face = model.predict(FRAME, smile=True, eye=True, hand=False, visibility=True).faces[0]

    # then
    assert (face.smile, face.eyes_open, face.visible) == (True, False, True)


@needs_yunet_cnn
def test_yunet_cnn_model_skips_cnn_for_hidden_face():
    """보이지 않는 얼굴은 CNN 을 실행하지 않고 웃음, 눈 None"""
    # given
    model = create_model("yunet_cnn")
    stub_faces(model, yaw=1.0)
    model._smile = model._eye = model._occ = fail

    # when
    face = model.predict(FRAME, smile=True, eye=True, hand=False, visibility=True).faces[0]

    # then
    assert (face.smile, face.eyes_open, face.visible) == (None, None, False)


@needs_yunet_cnn
def test_yunet_cnn_model_runs_only_requested_cnn():
    """smile 만 요청하면 eye CNN 은 실행하지 않음 (PREPARE)"""
    # given
    model = create_model("yunet_cnn")
    stub_faces(model)
    model._smile = lambda crop: 0.9
    model._eye = fail

    # when
    pred = model.predict(FRAME, smile=True, eye=False, hand=False, visibility=False)

    # then
    assert pred.faces[0].smile is True and pred.faces[0].eyes_open is None
    assert "eye" not in pred.timings


@needs_yunet_cnn
def test_yunet_cnn_model_smile_gets_lower_face(monkeypatch):
    """smile CNN 에는 얼굴 정사각형의 아래쪽 SMILE_BOTTOM 만 들어감"""
    # given
    model = create_model("yunet_cnn")
    stub_faces(model)
    shapes = []
    model._smile = lambda crop: shapes.append(crop.shape) or 0.9
    monkeypatch.setattr(yunet_cnn_config, "SMILE_BOTTOM", 0.45)

    # when
    model.predict(FRAME, smile=True, eye=False, hand=False, visibility=False)

    # then
    assert shapes == [(45, 100, 3)]


@needs_yunet_cnn
def test_yunet_cnn_model_occluded_face_is_hidden(monkeypatch):
    """가림 점수가 OCC_TH 이상이면 보이지 않음. 웃음, 눈 CNN 은 실행하지 않음"""
    # given
    model = create_model("yunet_cnn")
    stub_faces(model, occ=0.7)
    model._smile = model._eye = fail
    monkeypatch.setattr(yunet_cnn_config, "OCC_TH", 0.6)

    # when
    pred = model.predict(FRAME, smile=True, eye=True, hand=False, visibility=True)

    # then
    assert (pred.faces[0].smile, pred.faces[0].eyes_open, pred.faces[0].visible) == (None, None, False)
    assert pred.faces[0].occlusion == 0.7
    assert "occlusion" in pred.timings


@needs_yunet_cnn
def test_yunet_cnn_model_skips_occlusion_cnn_without_visibility():
    """visibility 를 요청하지 않으면 가림 CNN 을 실행하지 않음 (PREPARE)"""
    # given
    model = create_model("yunet_cnn")
    stub_faces(model)
    model._occ = fail
    model._smile = lambda crop: 0.9

    # when
    pred = model.predict(FRAME, smile=True, eye=False, hand=False, visibility=False)

    # then
    assert pred.faces[0].visible is None
    assert "occlusion" not in pred.timings


@needs_yunet_cnn
def test_yunet_cnn_model_visibility_only_request():
    """visibility 만 요청해도 가림 CNN 으로 보임 판정 (PREVIEW)"""
    # given
    model = create_model("yunet_cnn")
    stub_faces(model, occ=0.1)
    model._smile = model._eye = fail

    # when
    face = model.predict(FRAME, smile=False, eye=False, hand=False, visibility=True).faces[0]

    # then
    assert face.visible is True


@needs_yunet_cnn
def test_yunet_cnn_model_fills_scores(monkeypatch):
    """yunet_cnn 은 실행한 판정 항목의 (점수, 기준값, 통과) 를 scores 에 채움"""
    # given
    model = create_model("yunet_cnn")
    stub_faces(model, yaw=0.2, occ=0.1)
    model._smile = lambda crop: 0.6
    model._eye = lambda crop: 0.9
    monkeypatch.setattr(yunet_cnn_config, "SMILE_TH", 0.5)

    # when
    face = model.predict(FRAME, smile=True, eye=True, hand=False, visibility=True).faces[0]

    # then
    assert set(face.scores) == {"yaw", "occ", "smile", "eye"}
    assert face.scores["smile"] == (0.6, 0.5, True)


def test_square_crop_inside_frame():
    """박스 중심 기준 한 변 = 박스 높이 정사각형으로 자르고 RGB 로 바꿈"""
    # given
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[:, :, 0] = 255   # BGR 의 파랑

    # when
    crop = square_crop(frame, (40, 30, 20, 40))

    # then
    assert crop.shape == (40, 40, 3)
    assert crop[0, 0].tolist() == [0, 0, 255]


def test_square_crop_side_is_box_height():
    """한 변은 박스 높이 (가로가 더 긴 박스도 높이 기준)"""
    # given
    frame = np.zeros((100, 100, 3), np.uint8)

    # when
    crop = square_crop(frame, (20, 30, 50, 30))

    # then
    assert crop.shape == (30, 30, 3)


def test_square_crop_outside_frame_shifts_inside():
    """화면 밖으로 나가면 채우지 않고 안쪽으로 밀어 넣음 (학습 전처리와 같음)"""
    # given
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[:40, :] = 200   # 맨 위 40줄만 밝게

    # when
    crop = square_crop(frame, (10, -20, 40, 40))

    # then
    assert crop.shape == (40, 40, 3)
    assert (crop == 200).all()   # 프레임 맨 위 0~39 줄을 자름 (가장자리 복제 없음)


def test_square_crop_side_limited_by_frame():
    """박스가 프레임보다 크면 한 변은 프레임 크기 이하"""
    # given
    frame = np.zeros((60, 100, 3), np.uint8)

    # when
    crop = square_crop(frame, (0, -30, 120, 120))

    # then
    assert crop.shape == (60, 60, 3)


def test_upper_face():
    """얼굴 정사각형의 위쪽 비율만 남김"""
    # given
    face = np.zeros((100, 100, 3), np.uint8)

    # when
    top = upper(face, 0.6)

    # then
    assert top.shape == (60, 100, 3)


def test_lower_face():
    """얼굴 정사각형의 아래쪽 비율만 남김"""
    # given
    face = np.zeros((100, 100, 3), np.uint8)
    face[55:] = 200

    # when
    bottom = lower(face, 0.45)

    # then
    assert bottom.shape == (45, 100, 3)
    assert (bottom == 200).all()


def test_fake_model_eye_score_follows_eyes_open():
    """fake 모델의 눈 뜸 점수: 뜸 1.0, 감음 0.0, eye 미요청이면 None"""
    # given
    model = FakeModel(yunet_path=Path("no_such_file"))

    # when
    opened = model.predict(FRAME, smile=True, eye=True, hand=False, visibility=False).faces[0].eye_score
    model.handle_key("e")
    closed = model.predict(FRAME, smile=True, eye=True, hand=False, visibility=False).faces[0].eye_score
    unrequested = model.predict(FRAME, smile=True, eye=False, hand=False, visibility=False).faces[0].eye_score

    # then
    assert (opened, closed, unrequested) == (1.0, 0.0, None)


@needs_yunet_cnn
def test_yunet_cnn_model_returns_eye_score():
    """yunet_cnn 의 눈 뜸 점수는 eye CNN 점수"""
    # given
    model = create_model("yunet_cnn")
    stub_faces(model)
    model._eye = lambda crop: 0.83

    # when
    face = model.predict(FRAME, smile=False, eye=True, hand=False, visibility=False).faces[0]

    # then
    assert face.eye_score == 0.83
