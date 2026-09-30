import numpy as np
import pytest

from photo_app.models.util import hand
from photo_app.models.util.hand import HandDetector, decode, dequantize, letterbox, quantize

SIZE = 320


def yolo_output(*boxes):
    """YOLO 출력 (5, N). boxes: (중심 x, 중심 y, 너비, 높이, 점수), 좌표는 입력 기준 0~1"""
    return np.array(boxes, np.float32).T


def test_letterbox_keeps_ratio_and_centers():
    """가로로 긴 화면은 위아래에 여백을 두고 가운데 배치"""
    # given
    rgb = np.zeros((480, 640, 3), np.uint8)

    # when
    img, scale, pad = letterbox(rgb, SIZE)

    # then
    assert img.shape == (SIZE, SIZE, 3)
    assert scale == pytest.approx(0.5)
    assert pad == (0, 40)
    assert img[0, 0, 0] == 114 and img[40, 0, 0] == 0


def test_decode_maps_box_back_to_frame():
    """letterbox 좌표 -> 원래 화면 픽셀 bbox"""
    # given: 640x480 화면 (배율 0.5, 위 여백 40). 화면 (200, 100) ~ (300, 240) 의 손
    pred = yolo_output((125 / SIZE, 125 / SIZE, 50 / SIZE, 70 / SIZE, 0.9))

    # when
    found = decode(pred, SIZE, 0.5, (0, 40), 640, 480, score_th=0.5, iou_th=0.45, max_hands=4)

    # then
    assert len(found) == 1
    assert found[0][0] == (200, 100, 100, 140)
    assert found[0][1] == pytest.approx(0.9)


def test_decode_accepts_pixel_coords_and_transposed():
    """(N, 5) 출력, 입력 픽셀 좌표도 같은 결과"""
    # given
    pred = np.array([[125, 125, 50, 70, 0.9]], np.float32)

    # when
    found = decode(pred, SIZE, 0.5, (0, 40), 640, 480, score_th=0.5, iou_th=0.45, max_hands=4)

    # then
    assert found[0][0] == (200, 100, 100, 140)


def test_decode_drops_low_score():
    """점수가 기준 미만이면 손 없음"""
    # given
    pred = yolo_output((0.5, 0.5, 0.2, 0.2, 0.3))

    # when
    found = decode(pred, SIZE, 1.0, (0, 0), SIZE, SIZE, score_th=0.5, iou_th=0.45, max_hands=4)

    # then
    assert found == []


def test_decode_nms_keeps_best_of_overlapping():
    """겹친 박스는 점수 높은 1개만 남기고, 떨어진 손은 따로 남김 (점수 높은 순)"""
    # given
    pred = yolo_output((0.3, 0.3, 0.2, 0.2, 0.8), (0.31, 0.3, 0.2, 0.2, 0.6), (0.7, 0.7, 0.2, 0.2, 0.9))

    # when
    found = decode(pred, SIZE, 1.0, (0, 0), SIZE, SIZE, score_th=0.5, iou_th=0.45, max_hands=4)

    # then
    assert [round(score, 2) for _, score in found] == [0.9, 0.8]


def test_decode_limits_max_hands():
    """최대 손 수까지만 반환"""
    # given
    pred = yolo_output((0.2, 0.2, 0.1, 0.1, 0.9), (0.5, 0.5, 0.1, 0.1, 0.8), (0.8, 0.8, 0.1, 0.1, 0.7))

    # when
    found = decode(pred, SIZE, 1.0, (0, 0), SIZE, SIZE, score_th=0.5, iou_th=0.45, max_hands=2)

    # then
    assert len(found) == 2


def test_decode_clips_box_to_frame():
    """화면 밖으로 나간 박스는 잘라냄"""
    # given
    pred = yolo_output((0.0, 0.0, 0.2, 0.2, 0.9))

    # when
    found = decode(pred, SIZE, 1.0, (0, 0), SIZE, SIZE, score_th=0.5, iou_th=0.45, max_hands=4)

    # then
    assert found[0][0] == (0, 0, 32, 32)


def test_quantize_roundtrip_int8():
    """int8 텐서면 scale, zero point 로 양자화, 되돌리면 원래 값 근처"""
    # given
    detail = {"dtype": np.int8, "quantization": (1 / 255, -128)}
    x = np.array([0.0, 0.5, 1.0], np.float32)

    # when
    y = dequantize(quantize(x, detail), detail)

    # then
    assert y == pytest.approx(x, abs=1 / 255)


def test_quantize_float_passthrough():
    """float32 텐서면 그대로"""
    # given
    detail = {"dtype": np.float32, "quantization": (0.0, 0)}
    x = np.array([0.25], np.float32)

    # when
    y = quantize(x, detail)

    # then
    assert y.dtype == np.float32 and y[0] == 0.25


class FakeInterpreter:
    """입력 (1, 3, 320, 320) (NCHW), 출력 (1, 5, N) 인 YOLO 흉내. 받은 입력은 self.x"""

    def __init__(self, model_path, num_threads):
        pass

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"index": 0, "shape": np.array([1, 3, SIZE, SIZE]), "dtype": np.float32, "quantization": (0.0, 0)}]

    def get_output_details(self):
        return [{"index": 1, "shape": np.array([1, 5, 10]), "dtype": np.float32, "quantization": (0.0, 0)}]

    def set_tensor(self, index, x):
        self.x = x

    def invoke(self):
        pass

    def get_tensor(self, index):
        out = np.zeros((1, 5, 10), np.float32)
        out[0, :, 0] = (125 / SIZE, 125 / SIZE, 50 / SIZE, 70 / SIZE, 0.9)
        return out


def test_detector_feeds_nchw_input(monkeypatch, tmp_path):
    """입력이 채널 먼저 (1, 3, 320, 320) 인 모델에 맞는 모양으로 넣고, 손을 화면 좌표로 돌려줌"""
    # given
    monkeypatch.setattr(hand, "Interpreter", FakeInterpreter)
    path = tmp_path / "hand.tflite"
    path.touch()
    detector = HandDetector(path, num_hands=4, score_th=0.5, iou_th=0.45, num_threads=1)

    # when
    found = detector.detect(np.zeros((480, 640, 3), np.uint8))

    # then
    assert detector._it.x.shape == (1, 3, SIZE, SIZE)
    assert [(h.bbox, h.raised) for h in found] == [((200, 100, 100, 140), True)]
