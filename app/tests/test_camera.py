import cv2
import numpy as np

from photo_app import config
from photo_app.camera import Camera

FRAME = np.zeros((4, 4, 3), np.uint8)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeCap:
    """cv2.VideoCapture 흉내. frames 가 True 면 프레임을 줌"""

    def __init__(self, opened=True, frames=True):
        self.opened, self.frames = opened, frames
        self.props = []   # set 호출 순서 [(prop, value), ...]

    def isOpened(self):
        return self.opened

    def read(self):
        return (True, FRAME) if self.frames else (False, None)

    def set(self, prop, value):
        self.props.append((prop, value))
        return True

    def get(self, *_):
        return 0

    def release(self):
        self.opened = False


class Opener:
    """호출마다 caps 를 차례로 돌려줌 (마지막은 반복). 호출 수를 셈"""

    def __init__(self, *caps):
        self.caps, self.calls = list(caps), 0

    def __call__(self, index):
        self.calls += 1
        return self.caps.pop(0) if len(self.caps) > 1 else self.caps[0]


def make(opener, clock):
    return Camera(0, opener=opener, now=clock, lost_sec=2.0, retry_sec=1.0)


def test_frame_arrives_when_connected():
    """카메라가 열리면 최신 프레임을 보관하고 끊김이 아님"""
    # given
    clock = Clock()
    cam = make(Opener(FakeCap()), clock)

    # when
    cam._step()

    # then
    frame, _ = cam.latest()
    assert frame is not None and not cam.lost


def test_open_requests_fourcc_before_size():
    """카메라를 열 때 픽셀 포맷(CAMERA_FOURCC)을 해상도보다 먼저 요청"""
    # given
    cap = FakeCap()

    # when
    make(Opener(cap), Clock())

    # then
    props = [p for p, _ in cap.props]
    assert (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC)) in cap.props
    assert props.index(cv2.CAP_PROP_FOURCC) < props.index(cv2.CAP_PROP_FRAME_WIDTH)


def test_no_camera_at_start_is_lost_after_wait():
    """시작할 때 카메라가 없어도 멈추지 않고, CAMERA_LOST_SEC 뒤에 끊김"""
    # given
    clock = Clock()
    cam = make(Opener(FakeCap(opened=False)), clock)

    # when
    cam._step()
    before = cam.lost
    clock.t = 2.0
    cam._step()

    # then
    assert (before, cam.lost) == (False, True)


def test_lost_clears_frame_and_retries_open():
    """프레임이 끊기면 최신 프레임을 비우고 CAMERA_RETRY_SEC 마다 다시 엶"""
    # given
    clock = Clock()
    cap = FakeCap()
    opener = Opener(cap)
    cam = make(opener, clock)
    cam._step()
    cap.frames = False

    # when
    clock.t = 2.0
    cam._step()
    clock.t = 2.5
    cam._step()
    clock.t = 3.0
    cam._step()

    # then
    assert cam.lost and cam.latest()[0] is None
    assert opener.calls == 3   # 시작 1 + 2.0초 1 + 3.0초 1 (2.5초는 간격 전이라 안 엶)


def test_restored_after_reopen():
    """다시 연 카메라에서 프레임이 오면 끊김이 풀림"""
    # given
    clock = Clock()
    cam = make(Opener(FakeCap(frames=False), FakeCap()), clock)
    clock.t = 2.0
    cam._step()   # 끊김 -> 다시 엶 (프레임이 오는 카메라)

    # when
    clock.t = 2.1
    cam._step()

    # then
    assert not cam.lost and cam.latest()[0] is not None


def test_recording_collects_frames():
    """촬영 중에는 프레임을 전부 누적"""
    # given
    clock = Clock()
    cam = make(Opener(FakeCap()), clock)
    cam.start_recording()

    # when
    for t in (0.1, 0.2, 0.3):
        clock.t = t
        cam._step()

    # then
    assert [ts for ts, _ in cam.stop_recording()] == [0.1, 0.2, 0.3]
