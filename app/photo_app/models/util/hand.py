"""손 검출과 손 들기 판정 (landmark, yunet_landmark, yunet_cnn 공통)
- 전체 화면 YOLO (hand_yolo.tflite, LiteRT) -> 손바닥(palm) bbox 목록
  - 모델: Ultralytics YOLO11n, 클래스 1개 (palm), 입력 (1, 3, 320, 320) RGB 0~1 (채널 먼저, NCHW), 출력 (1, 5, N) = 중심 x, y, 너비, 높이 (0~1), 점수
  - 입력이 (1, 320, 320, 3) (NHWC) 인 모델도 받음
  - end2end 아님 -> NMS 는 여기서 (cv2.dnn.NMSBoxes)
- 손 들기: 손바닥(하이파이브 손 모양)이 점수 score_th 이상으로 검출되면 손 들기. pose 는 "palm 점수"
- 입력: 화면 비율을 유지해 줄이고 남는 곳은 회색(114)으로 채움 (letterbox, Ultralytics 학습과 같음)
- 설정값(모델 파일, 최대 손 수, 점수, NMS IoU, 스레드 수)은 각 모델이 자기 config 값을 인자로 넘김
- 호출하는 모델의 잠금 안에서 사용 (Interpreter 는 동시 호출 불가)
"""
from __future__ import annotations

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

from ...model import Hand

PAD_VALUE = 114   # letterbox 빈 곳 색 (Ultralytics 기본값)


class HandDetector:
    def __init__(self, path, num_hands: int, score_th: float, iou_th: float, num_threads: int):
        if not path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {path}")
        self._it = Interpreter(model_path=str(path), num_threads=num_threads)
        self._it.allocate_tensors()
        self._in = self._it.get_input_details()[0]
        self._out = self._it.get_output_details()[0]
        shape = self._in["shape"]
        self._nchw = shape[1] == 3              # (1, 3, size, size) 이면 채널 먼저
        self.size = int(shape[2] if self._nchw else shape[1])
        self.num_hands = num_hands
        self.score_th = score_th
        self.iou_th = iou_th

    def detect(self, frame) -> list[Hand]:
        """BGR 프레임 -> 손 목록 (bbox 는 frame 픽셀 좌표)"""
        h, w = frame.shape[:2]
        img, scale, pad = letterbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.size)
        x = img.astype(np.float32)[None] / 255
        if self._nchw:
            x = x.transpose(0, 3, 1, 2)
        self._it.set_tensor(self._in["index"], quantize(np.ascontiguousarray(x), self._in))
        self._it.invoke()
        pred = dequantize(self._it.get_tensor(self._out["index"]), self._out)[0]
        found = decode(pred, self.size, scale, pad, w, h, self.score_th, self.iou_th, self.num_hands)
        return [Hand(box, True, f"palm {score:.2f}") for box, score in found]


def letterbox(rgb, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """비율 유지 축소 + 가운데 배치 (size x size). -> (이미지, 배율, 왼쪽/위쪽 여백)"""
    h, w = rgb.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    left, top = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), PAD_VALUE, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, scale, (left, top)


def decode(pred, size: int, scale: float, pad: tuple[int, int], w: int, h: int,
           score_th: float, iou_th: float, max_hands: int) -> list[tuple[tuple[int, int, int, int], float]]:
    """YOLO 출력 (5, N) 또는 (N, 5) -> [(bbox (x, y, w, h) frame 픽셀, 점수), ...] 점수 높은 순
    - 좌표는 입력 크기 기준 0~1 (Ultralytics LiteRT export). 1 을 넘는 값이 있으면 입력 픽셀로 봄
    - score_th 미만 버림, NMS 후 max_hands 개까지. 화면 밖은 잘라냄
    """
    pred = np.asarray(pred, np.float32)
    if pred.shape[0] != 5:
        pred = pred.T
    keep = pred[4] >= score_th
    cx, cy, bw, bh, scores = pred[:, keep]
    if cx.size == 0:
        return []
    if max(cx.max(), cy.max(), bw.max(), bh.max()) <= 1.5:
        cx, cy, bw, bh = cx * size, cy * size, bw * size, bh * size
    x1 = (cx - bw / 2 - pad[0]) / scale
    y1 = (cy - bh / 2 - pad[1]) / scale
    boxes = np.stack([x1, y1, bw / scale, bh / scale], axis=1)
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_th, iou_th)
    out = []
    for i in np.array(idx).flatten()[:max_hands]:
        bx, by, bw_, bh_ = boxes[i]
        x1_, y1_ = max(int(bx), 0), max(int(by), 0)
        x2_, y2_ = min(int(bx + bw_), w), min(int(by + bh_), h)
        if x2_ > x1_ and y2_ > y1_:
            out.append(((x1_, y1_, x2_ - x1_, y2_ - y1_), float(scores[i])))
    return out


def quantize(x, detail) -> np.ndarray:
    """float 입력 -> 텐서 dtype (int8 모델이면 scale, zero point 로 양자화)"""
    scale, zero = detail["quantization"]
    if detail["dtype"] == np.float32 or not scale:
        return x.astype(detail["dtype"])
    info = np.iinfo(detail["dtype"])
    return np.clip(np.round(x / scale + zero), info.min, info.max).astype(detail["dtype"])


def dequantize(y, detail) -> np.ndarray:
    """텐서 출력 -> float (int8 모델이면 scale, zero point 로 되돌림)"""
    scale, zero = detail["quantization"]
    if detail["dtype"] == np.float32 or not scale:
        return y.astype(np.float32)
    return (y.astype(np.float32) - zero) * scale
