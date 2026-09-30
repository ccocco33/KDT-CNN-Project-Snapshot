"""YuNet + CNN(smile, eye, occlusion) 모델
- 얼굴: YuNet (신뢰도 SCORE_TH 이상)
- 보임 (visibility 요청 시): 고개 돌림 yaw < YAW_TH 이고 가림 점수 < OCC_TH
  - yaw: YuNet 랜드마크(두 눈, 코). yaw 로 이미 보이지 않으면 가림 CNN 은 실행하지 않음
  - 가림: occlusion CNN (입이 안 보이거나 두 눈이 모두 안 보임. 한쪽 눈만 가림, 선글라스류는 보임)
  - 보이지 않는 얼굴은 smile, eye CNN 을 실행하지 않음 (웃음, 눈 뜸 None)
- 웃음, 눈 뜸: 얼굴마다 CNN (LiteRT). 요청받은 결과의 CNN 만 실행
  - 입력: YuNet 박스 중심 정사각형 (한 변 = 박스 높이, 화면 밖으로 나가면 안쪽으로 밀어 넣음. 학습 전처리와 같음), RGB 0~255
  - occlusion: 정사각형 전체를 OCC_SIZE 로 / smile: 정사각형 아래쪽 SMILE_BOTTOM 을 SMILE_SIZE 로 / eye: 정사각형 위쪽 EYE_TOP 을 EYE_SIZE 로 늘림
  - smile 은 눈 영역을 넣지 않음 (선글라스 렌즈 반사를 웃음으로 보는 문제)
  - 좌우 뒤집기 평균은 모델 안에 포함
  - 선글라스류는 eye CNN 이 뜸으로 판정 (학습에 포함)
- 손: 전체 화면 손바닥 YOLO (util/hand.py). 키보드 h 도 보조로 허용
- timings: face (YuNet, 얼굴 자르기 포함), occlusion, smile, eye (CNN 실행 합계), hand
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

from . import config
from ...model import Face, Model, Prediction, requested
from ..landmark import rules
from ..util.hand import HandDetector
from ..util.keyboard import KeyboardHand


class YunetCnnModel(Model):
    def __init__(self):
        for path in (config.YUNET_PATH, config.SMILE_PATH, config.EYE_PATH, config.OCC_PATH, config.HAND_PATH):
            if not path.exists():
                raise FileNotFoundError(f"모델 파일 없음: {path}")
        self._detector = cv2.FaceDetectorYN.create(str(config.YUNET_PATH), "", (320, 320), score_threshold=config.SCORE_TH)
        self._smile = Classifier(config.SMILE_PATH, config.SMILE_SIZE, config.NUM_THREADS)
        self._eye = Classifier(config.EYE_PATH, config.EYE_SIZE, config.NUM_THREADS)
        self._occ = Classifier(config.OCC_PATH, config.OCC_SIZE, config.NUM_THREADS)
        self._hands = HandDetector(config.HAND_PATH, config.NUM_HANDS, config.HAND_SCORE_TH, config.HAND_IOU_TH, config.NUM_THREADS)
        self._key_hand = KeyboardHand()
        self._lock = threading.Lock()   # 판정 스레드와 메인 스레드(CHECKING)가 같이 호출

    def handle_key(self, key: str) -> None:
        self._key_hand.handle_key(key)

    def predict(self, frame, *, smile: bool, eye: bool, hand: bool, visibility: bool) -> Prediction:
        start = time.perf_counter()
        timings = {}
        faces = []
        with self._lock:
            t = time.perf_counter()
            found = self._detect_faces(frame)
            crops = [square_crop(frame, raw) if (smile or eye or visibility) else None for _, raw, _ in found]
            timings["face"] = (time.perf_counter() - t) * 1000
            occ_ms = smile_ms = eye_ms = 0.0
            for (box, _, yaw), crop in zip(found, crops):
                scores = {}
                vis = yaw < config.YAW_TH if visibility else None
                if visibility:
                    scores["yaw"] = (yaw, config.YAW_TH, vis)
                occ = None
                if vis:
                    t = time.perf_counter()
                    occ = self._occ(crop)
                    vis = occ < config.OCC_TH
                    scores["occ"] = (occ, config.OCC_TH, vis)
                    occ_ms += (time.perf_counter() - t) * 1000

                judged = vis is not False   # 보이지 않는 얼굴은 웃음, 눈 판정 안 함
                is_smiling = is_open = eye_score = None

                # 웃음 판정
                if smile and judged:
                    t = time.perf_counter()
                    smile_score = self._smile(lower(crop, config.SMILE_BOTTOM))
                    is_smiling = smile_score >= config.SMILE_TH
                    scores["smile"] = (smile_score, config.SMILE_TH, is_smiling)
                    smile_ms += (time.perf_counter() - t) * 1000
                    
                # 눈 판정
                if eye and judged:
                    t = time.perf_counter()
                    eye_score = self._eye(upper(crop, config.EYE_TOP))
                    is_open = eye_score >= config.EYE_TH
                    scores["eye"] = (eye_score, config.EYE_TH, is_open)
                    eye_ms += (time.perf_counter() - t) * 1000
                faces.append(Face(box, is_smiling, is_open, vis, eye_score, occ, scores))

            if visibility:
                timings["occlusion"] = occ_ms
            if smile:
                timings["smile"] = smile_ms
            if eye:
                timings["eye"] = eye_ms
            hands, raised = (None, None)

            # 손 디텍트
            if hand:
                t = time.perf_counter()
                hands = self._hands.detect(frame)
                timings["hand"] = (time.perf_counter() - t) * 1000
                raised = any(x.raised for x in hands) or self._key_hand.raised()

        return Prediction(faces, hands, raised, (time.perf_counter() - start) * 1000,
                          requested(smile, eye, hand, visibility), timings)

    def _detect_faces(self, frame) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int], float]]:
        """YuNet 얼굴 [(bbox, 원본 박스, yaw), ...]
        - bbox: 화면 밖을 잘라낸 박스 (Face.bbox). 원본 박스: 자르지 않은 박스 (CNN 입력 정사각형 계산용)
        """
        h, w = frame.shape[:2]
        target_w = config.YUNET_INPUT_WIDTH
        if target_w <= 0:
            raise ValueError("YUNET_INPUT_WIDTH는 1 이상이어야 합니다.")

        scale = target_w / w
        if target_w == w:
            detect_frame = frame
        else:
            target_h = max(1, round(h * scale))
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            detect_frame = cv2.resize(
                frame,
                (target_w, target_h),
                interpolation=interpolation,
            )

        detect_h, detect_w = detect_frame.shape[:2]
        self._detector.setInputSize((detect_w, detect_h))
        _, found = self._detector.detect(detect_frame)
        if found is None:
            return []

        # 박스와 5개 랜드마크 좌표를 원본 화면 좌표로 되돌립니다.
        if scale != 1.0:
            found = found.copy()
            found[:, :14] /= scale

        out = []
        for f in found:
            raw = tuple(int(v) for v in f[:4])
            x1, y1 = max(raw[0], 0), max(raw[1], 0)
            x2, y2 = min(raw[0] + raw[2], w), min(raw[1] + raw[3], h)
            yaw = rules.yaw((f[4], f[5]), (f[6], f[7]), (f[8], f[9]))   # YuNet 랜드마크: 오른눈, 왼눈, 코
            out.append(((x1, y1, x2 - x1, y2 - y1), raw, yaw))
        return out


class Classifier:
    """LiteRT 이진 분류 CNN 1개. 입력 RGB 0~255 (size x size), 출력 점수 0~1"""

    def __init__(self, path, size: int, num_threads: int):
        self._it = Interpreter(model_path=str(path), num_threads=num_threads)
        self._it.allocate_tensors()
        self._in = self._it.get_input_details()[0]["index"]
        self._out = self._it.get_output_details()[0]["index"]
        self.size = size

    def __call__(self, rgb) -> float:
        x = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype(np.float32)[None]
        self._it.set_tensor(self._in, x)
        self._it.invoke()
        return float(self._it.get_tensor(self._out)[0, 0])


def square_crop(frame, box) -> np.ndarray:
    """박스 중심 정사각형 (RGB). 학습 데이터 전처리와 같은 방식 (training/data/*/preprocessed/preprocess.py 의 square_crop_inside)
    - 한 변 = 박스 높이 (프레임 너비, 높이 이하)
    - 프레임 밖으로 나가면 안쪽으로 밀어 넣음 (채우는 영역 없음)
    - frame: BGR, box: 자르지 않은 YuNet 박스 (x, y, w, h)
    """
    x, y, w, h = box
    H, W = frame.shape[:2]
    side = max(1, min(int(round(h)), W, H))
    x1 = min(max(int(round(x + w / 2 - side / 2)), 0), W - side)
    y1 = min(max(int(round(y + h / 2 - side / 2)), 0), H - side)
    return cv2.cvtColor(frame[y1:y1 + side, x1:x1 + side], cv2.COLOR_BGR2RGB)


def upper(face, ratio: float) -> np.ndarray:
    """얼굴 정사각형의 위쪽 ratio (eye CNN 입력)"""
    return face[:max(1, int(face.shape[0] * ratio))]


def lower(face, ratio: float) -> np.ndarray:
    """얼굴 정사각형의 아래쪽 ratio (smile CNN 입력)"""
    return face[face.shape[0] - max(1, int(round(face.shape[0] * ratio))):]
