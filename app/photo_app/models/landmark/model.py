"""FaceLandmarker + 손 YOLO 모델
- 얼굴: 랜드마크 최소/최대 좌표로 bbox
- 웃음, 눈 뜸: blendshape 룰 판정 (rules.py)
- 손 들기: 손바닥(하이파이브 손 모양) YOLO 검출 (util/hand.py). 키보드 h 도 보조로 허용
- 보임: 얼굴 랜드마크(눈꼬리, 코끝)의 고개 돌림 yaw < yaw_th. 보이지 않는 얼굴은 웃음, 눈 뜸 None
- FaceLandmarker 는 항상 1회 실행, 룰은 요청받은 결과만 계산
- hand 미요청이면 손 YOLO 호출 생략
- 실행 모드: IMAGE. 판정 스레드와 CHECKING 이 같은 인스턴스를 쓰므로 VIDEO 모드의 증가 타임스탬프 조건을 지킬 수 없음
"""
from __future__ import annotations

import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from ... import config
from . import config as landmark_config
from ...model import Face, Hand, Model, Prediction, requested
from . import rules
from ..util.hand import HandDetector
from ..util.keyboard import KeyboardHand

RIGHT_EYE = (33, 133)  # FaceLandmarker 랜드마크 번호: 오른눈 바깥, 안쪽 눈꼬리
LEFT_EYE = (362, 263)  # 왼눈 안쪽, 바깥 눈꼬리
NOSE_TIP = 1


class LandmarkModel(Model):
    def __init__(
        self,
        face_path=landmark_config.FACE_MODEL_PATH,
        hand_path=landmark_config.HAND_MODEL_PATH,
        num_faces: int = config.NUM_FACES,
        num_hands: int = landmark_config.NUM_HANDS,
        smile_th: float = landmark_config.SMILE_TH,
        blink_th: float = landmark_config.BLINK_TH,
        hand_score_th: float = landmark_config.HAND_SCORE_TH,
        hand_iou_th: float = landmark_config.HAND_IOU_TH,
        num_threads: int = landmark_config.NUM_THREADS,
        yaw_th: float = landmark_config.YAW_TH,
    ):
        if not face_path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {face_path}")
        self._faces = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(face_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=num_faces,
            output_face_blendshapes=True,
        ))
        self._hands = HandDetector(hand_path, num_hands, hand_score_th, hand_iou_th, num_threads)
        self._lock = threading.Lock()   # 판정 스레드와 메인 스레드(CHECKING)가 같이 호출
        self._key_hand = KeyboardHand()
        self._smile_th = smile_th
        self._blink_th = blink_th
        self._yaw_th = yaw_th

    def handle_key(self, key: str) -> None:
        self._key_hand.handle_key(key)

    def predict(self, frame, *, smile: bool, eye: bool, hand: bool, visibility: bool) -> Prediction:
        start = time.perf_counter()
        h, w = frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timings = {}
        with self._lock:
            t = time.perf_counter()
            face_result = self._faces.detect(image)
            timings["face"] = (time.perf_counter() - t) * 1000
            hands, raised = self._detect_hands(frame, timings) if hand else (None, None)

        faces = []
        smile_ms = eye_ms = 0.0
        for landmarks, blendshapes in zip(face_result.face_landmarks, face_result.face_blendshapes):
            bs = {c.category_name: c.score for c in blendshapes}
            yaw = _yaw(landmarks, w, h)
            vis = yaw < self._yaw_th if visibility else None
            scores = {}
            if visibility:
                scores["yaw"] = (yaw, self._yaw_th, vis)
            if smile and vis is not False:
                scores["smile"] = (rules.smile_score(bs), self._smile_th, rules.is_smiling(bs, self._smile_th))
            if eye and vis is not False:
                scores["blink"] = (rules.blink_score(bs), self._blink_th, rules.is_eyes_open(bs, self._blink_th))
            t = time.perf_counter()
            is_smiling = rules.is_smiling(bs, self._smile_th) if smile and vis is not False else None
            t2 = time.perf_counter()
            is_open = rules.is_eyes_open(bs, self._blink_th) if eye and vis is not False else None
            smile_ms += (t2 - t) * 1000
            eye_ms += (time.perf_counter() - t2) * 1000
            eye_score = 1 - rules.blink_score(bs) if eye and vis is not False else None
            faces.append(Face(landmark_bbox(landmarks, w, h), is_smiling, is_open, vis, eye_score, scores=scores))
        if smile:
            timings["smile"] = smile_ms
        if eye:
            timings["eye"] = eye_ms

        return Prediction(faces, hands, raised, (time.perf_counter() - start) * 1000, requested(smile, eye, hand, visibility), timings)

    def _detect_hands(self, frame, timings: dict) -> tuple[list[Hand], bool]:
        """전체 화면 손 YOLO (BGR 프레임) -> (손 목록, 손 들기 여부). 잠금 안에서 호출"""
        t = time.perf_counter()
        hands = self._hands.detect(frame)
        timings["hand"] = (time.perf_counter() - t) * 1000
        return hands, any(h.raised for h in hands) or self._key_hand.raised()


def _yaw(landmarks, w: int, h: int) -> float:
    """정규화 얼굴 랜드마크 -> 고개 돌림 yaw (눈 가운데는 눈꼬리 두 점의 평균)"""
    def point(ids):
        return (sum(landmarks[i].x for i in ids) / len(ids) * w, sum(landmarks[i].y for i in ids) / len(ids) * h)
    return rules.yaw(point(RIGHT_EYE), point(LEFT_EYE), point((NOSE_TIP,)))


def landmark_bbox(landmarks, w: int, h: int) -> tuple[int, int, int, int]:
    """정규화 랜드마크(0~1) -> 픽셀 bbox (x, y, w, h). 화면 밖은 잘라냄"""
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    x1, y1 = max(int(min(xs) * w), 0), max(int(min(ys) * h), 0)
    x2, y2 = min(int(max(xs) * w), w), min(int(max(ys) * h), h)
    return x1, y1, x2 - x1, y2 - y1
