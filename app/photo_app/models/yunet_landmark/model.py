"""YuNet + FaceLandmarker 2단계 모델
- 1단계: YuNet 으로 전체 화면에서 얼굴 bbox 검출 (신뢰도 SCORE_TH 이상)
- 2단계: 얼굴마다 CROP_MARGIN 여백을 두고 잘라 FaceLandmarker 실행 -> blendshape 룰로 웃음, 눈 뜸 판정
  - 전체 화면을 FaceLandmarker 에 넣으면 작은 얼굴을 못 찾음. 잘라 넣으면 얼굴이 화면을 채워 찾음
  - 잘라낸 얼굴에서 FaceLandmarker 가 얼굴을 못 찾으면 웃음, 눈 뜸은 None (판정 불가 -> 조건 미충족)
  - smile, eye 미요청이면 2단계 생략 (bbox 만 반환)
- 보임: YuNet 랜드마크(두 눈, 코)의 고개 돌림 yaw < YAW_TH. 추가 모델 실행 없음
  - 보이지 않는 얼굴은 2단계 생략 (웃음, 눈 뜸은 None)
- 얼굴 bbox 는 YuNet 결과
- 손: landmark 모델과 같음 (전체 화면 손바닥 YOLO)
"""
from __future__ import annotations

import time

import cv2
import mediapipe as mp

from . import config
from ...model import Face, Prediction, requested
from ..landmark import LandmarkModel, rules


class YunetLandmarkModel(LandmarkModel):
    def __init__(self, yunet_path=config.YUNET_PATH, crop_margin: float = config.CROP_MARGIN, score_th: float = config.SCORE_TH):
        if not yunet_path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {yunet_path}")
        super().__init__(
            face_path=config.FACE_MODEL_PATH,
            hand_path=config.HAND_MODEL_PATH,
            num_faces=1,   # 잘라낸 얼굴 1개씩 판정
            num_hands=config.NUM_HANDS,
            smile_th=config.SMILE_TH,
            blink_th=config.BLINK_TH,
            hand_score_th=config.HAND_SCORE_TH,
            hand_iou_th=config.HAND_IOU_TH,
            num_threads=config.NUM_THREADS,
            yaw_th=config.YAW_TH,
        )
        self._detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), score_threshold=score_th)
        self._margin = crop_margin

    def predict(self, frame, *, smile: bool, eye: bool, hand: bool, visibility: bool) -> Prediction:
        start = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timings = {}
        with self._lock:
            t = time.perf_counter()
            found = self._detect_faces(frame)
            timings["face"] = (time.perf_counter() - t) * 1000
            # 보임: yaw 가 기준 미만. visibility 미요청이면 None (모든 얼굴 판정)
            visible = [yaw < self._yaw_th if visibility else None for _, yaw in found]
            blendshapes = [None] * len(found)
            if smile or eye:
                t = time.perf_counter()
                blendshapes = [self._blendshapes(rgb, box) if vis is not False else None
                               for (box, _), vis in zip(found, visible)]
                timings["landmark"] = (time.perf_counter() - t) * 1000
            hands, raised = (None, None)
            if hand:
                hands, raised = self._detect_hands(frame, timings)

        faces = []
        smile_ms = eye_ms = 0.0
        for (box, yaw), vis, bs in zip(found, visible, blendshapes):
            scores = {}
            if visibility:
                scores["yaw"] = (yaw, self._yaw_th, vis)
            if smile and bs:
                scores["smile"] = (rules.smile_score(bs), self._smile_th, rules.is_smiling(bs, self._smile_th))
            if eye and bs:
                scores["blink"] = (rules.blink_score(bs), self._blink_th, rules.is_eyes_open(bs, self._blink_th))
            t = time.perf_counter()
            is_smiling = rules.is_smiling(bs, self._smile_th) if smile and bs else None
            t2 = time.perf_counter()
            is_open = rules.is_eyes_open(bs, self._blink_th) if eye and bs else None
            smile_ms += (t2 - t) * 1000
            eye_ms += (time.perf_counter() - t2) * 1000
            eye_score = 1 - rules.blink_score(bs) if eye and bs else None
            faces.append(Face(box, is_smiling, is_open, vis, eye_score, scores=scores))
        if smile:
            timings["smile"] = smile_ms
        if eye:
            timings["eye"] = eye_ms

        return Prediction(faces, hands, raised, (time.perf_counter() - start) * 1000,
                          requested(smile, eye, hand, visibility), timings)

    def _detect_faces(self, frame) -> list[tuple[tuple[int, int, int, int], float]]:
        """YuNet 얼굴 [(bbox (x, y, w, h), 고개 돌림 yaw), ...]. bbox 의 화면 밖은 잘라냄"""
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, found = self._detector.detect(frame)
        if found is None:
            return []
        out = []
        for f in found:
            x1, y1 = max(int(f[0]), 0), max(int(f[1]), 0)
            x2, y2 = min(int(f[0] + f[2]), w), min(int(f[1] + f[3]), h)
            yaw = rules.yaw((f[4], f[5]), (f[6], f[7]), (f[8], f[9]))   # YuNet 랜드마크: 오른눈, 왼눈, 코
            out.append(((x1, y1, x2 - x1, y2 - y1), yaw))
        return out

    def _blendshapes(self, rgb, box) -> dict[str, float] | None:
        """여백을 두고 잘라낸 얼굴의 blendshape {이름: 점수}. FaceLandmarker 가 못 찾으면 None"""
        h, w = rgb.shape[:2]
        x, y, bw, bh = box
        mx, my = bw * self._margin, bh * self._margin
        x1, y1 = max(int(x - mx), 0), max(int(y - my), 0)
        x2, y2 = min(int(x + bw + mx), w), min(int(y + bh + my), h)
        crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb[y1:y2, x1:x2].copy())
        result = self._faces.detect(crop)
        if not result.face_blendshapes:
            return None
        return {c.category_name: c.score for c in result.face_blendshapes[0]}
