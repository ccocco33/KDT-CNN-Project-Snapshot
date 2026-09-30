"""눈 뜬 정도 기하 지표 (FaceLandmarker 랜드마크와 얼굴 변환 행렬만. blendshape, 손으로 고른 상수 없음)
- 입력: ../preprocessed/{opened,closed}/*.png. 192x192 로 키우고 둘레 32px 검은 여백 후 FaceLandmarker
- 좌표: 두 눈 중심을 이은 선이 수평이 되도록 돌린 뒤 계산 (고개 기울기 roll 제거). y 는 아래로 증가
- 지표 (눈은 두 눈 평균. 거리는 눈 또는 입 가로 길이로 나눔)
  - eye_open: 눈 열림. 위 눈꺼풀 중앙 ~ 아래 눈꺼풀 중앙 거리 / 눈 가로 길이
  - upper_arch: 눈 모양 (위). 두 눈꼬리 중점에서 위 눈꺼풀 중앙이 위로 떨어진 높이 / 눈 가로 길이. 클수록 위 눈꺼풀이 둥글게 휨
  - lower_arch: 눈 모양 (아래). 두 눈꼬리 중점에서 아래 눈꺼풀 중앙이 위로 떨어진 높이 / 눈 가로 길이. 클수록 아래 눈꺼풀이 밀려 올라감 (^^)
  - pitch: 아래봄. 얼굴 변환 행렬의 위아래 회전 각도(도). 부호는 측정 후 확인 (숙이면 커지도록 맞춤)
  - mouth_lift: 입. 입술 중앙(윗입술 안쪽, 아랫입술 안쪽 평균)보다 두 입꼬리가 위에 있는 높이 / 입 가로 길이. 클수록 입꼬리가 올라감
- FaceLandmarker 랜드마크 번호
  - 오른눈(이미지 왼쪽): 눈꼬리 33, 133 / 위 눈꺼풀 159 / 아래 눈꺼풀 145 / 왼눈: 눈꼬리 362, 263 / 위 386 / 아래 374
  - 입꼬리 61, 291 / 윗입술 안쪽 13 / 아랫입술 안쪽 14
- 출력: ./geometry_metrics.csv (folder, file, level, 지표들), 터미널: 단계별 분포, 이웃 단계 AUC

실행: app/.venv/bin/python training/data/eye/levels/geometry_metrics.py
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[2]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "eye" / "levels"   # 데이터 위치 (training/paths.py)
EYE = HERE.parent
MODEL = REPO / "app/models/face_landmarker.task"
OUT = HERE / "geometry_metrics.csv"
UPSCALE, PAD = 192, 32
EYES = ((33, 133, 159, 145), (362, 263, 386, 374))   # (눈꼬리, 눈꼬리, 위 눈꺼풀 중앙, 아래 눈꺼풀 중앙)
MOUTH = (61, 291, 13, 14)                           # (입꼬리, 입꼬리, 윗입술 안쪽, 아랫입술 안쪽)
METRICS = ("eye_open", "upper_arch", "lower_arch", "pitch", "mouth_lift")
NAMES = {1: "1 뜸", 2: "2 웃는 눈", 3: "3 감기는 눈", 4: "4 감은 눈"}


def pitch_deg(matrix):
    """얼굴 변환 행렬(4x4) -> 위아래 회전 각도(도). 회전 행렬 R 의 x 축 회전"""
    r = np.asarray(matrix)[:3, :3]
    return math.degrees(math.atan2(r[2, 1], r[2, 2]))


def metrics_from(landmarks, matrix, w, h):
    """FaceLandmarker 결과 1개 -> 지표 dict. landmarks: 정규화 좌표, w, h: 입력 이미지 크기"""
    p = np.array([[q.x * w, q.y * h] for q in landmarks])
    centers = [(p[a] + p[b]) / 2 for a, b, _, _ in EYES]
    angle = math.atan2(centers[1][1] - centers[0][1], centers[1][0] - centers[0][0])
    rot = np.array([[math.cos(-angle), -math.sin(-angle)], [math.sin(-angle), math.cos(-angle)]])
    p = p @ rot.T   # 두 눈 중심 선을 수평으로
    opens, uppers, lowers = [], [], []
    for a, b, top, bottom in EYES:
        width = np.linalg.norm(p[a] - p[b]) or 1e-6
        mid_y = (p[a][1] + p[b][1]) / 2
        opens.append(np.linalg.norm(p[top] - p[bottom]) / width)
        uppers.append((mid_y - p[top][1]) / width)
        lowers.append((mid_y - p[bottom][1]) / width)
    a, b, up, low = MOUTH
    mouth_w = np.linalg.norm(p[a] - p[b]) or 1e-6
    lift = ((p[up][1] + p[low][1]) / 2 - (p[a][1] + p[b][1]) / 2) / mouth_w
    return {"eye_open": float(np.mean(opens)), "upper_arch": float(np.mean(uppers)),
            "lower_arch": float(np.mean(lowers)), "pitch": pitch_deg(matrix), "mouth_lift": float(lift)}


def make_landmarker():
    from mediapipe.tasks.python import BaseOptions, vision
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.IMAGE,
        num_faces=1, output_facial_transformation_matrixes=True))


def measure_rgb(landmarker, rgb):
    """RGB 얼굴 이미지 -> 지표 dict 또는 None"""
    import mediapipe as mp
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.face_landmarks:
        return None
    return metrics_from(res.face_landmarks[0], res.facial_transformation_matrixes[0], rgb.shape[1], rgb.shape[0])


def measure():
    import cv2
    lm = make_landmarker()
    rows = []
    for r in csv.DictReader(open(HERE / "levels.csv")):
        img = cv2.cvtColor(cv2.imread(str(EYE / "preprocessed" / r["folder"] / r["file"])), cv2.COLOR_BGR2RGB)
        big = cv2.copyMakeBorder(cv2.resize(img, (UPSCALE, UPSCALE), interpolation=cv2.INTER_CUBIC),
                                 PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT)
        m = measure_rgb(lm, big)
        rows.append({"folder": r["folder"], "file": r["file"], "level": r["level"],
                     **({k: round(m[k], 4) for k in METRICS} if m else {k: "" for k in METRICS})})
    with open(OUT, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def auc(pos, neg):
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(np.mean(pos > neg) + 0.5 * np.mean(pos == neg))


def main():
    if not OUT.exists():
        measure()
    rows = [r for r in csv.DictReader(open(OUT)) if r["eye_open"] != ""]
    lv = np.array([int(r["level"]) for r in rows])
    print(f"지표 {len(rows)}장. 단계별 중앙값 (괄호: 25%~75%)")
    print(f"  {'지표':<11}" + "".join(f"{NAMES[k]:>22}" for k in (1, 2, 3, 4)))
    for m in METRICS:
        v = np.array([float(r[m]) for r in rows])
        cells = [f"{np.median(v[lv == k]):.3f} ({np.percentile(v[lv == k], 25):.2f}~{np.percentile(v[lv == k], 75):.2f})"
                 for k in (1, 2, 3, 4)]
        print(f"  {m:<11}" + "".join(f"{c:>22}" for c in cells))
    print("이웃 단계 AUC (앞 단계 값이 클 확률. 0.5 는 구분 못 함, 0 또는 1 에 가까울수록 잘 구분)")
    for m in METRICS:
        v = np.array([float(r[m]) for r in rows])
        print(f"  {m:<11}" + "".join(f"  {a}>{a + 1}: {auc(v[lv == a], v[lv == a + 1]):.2f}" for a in (1, 2, 3)))


if __name__ == "__main__":
    main()
