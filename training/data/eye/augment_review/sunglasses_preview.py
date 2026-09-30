"""eye 증강 검토: 합성 선글라스
- 1단계: 전처리 이미지(../../preprocessed/{opened,closed}/*.png, 64x64)의 두 눈 좌표를 YuNet 으로 구해 저장
  - 출력: ../../landmarks/eye_landmarks.csv (class, file, rx, ry, lx, ly. 64px 좌표. 미검출이면 빈칸)
  - 학습 노트북의 선글라스 합성 증강이 이 파일을 씀
- 2단계: 합성 선글라스 미리보기
  - 두 눈 위에 어두운 렌즈, 테, 코 받침을 그림. 두 눈 각도만큼 기울임
  - 렌즈 모양(타원, 둥근 사각형, 보잉, 캣아이), 크기, 색, 불투명도는 무작위 (add_sunglasses 의 범위)
  - 합성한 이미지는 라벨과 상관없이 "뜸" (요구사항: 선글라스류로 눈이 보이지 않으면 눈 조건 충족)
  - 감은 눈 이미지는 렌즈 불투명도 0.9 이상 (감은 눈이 비치면 "뜸" 라벨과 모순). 옅은 렌즈는 뜬 눈에만
- 출력: ./preview.png (1줄 실제 선글라스, 2줄 뜬 눈 합성, 3줄 감은 눈 합성, 4줄 그늘진 감은 눈)

실행: app/.venv/bin/python training/data/eye/augment_review/sunglasses_preview.py
"""
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[2]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "eye" / "augment_review"   # 데이터 위치 (training/paths.py)
EYE = HERE.parent
PRE = EYE / "preprocessed"
LANDMARKS = EYE / "landmarks" / "eye_landmarks.csv"
YUNET = REPO / "app/models/face_detection_yunet_2023mar.onnx"

SIZE = 256   # 검출용 확대 크기
PAD = 64     # 검출용 둘레 여백
REAL_SUNGLASSES = ["01450", "00184", "00873", "01114", "01484", "00720", "00265", "00786", "00951",
                   "01142", "00824", "00836", "00257"]
SHADOW_CLOSED = ['closed_eye_1859.jpg_face_1','closed_eye_2111.jpg_face_1','closed_eye_2536.BMP_face_1','closed_eye_0014.jpg_face_2']
SEED = 0


def find_landmarks():
    """전처리 이미지마다 두 눈 좌표 (64px 기준)"""
    yunet = cv2.FaceDetectorYN.create(str(YUNET), "", (SIZE + 2 * PAD,) * 2, score_threshold=0.5)
    rows = []
    for cls in ("opened", "closed"):
        for path in sorted((PRE / cls).glob("*.png")):
            img = cv2.imread(str(path))
            k = SIZE / img.shape[1]
            big = cv2.copyMakeBorder(cv2.resize(img, (SIZE, SIZE)), PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT)
            _, det = yunet.detect(big)
            row = {"class": cls, "file": path.name, "rx": "", "ry": "", "lx": "", "ly": ""}
            if det is not None:
                d = max(det, key=lambda r: r[2] * r[3])
                row.update({k2: round(float((d[i] - PAD) / k), 2) for k2, i in (("rx", 4), ("ry", 5), ("lx", 6), ("ly", 7))})
            rows.append(row)
    LANDMARKS.parent.mkdir(exist_ok=True)
    with open(LANDMARKS, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


LENS_SHAPES = ("ellipse", "rounded_rect", "aviator", "cat_eye")


def lens_polygon(cx, cy, a, b, angle_deg, shape, side, n=48):
    """렌즈 외곽 좌표 (정수 배열)
    - a, b: 반폭, 반높이. side: -1 오른눈(이미지 왼쪽), +1 왼눈(이미지 오른쪽). 캣아이의 바깥쪽 방향
    - ellipse: 타원 / rounded_rect: 초타원(n=4) / aviator: 아래가 넓은 물방울 / cat_eye: 바깥쪽 위가 올라감
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    if shape == "rounded_rect":
        x, y = a * np.sign(c) * np.abs(c) ** 0.5, b * np.sign(s) * np.abs(s) ** 0.5
    elif shape == "aviator":
        x, y = a * c * (1 + 0.18 * s), b * s   # 아래(s > 0)가 넓음
        y = np.where(s > 0, y * 1.15, y)
    elif shape == "cat_eye":
        x, y = a * c, b * s
        y = y - 0.35 * b * np.clip(c * side, 0, None) * (s < 0)   # 바깥쪽 위를 끌어올림
    else:
        x, y = a * c, b * s
    r = np.deg2rad(angle_deg)
    xr, yr = x * np.cos(r) - y * np.sin(r), x * np.sin(r) + y * np.cos(r)
    return np.stack([cx + xr, cy + yr], 1).round().astype(np.int32)


def add_sunglasses(img, right_eye, left_eye, rng, min_alpha=0.70):
    """두 눈 위에 합성 선글라스. img: BGR uint8, 눈 좌표는 img 픽셀 기준
    - 렌즈 모양 LENS_SHAPES 중 하나, 크기, 색, 불투명도, 반사광은 무작위
    - min_alpha: 렌즈 불투명도 하한. 감은 눈 이미지는 0.9 이상 (눈이 비치면 "뜸" 라벨과 모순)
    """
    (rx, ry), (lx, ly) = right_eye, left_eye
    dist = math.hypot(lx - rx, ly - ry)
    angle = math.degrees(math.atan2(ly - ry, lx - rx))
    shape = LENS_SHAPES[rng.integers(len(LENS_SHAPES))]
    w = dist * rng.uniform(0.36, 0.46)       # 렌즈 반폭
    h = w * rng.uniform(0.62, 0.85)          # 렌즈 반높이
    dy = h * rng.uniform(-0.10, 0.15)        # 렌즈 중심 높이 (눈 기준)
    base = rng.uniform(5, 50)                # 렌즈 밝기 (어두움)
    tint = base + rng.uniform(-5, 1, 3) * rng.uniform(0, 4)   # 채널마다 조금씩 달라 갈색, 회색, 녹색 기운
    alpha = rng.uniform(min_alpha, 0.97)     # 렌즈 불투명도 (낮으면 눈이 살짝 비침)
    frame = tuple(int(v) for v in rng.uniform(0, 40, 3))
    thick = max(1, int(round(dist * 0.05)))

    lens = np.zeros(img.shape[:2], np.uint8)
    polys, centers = [], []
    for (ex, ey), side in (((rx, ry), -1), ((lx, ly), 1)):
        c = (ex, ey + dy)
        centers.append(c)
        poly = lens_polygon(c[0], c[1], w, h, angle, shape, side)
        polys.append(poly)
        cv2.fillPoly(lens, [poly], 255)
    out = img.astype(np.float32)
    m = (lens > 0)[..., None]
    # 위쪽이 더 진한 렌즈 (그라데이션): 렌즈 위 -> 아래로 불투명도 alpha -> alpha * 0.85
    ys = np.arange(img.shape[0], dtype=np.float32)[:, None, None]
    top = min(ry, ly) + dy - h
    grad = np.clip((ys - top) / (2 * h), 0, 1)
    a = alpha * (1 - 0.15 * grad)
    out = np.where(m, out * (1 - a) + np.clip(tint, 0, 255) * a, out)
    # 렌즈 위쪽 옅은 반사광
    shine = np.zeros(img.shape[:2], np.uint8)
    for cx, cy in centers:
        cv2.ellipse(shine, (int(cx), int(cy - h * 0.45)), (int(w * 0.6), max(1, int(h * 0.2))), angle, 0, 360, 255, -1)
    shine = cv2.GaussianBlur((shine & lens).astype(np.float32) / 255, (0, 0), max(0.5, dist * 0.05))[..., None]
    out = out + shine * rng.uniform(10, 45)
    out = np.clip(out, 0, 255).astype(np.uint8)
    cv2.polylines(out, polys, True, frame, thick, cv2.LINE_AA)
    cv2.line(out, (int(centers[0][0] + w * 0.9), int(centers[0][1])), (int(centers[1][0] - w * 0.9), int(centers[1][1])),
             frame, thick, cv2.LINE_AA)
    return out


def main():
    rows = find_landmarks()
    lm = {(r["class"], r["file"]): r for r in rows}
    print(f"눈 좌표: {sum(r['rx'] != '' for r in rows)} / {len(rows)} -> {LANDMARKS}")
    rng = np.random.default_rng(SEED)
    cell = 96

    def show(img):
        return cv2.resize(img, (cell, cell), interpolation=cv2.INTER_NEAREST)

    def synth(cls, name):
        r = lm[(cls, name)]
        img = cv2.imread(str(PRE / cls / name))
        return add_sunglasses(img, (float(r["rx"]), float(r["ry"])), (float(r["lx"]), float(r["ly"])), rng,
                              min_alpha=0.70 if cls == "opened" else 0.90)

    real = [show(cv2.imread(str(PRE / "opened" / f"{n}.png"))) for n in REAL_SUNGLASSES]
    opened = [r["file"] for r in rows if r["class"] == "opened" and r["rx"] != ""]
    closed = [r["file"] for r in rows if r["class"] == "closed" and r["rx"] != ""]
    pick = np.random.default_rng(1)
    syn_open = [show(synth("opened", f)) for f in pick.choice(opened, 13, replace=False)]
    syn_closed = [show(synth("closed", f)) for f in pick.choice(closed, 13, replace=False)]
    shadow = [show(cv2.imread(str(PRE / "closed" / f"{n}.png"))) for n in SHADOW_CLOSED]
    blank = np.full((cell, cell, 3), 255, np.uint8)
    lines = [real, syn_open, syn_closed, shadow + [blank] * (13 - len(shadow))]
    grid = np.vstack([np.hstack(line) for line in lines])
    cv2.imwrite(str(HERE / "preview.png"), grid)
    print(f"미리보기: {HERE / 'preview.png'}")


if __name__ == "__main__":
    main()
