"""가림 합성 미리보기 (학습 전 확인용)
- 원본: smile, eye 전처리본 얼굴 (64x64, 기울기 보정)
- 눈, 입 위치: YuNet 랜드마크 (가장자리 복제 여백 + 4배 확대 후 검출). 못 찾으면 평균 위치
- 가리개 4종
  - person: 다른 얼굴 이미지를 머리 모양 타원으로 잘라 옆이나 아래에서 겹침 (앞사람)
  - object: 다른 이미지의 무작위 조각을 불규칙 다각형으로 (물건)
  - hand: 얼굴에서 뽑은 피부색으로 손바닥 타원 + 손가락 (손)
  - shape: 단색 또는 줄무늬 다각형 (종이, 모자챙 등)
- 라벨: 눈 한쪽 또는 입 영역을 COVER_TH 이상 덮으면 가림(1), 아니면 보임(0)
  - 눈 영역: 눈 중심 반지름 0.22 x 눈 사이 거리 원 / 입 영역: 입 양끝을 잇는 타원
- 경계는 살짝 흐리게 (feather)
- 출력: preview_grid.png (종류별 줄, 칸 아래 라벨과 덮은 비율), preview_sunglasses.png (선글라스는 보임)

실행: app/.venv/bin/python training/data/occlusion/preview.py
"""
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[1]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "occlusion"   # 데이터 위치 (training/paths.py)
DATA = HERE.parent
ROOT = REPO
sys.path.insert(0, str(CODE.parent / "eye" / "augment_review"))
import eyewear_fn  # noqa: E402

eyewear_fn.math, eyewear_fn.np, eyewear_fn.cv2 = math, np, cv2   # 노트북용 함수 모음이라 import 가 없음
add_sunglasses = eyewear_fn.add_sunglasses

YUNET = ROOT / "app" / "models" / "face_detection_yunet_2023mar.onnx"
COVER_TH = 0.5
KINDS = ("person", "object", "hand", "shape")
N_PER_KIND = 12
SIZE = 64


def load_faces(rng, n):
    paths = sorted((DATA / "smile" / "preprocessed").glob("*_train/*.png")) + \
        sorted((DATA / "eye" / "preprocessed" / "opened").glob("*.png"))
    pick = rng.choice(len(paths), n, replace=False)
    return [cv2.imread(str(paths[i])) for i in pick]


class Landmarks:
    """64px 얼굴 -> (오른눈, 왼눈, 입 오른끝, 입 왼끝) 픽셀 좌표"""
    MEAN = ((21.0, 25.0), (43.0, 25.0), (24.0, 46.0), (40.0, 46.0))

    def __init__(self):
        self.det = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.3)

    def __call__(self, img):
        pad, k = 32, 4
        big = cv2.resize(cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE), None, fx=k, fy=k)
        self.det.setInputSize((big.shape[1], big.shape[0]))
        _, d = self.det.detect(big)
        if d is None:
            return self.MEAN, False
        d = max(d, key=lambda r: r[14])
        pts = [(d[i] / k - pad, d[i + 1] / k - pad) for i in (4, 6, 10, 12)]
        return tuple(pts), True


def feather(mask, r=1.2):
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), r)


def region_masks(lm):
    (rx, ry), (lx, ly), (mrx, mry), (mlx, mly) = lm
    dist = math.hypot(lx - rx, ly - ry)
    eyes = []
    for ex, ey in ((rx, ry), (lx, ly)):
        m = np.zeros((SIZE, SIZE), np.uint8)
        cv2.circle(m, (int(round(ex)), int(round(ey))), max(2, int(round(dist * 0.22))), 1, -1)
        eyes.append(m)
    mouth = np.zeros((SIZE, SIZE), np.uint8)
    cx, cy = (mrx + mlx) / 2, (mry + mly) / 2
    cv2.ellipse(mouth, (int(round(cx)), int(round(cy))), (max(3, int(round(abs(mlx - mrx) / 2 + 2))), max(2, int(round(dist * 0.15)))),
                0, 0, 360, 1, -1)
    return {"right_eye": eyes[0], "left_eye": eyes[1], "mouth": mouth}


def random_target(lm, rng):
    """가리개를 겨눌 점: 눈, 입, 또는 이마/볼 (빗나가는 음성도 나오도록)"""
    (rx, ry), (lx, ly), (mrx, mry), (mlx, mly) = lm
    options = [(rx, ry), (lx, ly), ((mrx + mlx) / 2, (mry + mly) / 2),
               ((rx + lx) / 2, ry - 12), (rx - 6, (ry + mry) / 2 + 4), (lx + 6, (ly + mly) / 2 + 4)]
    return options[rng.integers(len(options))]


def blob_polygon(cx, cy, r, rng, n=10):
    t = np.sort(rng.uniform(0, 2 * np.pi, n))
    rr = r * rng.uniform(0.6, 1.3, n)
    return np.stack([cx + rr * np.cos(t), cy + rr * np.sin(t)], 1).round().astype(np.int32)


def occ_person(img, lm, donor, rng):
    """앞사람: 다른 얼굴을 머리 모양 타원으로 잘라 옆이나 아래에서 겹침"""
    tx, ty = random_target(lm, rng)
    s = rng.uniform(0.9, 1.4)
    d = cv2.resize(donor, None, fx=s, fy=s)
    if rng.random() < 0.5:
        d = d[:, ::-1]
    h, w = d.shape[:2]
    ang = rng.uniform(0, 2 * np.pi) if rng.random() < 0.4 else rng.uniform(np.pi * 0.25, np.pi * 0.75)  # 주로 아래에서
    off = rng.uniform(0.25, 0.45) * w
    cx, cy = tx + off * math.cos(ang), ty + off * math.sin(ang)
    layer = np.zeros_like(img)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(SIZE, x0 + w), min(SIZE, y0 + h)
    if xe <= xs or ye <= ys:
        return layer, mask
    layer[ys:ye, xs:xe] = d[ys - y0:ye - y0, xs - x0:xe - x0]
    cv2.ellipse(mask, (int(cx), int(cy)), (int(w * 0.42), int(h * 0.5)), 0, 0, 360, 1, -1)
    return layer, mask


def occ_object(img, lm, donor, rng):
    """물건: 다른 이미지의 무작위 조각을 불규칙 다각형으로"""
    tx, ty = random_target(lm, rng)
    r = rng.uniform(7, 16)
    cx, cy = tx + rng.uniform(-r, r) * 0.6, ty + rng.uniform(-r, r) * 0.6
    src = cv2.resize(donor, None, fx=rng.uniform(1.5, 3.0), fy=rng.uniform(1.5, 3.0))
    oy, ox = rng.integers(0, src.shape[0] - SIZE + 1), rng.integers(0, src.shape[1] - SIZE + 1)
    layer = src[oy:oy + SIZE, ox:ox + SIZE].copy()
    layer = cv2.convertScaleAbs(layer, alpha=rng.uniform(0.6, 1.3), beta=rng.uniform(-40, 40))
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillPoly(mask, [blob_polygon(cx, cy, r, rng, n=rng.integers(4, 9))], 1)
    return layer, mask


def occ_hand(img, lm, donor, rng):
    """손: 얼굴 피부색으로 손바닥 타원 + 손가락 4개"""
    (rx, ry), (lx, ly), _, _ = lm
    tx, ty = random_target(lm, rng)
    skin = img[int(ry) + 6:int(ry) + 12, int(rx):int(lx)].reshape(-1, 3).mean(0) if lx > rx else img.reshape(-1, 3).mean(0)
    skin = np.clip(skin * rng.uniform(0.85, 1.15) + rng.uniform(-10, 10, 3), 0, 255)
    palm_w, palm_h = rng.uniform(8, 13), rng.uniform(10, 15)
    ang = rng.uniform(-60, 60)
    base_x, base_y = tx + rng.uniform(-5, 5), ty + rng.uniform(2, 10)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(mask, (int(base_x), int(base_y)), (int(palm_w), int(palm_h)), ang, 0, 360, 1, -1)
    a = math.radians(ang - 90)
    for i in range(4):
        fx = base_x + (i - 1.5) * palm_w * 0.5 * math.cos(math.radians(ang)) + palm_h * 1.2 * math.cos(a)
        fy = base_y + (i - 1.5) * palm_w * 0.5 * math.sin(math.radians(ang)) + palm_h * 1.2 * math.sin(a)
        cv2.line(mask, (int(base_x + (i - 1.5) * palm_w * 0.5), int(base_y)), (int(fx), int(fy)), 1, max(2, int(palm_w * 0.35)))
    layer = np.empty_like(img)
    layer[:] = skin.astype(np.uint8)
    shade = (np.linspace(0.85, 1.1, SIZE)[None, :, None] * rng.uniform(0.9, 1.1))
    layer = np.clip(layer * shade + rng.normal(0, 4, layer.shape), 0, 255).astype(np.uint8)
    return layer, mask


def occ_shape(img, lm, donor, rng):
    """단색 또는 줄무늬 다각형"""
    tx, ty = random_target(lm, rng)
    r = rng.uniform(7, 16)
    layer = np.empty_like(img)
    layer[:] = rng.integers(0, 256, 3).astype(np.uint8)
    if rng.random() < 0.4:
        c2 = rng.integers(0, 256, 3).astype(np.uint8)
        period = rng.integers(3, 7)
        stripe = ((np.arange(SIZE)[:, None] + np.arange(SIZE)[None, :] * rng.choice([0, 1])) // period) % 2 == 1
        layer[stripe] = c2
    mask = np.zeros((SIZE, SIZE), np.uint8)
    if rng.random() < 0.5:
        cv2.fillPoly(mask, [blob_polygon(tx, ty, r, rng, n=rng.integers(3, 7))], 1)
    else:
        x0, y0 = int(tx - r * rng.uniform(0.3, 1.5)), int(ty - r * rng.uniform(0.3, 1.5))
        cv2.rectangle(mask, (x0, y0), (int(x0 + r * 2), int(y0 + r * rng.uniform(1, 2.5))), 1, -1)
    return layer, mask


OCCLUDERS = {"person": occ_person, "object": occ_object, "hand": occ_hand, "shape": occ_shape}


def apply(img, layer, mask, rng):
    a = feather(mask, rng.uniform(0.6, 1.5))[..., None]
    return np.clip(img * (1 - a) + layer * a, 0, 255).astype(np.uint8), (a[..., 0] > 0.5)


def label(covered, regions):
    ratios = {k: float((covered & (m > 0)).sum()) / max(1, (m > 0).sum()) for k, m in regions.items()}
    return int(max(ratios.values()) >= COVER_TH), ratios


def tile(img, text, color, scale=2):
    t = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    bar = np.full((18, t.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
    return np.vstack([t, bar])


def main():
    rng = np.random.default_rng(0)
    lmk = Landmarks()
    faces = load_faces(rng, 200)
    rows, found = [], 0
    for kind in KINDS:
        tiles = []
        for i in range(N_PER_KIND):
            img, donor = faces[rng.integers(len(faces))], faces[rng.integers(len(faces))]
            lm, ok = lmk(img)
            found += ok
            layer, mask = OCCLUDERS[kind](img, lm, donor, rng)
            out, covered = apply(img, layer, mask, rng)
            y, ratios = label(covered, region_masks(lm))
            worst = max(ratios, key=ratios.get)
            short = {"right_eye": "Reye", "left_eye": "Leye", "mouth": "mouth"}[worst]
            text = f"{'OCC' if y else 'vis'} {short} {ratios[worst]:.0%}"
            tiles.append(tile(out, text, (80, 80, 255) if y else (120, 230, 120)))
        rows.append(np.hstack(tiles))
    head = []
    for kind, row in zip(KINDS, rows):
        bar = np.full((20, row.shape[1], 3), 60, np.uint8)
        cv2.putText(bar, kind, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        head += [bar, row]
    cv2.imwrite(str(HERE / "preview_grid.png"), np.vstack(head))

    # 원본 + 랜드마크 확인, 선글라스 (보임)
    tiles_lm, tiles_sg = [], []
    for i in range(N_PER_KIND):
        img = faces[rng.integers(len(faces))]
        lm, ok = lmk(img)
        vis = img.copy()
        for p in lm:
            cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 1, (0, 255, 255), -1)
        tiles_lm.append(tile(vis, "yunet" if ok else "mean", (200, 200, 200)))
        sg = add_sunglasses(img.copy(), lm[0], lm[1], rng)
        tiles_sg.append(tile(sg, "vis sunglasses", (120, 230, 120)))
    cv2.imwrite(str(HERE / "preview_sunglasses.png"), np.vstack([np.hstack(tiles_lm), np.hstack(tiles_sg)]))
    print(f"랜드마크 YuNet 검출 {found}/{len(KINDS) * N_PER_KIND}")


if __name__ == "__main__":
    main()
