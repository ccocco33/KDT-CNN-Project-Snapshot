"""가림 합성 v2 미리보기: COFW 실제 가림 부위를 우리 얼굴로 옮겨 붙이기 + 앞사람 겹침
- 원본: smile, eye 전처리본 얼굴 64x64 (눈, 입 위치는 YuNet 랜드마크, preview.Landmarks)
- COFW 가리개: cofw/faces.csv 의 M (입 6점 가림), E (눈 한쪽 5점 가림). review.csv 가 있으면 1 가림만
  - COFW 얼굴을 두 눈 중심, 입 중심 3점 affine 으로 우리 얼굴 위치에 정렬
  - 가려진 부위(입 또는 가려진 눈) 중심 타원만 섞음. 경계는 흐리게
- person: preview.occ_person (다른 얼굴을 머리 모양 타원으로 겹침)
- 출력: preview_transfer.png (줄: 입 옮김, 눈 옮김, person)

실행: app/.venv/bin/python training/data/occlusion/preview_transfer.py
"""
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from preview import SIZE, Landmarks, apply, label, load_faces, occ_person, region_masks, tile

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[1]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "occlusion"   # 데이터 위치 (training/paths.py)
COFW = HERE / "cofw"
EYES = ((8, 10, 12, 13, 16), (9, 11, 14, 15, 17))
MOUTH = (22, 23, 24, 25, 26, 27)
N = 12


def cofw_items(group):
    ok = None
    if (HERE / "review.csv").exists():
        ok = {r["file"] for r in csv.DictReader(open(HERE / "review.csv")) if r["label"] == "1"}
    out = []
    for r in csv.DictReader(open(COFW / "faces.csv")):
        if r["group"] != group or (ok is not None and r["file"] not in ok):
            continue
        pt = lambda ids: (np.mean([float(r[f"x{i}"]) for i in ids]), np.mean([float(r[f"y{i}"]) for i in ids]))  # noqa: E731
        side = None
        if group == "E":
            side = 0 if int(r["eye_r_occ"]) == 5 else 1   # EYES[0] 은 이미지 왼쪽 눈
        out.append((r["file"], pt(EYES[0]), pt(EYES[1]), pt(MOUTH), side))
    return out


def transfer(img, lm, item, rng):
    """COFW 얼굴을 정렬해 가려진 부위만 섞음 -> (결과, 섞인 영역 마스크)"""
    f, ce0, ce1, cm, side = item
    src = cv2.imread(str(COFW / "faces" / f))
    (rx, ry), (lx, ly), (mrx, mry), (mlx, mly) = lm
    dst_pts = np.float32([(rx, ry), (lx, ly), ((mrx + mlx) / 2, (mry + mly) / 2)])
    M = cv2.getAffineTransform(np.float32([ce0, ce1, cm]), dst_pts)
    warped = cv2.warpAffine(src, M, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    dist = math.hypot(lx - rx, ly - ry)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    if side is None:
        c = dst_pts[2]
        axes = (dist * rng.uniform(0.75, 1.0), dist * rng.uniform(0.45, 0.6))
    else:
        c = dst_pts[side]
        axes = (dist * rng.uniform(0.45, 0.6), dist * rng.uniform(0.4, 0.55))
    cv2.ellipse(mask, (int(c[0]), int(c[1])), (int(axes[0]), int(axes[1])), 0, 0, 360, 1, -1)
    return warped, mask


def main():
    rng = np.random.default_rng(0)
    lmk = Landmarks()
    faces = load_faces(rng, 200)
    rows = []
    for name, maker in (("mouth transfer", "M"), ("eye transfer", "E"), ("person", None)):
        items = cofw_items(maker) if maker else None
        tiles = []
        for _ in range(N):
            img = faces[rng.integers(len(faces))]
            lm, _ = lmk(img)
            if maker:
                layer, mask = transfer(img, lm, items[rng.integers(len(items))], rng)
                out, covered = apply(img, layer, mask, rng)
                out = cv2.GaussianBlur(out, (0, 0), 0.01)
            else:
                layer, mask = occ_person(img, lm, faces[rng.integers(len(faces))], rng)
                out, covered = apply(img, layer, mask, rng)
            y, ratios = label(covered, region_masks(lm))
            worst = max(ratios, key=ratios.get)
            short = {"right_eye": "Reye", "left_eye": "Leye", "mouth": "mouth"}[worst]
            tiles.append(tile(out, f"{'OCC' if y else 'vis'} {short} {ratios[worst]:.0%}", (80, 80, 255) if y else (120, 230, 120)))
        bar = np.full((20, len(tiles) * tiles[0].shape[1], 3), 60, np.uint8)
        cv2.putText(bar, name, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rows += [bar, np.hstack(tiles)]
    cv2.imwrite(str(HERE / "preview_transfer.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
