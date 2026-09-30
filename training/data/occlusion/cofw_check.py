"""COFW (color) 확인: 얼굴 크기, 우리 기준 가림 라벨 수, 예시 그리드
- 데이터: cofw/COFW_{train,test}_color.mat (CaltechDATA, CC BY 4.0. Burgos-Artizzu, Perona, Dollar, ICCV 2013)
  - train 1,345 = LFPW 845 (가림 거의 없음) + COFW 500 / test 507 (COFW)
  - phis: 랜드마크 29개 x, y, 가림(1 = 가려짐). bboxes: x, y, w, h
- 랜드마크 번호 (평균 위치로 확인, cofw/lm_index.png)
  - 눈: 8, 10, 12, 13, 16 / 9, 11, 14, 15, 17. 입: 22~27. 코 18~21, 눈썹 0~7, 턱 28
- 우리 기준 가림: 눈 한쪽 5점 중 EYE_MIN 점 이상 가림, 또는 입 6점 중 MOUTH_MIN 점 이상 가림
  - COFW 는 선글라스도 가림으로 표시 -> 두 눈만 가려진 얼굴은 선글라스 후보로 따로 셈 (검수 필요)
- 출력: cofw/check_grid_{가림,보임,선글라스후보}.png (칸 위 점: 빨강 가려짐, 초록 보임)
- 필요: h5py (앱 venv 에는 없음)

실행: python training/data/occlusion/cofw_check.py
"""
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[1]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "occlusion" / "cofw"   # 데이터 위치 (training/paths.py)
EYES = ((8, 10, 12, 13, 16), (9, 11, 14, 15, 17))
MOUTH = (22, 23, 24, 25, 26, 27)
EYE_MIN, MOUTH_MIN = 3, 3


def load(name, img_key, phis_key, bb_key):
    m = h5py.File(HERE / name, "r")
    phis, bbs = m[phis_key][()].T, m[bb_key][()].T
    imgs = []
    for ref in m[img_key][0]:
        a = m[ref][()]
        a = a.T if a.ndim == 2 else a.transpose(2, 1, 0)   # MATLAB 열 우선 -> (H, W[, C])
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        imgs.append(a)
    return imgs, phis, bbs


def classify(occ):
    eye_occ = [int(occ[list(e)].sum()) for e in EYES]
    mouth_occ = int(occ[list(MOUTH)].sum())
    eyes_hidden = [n >= EYE_MIN for n in eye_occ]
    mouth_hidden = mouth_occ >= MOUTH_MIN
    if all(eyes_hidden) and not mouth_hidden:
        return "sunglasses_candidate"
    return "occluded" if any(eyes_hidden) or mouth_hidden else "visible"


def crop(img, bb, phis, occ, size=112):
    x, y, w, h = bb
    c = max(w, h) * 1.2
    cx, cy = x + w / 2, y + h / 2
    x1, y1 = int(cx - c / 2), int(cy - c / 2)
    pad = int(c)
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    f = big[y1 + pad:y1 + pad + int(c), x1 + pad:x1 + pad + int(c)]
    s = size / f.shape[0]
    f = cv2.resize(f, (size, size))
    for i in list(EYES[0]) + list(EYES[1]) + list(MOUTH):
        p = (int((phis[i] - x1) * s), int((phis[29 + i] - y1) * s))
        cv2.circle(f, p, 2, (0, 0, 255) if occ[i] else (0, 200, 0), -1)
    return f


def main():
    sets = {"train": load("COFW_train_color.mat", "IsTr", "phisTr", "bboxesTr"),
            "test": load("COFW_test_color.mat", "IsT", "phisT", "bboxesT")}
    rng = np.random.default_rng(0)
    samples = {"occluded": [], "visible": [], "sunglasses_candidate": []}
    for name, (imgs, phis, bbs) in sets.items():
        parts = {"LFPW": range(0, 845), "COFW": range(845, len(imgs))} if name == "train" else {"COFW": range(len(imgs))}
        for part, idx in parts.items():
            counts = {"occluded": 0, "visible": 0, "sunglasses_candidate": 0}
            hs = []
            for i in idx:
                occ = phis[i, 58:87]
                k = classify(occ)
                counts[k] += 1
                hs.append(bbs[i, 3])
                if part == "COFW":
                    samples[k].append((imgs[i], bbs[i], phis[i], occ))
            hs = np.array(hs)
            print(f"{name:<5} {part:<4} {len(idx):>4}장  얼굴 높이 중앙값 {np.median(hs):.0f}px (10% {np.percentile(hs, 10):.0f}, 90% {np.percentile(hs, 90):.0f})  "
                  f"가림 {counts['occluded']}  보임 {counts['visible']}  선글라스 후보 {counts['sunglasses_candidate']}")
    for k, items in samples.items():
        pick = rng.choice(len(items), min(24, len(items)), replace=False)
        tiles = [crop(*items[i]) for i in pick]
        rows = [np.hstack(tiles[r:r + 8]) for r in range(0, len(tiles) - len(tiles) % 8, 8)]
        cv2.imwrite(str(HERE / f"check_grid_{k}.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
