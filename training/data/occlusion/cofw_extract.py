"""COFW (color) .mat -> 얼굴 PNG + 랜드마크 csv
- 얼굴: bbox 중심 정사각형 (한 변 = bbox 긴 변 x MARGIN), 화면 밖은 가장자리 복제. 원본 해상도 그대로
- 묶음: lfpw (train 앞 845, 가림 거의 없음), cofw_train (train 뒤 500), cofw_test (test 507)
- 우리 기준 분류 (group)
  - M 입 가림 (엄격): 입 6점 모두 가림, 눈은 한쪽도 5점 모두 가려지지 않음
  - E 눈 한쪽 가림 (엄격): 눈 한쪽 5점 모두 가림 (두 눈 다는 아님), 또는 눈 5점 + 입 6점
  - S 선글라스 후보: 두 눈 5점 모두 가림, 입 6점 가림 아님 -> 검수 (COFW 는 선글라스도 가림으로 표시)
  - V 보임: 눈 한쪽 가림 1점 이하, 입 가림 1점 이하
  - X 애매: 나머지 (학습에서 제외)
- 출력: cofw/faces/{묶음}_{번호}.png, cofw/faces.csv (file, part, group, eye_r_occ, eye_l_occ, mouth_occ, 29점 x, y, occ 은 PNG 좌표)
- 필요: h5py (앱 venv 에는 없음. 따로 만든 환경에서 한 번 실행)
- 데이터 출처: Burgos-Artizzu, Perona, Dollar, "Robust face landmark estimation under occlusion", ICCV 2013. CC BY 4.0

실행: python training/data/occlusion/cofw_extract.py
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from cofw_check import EYES, MOUTH, load

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[1]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "occlusion" / "cofw"   # 데이터 위치 (training/paths.py)
MARGIN = 1.6


def group(occ):
    e = [int(occ[list(E)].sum()) for E in EYES]
    m = int(occ[list(MOUTH)].sum())
    eye_full = [x == 5 for x in e]
    if all(eye_full) and m < 6:
        return "S", e, m
    if any(eye_full):
        return "E", e, m
    if m == 6:
        return "M", e, m
    if max(e) <= 1 and m <= 1:
        return "V", e, m
    return "X", e, m


def square(img, bb):
    x, y, w, h = bb
    c = int(round(max(w, h) * MARGIN))
    x1, y1 = int(round(x + w / 2 - c / 2)), int(round(y + h / 2 - c / 2))
    H, W = img.shape[:2]
    top, left = max(0, -y1), max(0, -x1)
    pad = cv2.copyMakeBorder(img, top, max(0, y1 + c - H), left, max(0, x1 + c - W), cv2.BORDER_REPLICATE)
    return pad[y1 + top:y1 + top + c, x1 + left:x1 + left + c], x1, y1


def main():
    (HERE / "faces").mkdir(exist_ok=True)
    rows = []
    for mat, ik, pk, bk in (("COFW_train_color.mat", "IsTr", "phisTr", "bboxesTr"), ("COFW_test_color.mat", "IsT", "phisT", "bboxesT")):
        imgs, phis, bbs = load(mat, ik, pk, bk)
        for i, (img, ph, bb) in enumerate(zip(imgs, phis, bbs)):
            part = ("lfpw" if i < 845 else "cofw_train") if "train" in mat else "cofw_test"
            face, x1, y1 = square(img, bb)
            name = f"{part}_{i:04d}.png"
            cv2.imwrite(str(HERE / "faces" / name), face)
            occ = ph[58:87]
            g, e, m = group(occ)
            row = {"file": name, "part": part, "group": g, "eye_r_occ": e[0], "eye_l_occ": e[1], "mouth_occ": m, "size": face.shape[0]}
            for k in range(29):
                row[f"x{k}"] = round(float(ph[k] - x1), 1)
                row[f"y{k}"] = round(float(ph[29 + k] - y1), 1)
                row[f"o{k}"] = int(occ[k])
            rows.append(row)
    with open(HERE / "faces.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(Counter((r["part"], r["group"]) for r in rows))


if __name__ == "__main__":
    main()
