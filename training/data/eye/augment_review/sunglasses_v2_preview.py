"""합성 선글라스 렌즈 종류별 미리보기 (eyewear_fn.add_sunglasses 의 style)
- dark, reflect, mirror, light. 원본: eye 전처리본 opened 얼굴 (64px, 눈 좌표 ../landmarks/eye_landmarks.csv). 표시는 3배 확대
- 출력: sunglasses_v2_preview.png (줄: 종류, 칸: 같은 얼굴들)

실행: app/.venv/bin/python training/data/eye/augment_review/sunglasses_v2_preview.py
"""
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

import eyewear_fn

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[2]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "eye" / "augment_review"   # 데이터 위치 (training/paths.py)
eyewear_fn.math, eyewear_fn.np, eyewear_fn.cv2 = math, np, cv2


def sunglasses_v2(img, re, le, rng, style):
    return eyewear_fn.add_sunglasses(img.copy(), re, le, rng, min_alpha=0.7, style=style, bgr=True)


def main():
    rng = np.random.default_rng(7)
    lm = {r["file"]: r for r in csv.DictReader(open(HERE.parent / "landmarks" / "eye_landmarks.csv")) if r["class"] == "opened"}
    files = rng.choice(sorted(lm), 10, replace=False)
    rows = []
    for style in ("dark", "reflect", "mirror", "light"):
        tiles = []
        for f in files:
            img = cv2.imread(str(HERE.parent / "preprocessed" / "opened" / f))
            r = lm[f]
            out = sunglasses_v2(img, (float(r["rx"]), float(r["ry"])), (float(r["lx"]), float(r["ly"])), rng, style)
            tiles.append(cv2.resize(out, (192, 192), interpolation=cv2.INTER_CUBIC))
        bar = np.full((192, 110, 3), 35, np.uint8)
        cv2.putText(bar, style, (8, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.hstack([bar] + tiles))
    cv2.imwrite(str(HERE / "sunglasses_v2_preview.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
