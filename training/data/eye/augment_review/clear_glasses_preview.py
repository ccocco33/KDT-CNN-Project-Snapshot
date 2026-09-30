"""eye 증강 검토: 합성 투명 안경
- 함수: ./eyewear_fn.py 의 add_clear_glasses (학습 노트북에 같은 코드가 들어감)
- 눈 좌표: ../landmarks/eye_landmarks.csv (sunglasses_preview.py 가 만든 파일)
- 출력: ./clear_glasses_preview.png
  - 1줄: 뜬 눈 + 투명 안경 / 2줄: 감은 눈 + 투명 안경 / 3줄: 감은 눈 + 투명 안경 (반사광 항상)

실행: app/.venv/bin/python training/data/eye/augment_review/clear_glasses_preview.py
"""
import csv
import math  # noqa: F401  (eyewear_fn 이 사용)
import sys
from pathlib import Path

import cv2
import numpy as np

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[2]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "eye" / "augment_review"   # 데이터 위치 (training/paths.py)
PRE = HERE.parent / "preprocessed"
LANDMARKS = HERE.parent / "landmarks" / "eye_landmarks.csv"
exec((CODE / "eyewear_fn.py").read_text())   # 노트북과 같은 코드를 그대로 씀

N = 12
CELL = 96


class AlwaysReflect:
    """반사광 분기(rng.random() < 0.7)가 항상 참이 되도록 감싼 난수 생성기"""

    def __init__(self, rng):
        self._rng = rng

    def random(self):
        return 0.0

    def __getattr__(self, name):
        return getattr(self._rng, name)


def main():
    rows = [r for r in csv.DictReader(open(LANDMARKS)) if r["rx"] != ""]
    pick = np.random.default_rng(1)
    rng = np.random.default_rng(0)
    lines = []
    for cls, wrap in (("opened", False), ("closed", False), ("closed", True)):
        files = [r for r in rows if r["class"] == cls]
        tiles = []
        for r in pick.choice(files, N, replace=False):
            img = cv2.imread(str(PRE / cls / r["file"]))
            out = add_clear_glasses(img, (float(r["rx"]), float(r["ry"])), (float(r["lx"]), float(r["ly"])),  # noqa: F821
                                    AlwaysReflect(rng) if wrap else rng, bgr=True)
            tiles.append(cv2.resize(out, (CELL, CELL), interpolation=cv2.INTER_NEAREST))
        lines.append(np.hstack(tiles))
    cv2.imwrite(str(HERE / "clear_glasses_preview.png"), np.vstack(lines))
    print(HERE / "clear_glasses_preview.png")


if __name__ == "__main__":
    main()
