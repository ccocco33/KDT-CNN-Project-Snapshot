"""smile 전처리본 (64x64) 의 두 눈 좌표 csv (학습 노트북 합성 선글라스용)
- YuNet 랜드마크: 가장자리 복제 여백 + 4배 확대 후 검출 (occlusion/preview.py 의 Landmarks)
- 못 찾으면 빈칸 (노트북에서 그 이미지는 선글라스를 씌우지 않음)
- 출력: ./eye_landmarks.csv (folder, file, rx, ry, lx, ly. 64px 좌표. rx 가 이미지 왼쪽 눈)
  - folder, file 은 preprocessed.zip 안의 폴더, 파일 이름과 같음

실행: app/.venv/bin/python training/data/smile/landmarks/make_eye_landmarks.py
"""
import csv
import sys
from pathlib import Path

import cv2

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[2]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "smile" / "landmarks"   # 데이터 위치 (training/paths.py)
sys.path.insert(0, str(CODE.parents[1] / "occlusion"))
from preview import Landmarks  # noqa: E402

FOLDERS = ("smile_train", "no_smile_train", "smile_test", "no_smile_test")


def main():
    lmk = Landmarks()
    rows, missing = [], 0
    for folder in FOLDERS:
        for p in sorted((HERE.parent / "preprocessed" / folder).glob("*.png")):
            (re, le, _, _), ok = lmk(cv2.imread(str(p)))
            missing += not ok
            eyes = ({"rx": round(re[0], 2), "ry": round(re[1], 2), "lx": round(le[0], 2), "ly": round(le[1], 2)} if ok
                    else {"rx": "", "ry": "", "lx": "", "ly": ""})
            rows.append({"folder": folder, "file": p.name, **eyes})
    with open(HERE / "eye_landmarks.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["folder", "file", "rx", "ry", "lx", "ly"])
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {HERE / 'eye_landmarks.csv'} ({len(rows)}장, 못 찾음 {missing})")


if __name__ == "__main__":
    main()
