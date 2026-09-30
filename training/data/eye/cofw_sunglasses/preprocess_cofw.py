"""COFW 선글라스 얼굴을 eye 학습 데이터 형식으로 전처리
- 입력: ./review.csv (검수 결과), ../../occlusion/cofw/faces/*.png (COFW 에서 잘라 둔 얼굴 192px)
  - 1 선글라스, 2 눈 보임 뜸 -> opened / 3 눈 보임 감음 -> closed / 4 제외
- 처리: eye 전처리(../preprocessed/preprocess.py) 의 process 를 그대로 사용 (YuNet 다시 검출, 박스 높이 정사각형, 64px, 흑백 제외)
- 출력
  - ./raw/{opened,closed}/cofw_*.png: 입력 복사본 (process 가 eye 폴더 기준 상대 경로를 기록하므로)
  - ./{opened,closed}/cofw_*.png: 64px 결과
  - ./manifest.csv: process 기록 + review 라벨

실행: app/.venv/bin/python training/data/eye/cofw_sunglasses/preprocess_cofw.py
"""
import csv
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import cv2

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[2]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "eye" / "cofw_sunglasses"   # 데이터 위치 (training/paths.py)
sys.path.insert(0, str(CODE.parent / "preprocessed"))
import preprocess as pp  # noqa: E402

FACES = HERE.parents[1] / "occlusion" / "cofw" / "faces"
CLASS = {"1": "opened", "2": "opened", "3": "closed"}


def main():
    yunet = cv2.FaceDetectorYN.create(str(pp.YUNET), "", (320, 320), score_threshold=pp.SCORE_TH)
    args = Namespace(align=False, clahe=False)
    for d in ("raw", "opened", "closed"):
        if (HERE / d).exists():
            shutil.rmtree(HERE / d)
    rows = []
    for r in csv.DictReader(open(HERE / "review.csv")):
        cls = CLASS.get(r["label"])
        if cls is None:
            continue
        name = "cofw_" + Path(r["file"]).name.replace("cofw_", "")
        src = HERE / "raw" / cls / name
        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FACES / r["file"], src)
        row, face = pp.process(yunet, src, cls, args)
        if face is not None:
            dst = HERE / cls / name
            dst.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(dst), face)
            row["dst"] = str(dst.relative_to(HERE))
        row["review"] = r["label"]
        rows.append(row)
    with open(HERE / "manifest.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(pp.FIELDS) + ["review"])
        w.writeheader()
        w.writerows(rows)
    for cls in ("opened", "closed"):
        rs = [r for r in rows if r["class"] == cls]
        n = lambda s: sum(r["status"] == s for r in rs)
        print(f"{cls}: {len(rs)} -> ok {n('ok')}, 흑백 제외 {n('excluded_gray')}, 미검출 제외 {n('excluded_no_face')}")


if __name__ == "__main__":
    main()
