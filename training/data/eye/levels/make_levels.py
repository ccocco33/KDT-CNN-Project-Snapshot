"""eye 눈 뜬 정도 4단계 라벨 만들기 (전체 이미지)
- 검수한 이미지: ./review.csv 의 단계 (review.py)
  - 0 (선글라스) 은 1 눈 뜸으로 (요구사항: 선글라스류는 눈 조건 충족). source 는 review_sunglasses
- 나머지: 폴더와 blendshape 규칙 (../closed_review/blendshapes.csv)
  - opened 이고 검수 대상이 아님 (눈 감음 < 0.3): 1 눈 뜸
  - closed 이고 검수 대상이 아님 (눈 감음 >= 0.5): 4 감은 눈
  - blendshape 미검출: opened 1, closed 4
- 출력: ./levels.csv (folder, file, level, source). level 1~4. source: review | review_sunglasses | rule
  - folder, file 은 preprocessed.zip 안의 폴더 이름, 파일 이름과 같음
- 요약: 단계별 장수 (폴더별)

실행: app/.venv/bin/python training/data/eye/levels/make_levels.py
"""
import csv
from collections import Counter
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[2]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "eye" / "levels"   # 데이터 위치 (training/paths.py)
EYE = HERE.parent
REVIEW = HERE / "review.csv"
OUT = HERE / "levels.csv"


def main():
    reviewed = {(r["folder"], r["file"]): int(r["level"]) for r in csv.DictReader(open(REVIEW))}
    rows = []
    for folder, default in (("opened", 1), ("closed", 4)):
        for path in sorted((EYE / "preprocessed" / folder).glob("*.png")):
            key = (folder, path.name)
            if key in reviewed and reviewed[key] == 0:
                rows.append({"folder": folder, "file": path.name, "level": 1, "source": "review_sunglasses"})
            elif key in reviewed:
                rows.append({"folder": folder, "file": path.name, "level": reviewed[key], "source": "review"})
            else:
                rows.append({"folder": folder, "file": path.name, "level": default, "source": "rule"})
    with open(OUT, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"저장: {OUT} ({len(rows)}장, 검수 {len(reviewed)}장)")
    for folder in ("opened", "closed"):
        c = Counter(r["level"] for r in rows if r["folder"] == folder)
        print(f"  {folder}: " + ", ".join(f"{k} {c.get(k, 0)}" for k in (1, 2, 3, 4)))


if __name__ == "__main__":
    main()
