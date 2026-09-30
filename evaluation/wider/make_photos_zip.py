"""WIDER 평가 사진만 zip 으로 묶음 (팀 드라이브에 올려 공유)
- 대상: labels/labels.csv 의 사진 전부 (1,040장). WIDER 원본 사진 그대로 (다시 저장하지 않음)
- zip 안 경로: WIDER 폴더 기준 (WIDER_val/images/...). 받은 쪽은 evaluation/work/wider_face/ 에 풀면 됨
- 입력: WIDER 원본 폴더 (evaluation/paths.py 의 WIDER)
- 출력: evaluation/work/wider_eval_photos.zip
- WIDER FACE 는 CC BY-NC-ND: 팀 내부에만 공유

실행: app/.venv/bin/python evaluation/wider/make_photos_zip.py
"""
import csv
import sys
import zipfile
from pathlib import Path

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, WIDER, WORK  # noqa: E402

OUT = WORK / "wider_eval_photos.zip"


def main():
    images = sorted({r["image"] for r in csv.DictReader(open(LABELS / "labels.csv", encoding="utf-8"))})
    missing = [p for p in images if not (WIDER / p).exists()]
    if missing:
        raise SystemExit(f"사진 없음 {len(missing)}장 (예: {WIDER / missing[0]})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as zf:   # jpg 라 압축하지 않음
        for p in images:
            zf.write(WIDER / p, p)
    print(f"{OUT} ({len(images)}장, {OUT.stat().st_size / 1e6:.0f}MB)")


if __name__ == "__main__":
    main()
