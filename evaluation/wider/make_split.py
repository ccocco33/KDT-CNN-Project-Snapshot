"""WIDER 평가 데이터를 dev / holdout 으로 나눔
- dev: 설정, 모델을 고를 때 비교용 (자주 봄)
- holdout: 최종 확인용 (마지막에만 봄). 고른 결과가 dev 에 맞춘 것인지 검사
- 사진 단위, 행사 종류(폴더 이름)별로 반반. 홀수면 행사마다 번갈아 dev / holdout 에 하나 더
- 행사마다 고정 시드로 섞음. labels.csv 의 사진 전부 (부적절 사진도 기록. 평가에서는 따로 빠짐)
- 출력: split.csv (image, event, subset). 이미 있으면 덮어쓰지 않고 중단 (분할이 바뀌면 holdout 이 무의미해짐)

실행: app/.venv/bin/python evaluation/wider/make_split.py
"""
import csv
import random
from collections import defaultdict
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, RESULTS, WIDER  # noqa: E402

HERE = RESULTS   # 평가 결과 위치 (evaluation/paths.py)
OUT = LABELS / "split.csv"
SEED = 0


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} 가 이미 있음. 분할을 바꾸려면 직접 지우고 다시 실행")
    images = sorted({r["image"] for r in csv.DictReader(open(LABELS / "labels.csv"))})
    by_event = defaultdict(list)
    for img in images:
        by_event[Path(img).parent.name].append(img)
    rows, extra_to_dev = [], True
    for event in sorted(by_event):
        items = by_event[event]
        random.Random(f"{SEED}:{event}").shuffle(items)
        n_dev = len(items) // 2 + (len(items) % 2 if extra_to_dev else 0)
        if len(items) % 2:
            extra_to_dev = not extra_to_dev
        rows += [{"image": img, "event": event, "subset": "dev" if i < n_dev else "holdout"}
                 for i, img in enumerate(items)]
    with open(OUT, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["image", "event", "subset"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["image"]))
    for event in sorted(by_event):
        rs = [r for r in rows if r["event"] == event]
        print(f"{event:<32} dev {sum(r['subset'] == 'dev' for r in rs):>4}  holdout {sum(r['subset'] == 'holdout' for r in rs):>4}")
    print(f"합계 dev {sum(r['subset'] == 'dev' for r in rows)}, holdout {sum(r['subset'] == 'holdout' for r in rows)} -> {OUT}")


if __name__ == "__main__":
    main()
