"""GENKI 웃는 눈 후보 검수 도구 (eye 학습 데이터 추가용)
- 대상: ./candidates.csv (select_candidates.py. 눈이 가늘게 잡힌 GENKI 웃음 사진 300장, 눈 열림 작은 순)
- 단계 (eye 단계 검수 levels/review.py 와 같은 기준)
  - 1 눈 뜸: 눈이 보이는 사람
  - 2 웃는 눈: 웃어서 눈이 ^^ 처럼 된 사람 (눈동자가 조금이라도 보이거나, 뜬 채로 가늘어짐)
  - 3 감기는 눈: 웃어도 눈이 -- 처럼 감기는 사람 (애매한 경우)
  - 4 감은 눈: 확실히 감은 사람
  - 0 선글라스: 선글라스류로 눈이 안 보임 (학습에서는 1)
  - 5 제외: 판단 어려움 (흐림, 가림 등)
- 처음 값: candidates.csv 의 level (blendshape 눈 감음 < 0.35 면 2, 아니면 3)
- 조작 (한글 입력 상태에서도 동작하는 키만 사용)
  - 숫자 0 ~ 5: 칠할 단계 선택 / 사진 클릭: 그 사진을 선택한 단계로
  - 스페이스: 다음 화면 / 백스페이스(delete): 이전 화면 / ESC: 저장 후 종료
- 저장: ./review.csv (file, level). 화면을 넘길 때마다 저장. 다시 실행하면 이어서

실행: app/.venv/bin/python training/data/eye/genki_smile/review.py
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[2]))   # training/
from paths import DATASET  # noqa: E402

HERE = DATASET / "eye" / "genki_smile"   # 데이터 위치 (training/paths.py)
SMILE = DATASET / "smile" / "preprocessed"
OUT = HERE / "review.csv"
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

COLS, ROWS, CELL, HEAD = 6, 4, 170, 56
NAMES = {1: "눈 뜸", 2: "웃는 눈", 3: "감기는 눈", 4: "감은 눈", 0: "선글라스", 5: "제외"}
COLORS = {1: (60, 180, 60), 2: (40, 200, 230), 3: (30, 140, 250), 4: (60, 60, 230), 0: (150, 150, 150), 5: (80, 80, 80)}   # BGR
KEYS = {ord(str(k)): k for k in NAMES}


def load():
    items = list(csv.DictReader(open(HERE / "candidates.csv")))
    labels = {r["file"]: int(r["level"]) for r in items}
    if OUT.exists():
        for r in csv.DictReader(open(OUT)):
            labels[r["file"]] = int(r["level"])
    return items, labels


def save(items, labels):
    with open(OUT, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "level"])
        for r in items:
            w.writerow([r["file"], labels[r["file"]]])


def render(items, labels, page, pages, brush, font, small):
    per = COLS * ROWS
    canvas = np.full((HEAD + ROWS * CELL, COLS * CELL, 3), 255, np.uint8)
    chunk = items[page * per:(page + 1) * per]
    for i, r in enumerate(chunk):
        img = cv2.resize(cv2.imread(str(SMILE / r["src"])), (CELL - 8, CELL - 26), interpolation=cv2.INTER_CUBIC)
        y, x = HEAD + (i // COLS) * CELL, (i % COLS) * CELL
        canvas[y:y + CELL, x:x + CELL] = COLORS[labels[r["file"]]]
        canvas[y + 4:y + CELL - 22, x + 4:x + CELL - 4] = img
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    done = min(len(items), (page + 1) * per)
    d.text((8, 4), f"화면 {page + 1}/{pages} (검수 {done}/{len(items)})   칠할 단계: {brush} {NAMES[brush]}",
           font=font, fill=(0, 0, 0))
    d.text((8, 30), "숫자 1 뜸 2 웃는 눈 3 감기는 눈 4 감은 눈 0 선글라스 5 제외 / 클릭: 칠하기 / 스페이스: 다음 / delete: 이전 / ESC: 종료",
           font=small, fill=(90, 90, 90))
    for i, r in enumerate(chunk):
        y, x = HEAD + (i // COLS) * CELL, (i % COLS) * CELL
        lb = labels[r["file"]]
        d.text((x + 6, y + CELL - 21), f"{lb} {NAMES[lb]}", font=small, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), chunk


def main():
    items, labels = load()
    per = COLS * ROWS
    pages = (len(items) + per - 1) // per
    font, small = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 14)
    state = {"page": 0, "brush": 2, "chunk": []}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONUP or y < HEAD:
            return
        i = ((y - HEAD) // CELL) * COLS + x // CELL
        if x < COLS * CELL and i < len(state["chunk"]):
            labels[state["chunk"][i]["file"]] = state["brush"]

    cv2.namedWindow("genki smile eyes", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("genki smile eyes", on_mouse)
    while True:
        img, state["chunk"] = render(items, labels, state["page"], pages, state["brush"], font, small)
        cv2.imshow("genki smile eyes", img)
        key = cv2.waitKey(50) & 0xFF
        if key == 27:
            break
        if key in KEYS:
            state["brush"] = KEYS[key]
        elif key == 32:
            save(items, labels)
            state["page"] = min(state["page"] + 1, pages - 1)
        elif key in (8, 127):
            save(items, labels)
            state["page"] = max(state["page"] - 1, 0)
    save(items, labels)
    cv2.destroyAllWindows()
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
