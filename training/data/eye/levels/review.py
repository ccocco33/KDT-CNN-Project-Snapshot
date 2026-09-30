"""eye 눈 뜬 정도 4단계 검수 도구
- 단계 (검수 기준, 2026-09-26 thinkcat)
  - 1 눈 뜸: 눈이 보이는 사람 (웃든 안 웃든). 웃음이 아닌 표정(찡그림 등)으로 눈이 찌푸려진 경우도 1
  - 2 웃는 눈: 웃어서 눈이 ^^ 처럼 된 사람
  - 3 감기는 눈: 아래를 보거나, 웃어도 눈이 -- 처럼 감기는 사람 (애매한 경우가 꽤 있음)
  - 4 감은 눈: 확실히 감은 사람. -- 이고 무표정, 또는 -- 이고 눈에 찡그림 없이 웃지만 그냥 감은 것 같은 사람
  - 0 선글라스: 선글라스류로 눈이 보이지 않는 사람. 학습에서는 1 로 넣음 (요구사항: 선글라스류는 눈 조건 충족. make_levels.py)
- 검수 대상: blendshape 로 뽑은 후보 (../closed_review/blendshapes.csv)
  - A 웃는 눈 후보: opened, 눈 감음 >= 0.3, 웃음 >= 0.5 (처음 값 2)
  - B 감기는 눈 후보 (opened): opened, 눈 감음 >= 0.3, 웃음 < 0.5 (처음 값 3)
  - C 감기는 눈 후보 (closed): closed, 눈 감음 < 0.5 (처음 값 3)
- 조작 (한글 입력 상태에서도 동작하는 키만 사용)
  - 숫자 1 2 3 4 0: 칠할 단계 선택 (0 은 선글라스)
  - 사진 클릭: 그 사진을 선택한 단계로
  - 스페이스: 다음 화면 / 백스페이스(delete): 이전 화면 / ESC: 저장 후 종료
  - 테두리 색: 초록 1, 노랑 2, 주황 3, 빨강 4, 회색 0. 사진 아래 글자는 현재 단계
- 저장: ./review.csv (folder, file, group, level). 화면을 넘길 때마다 저장. 다시 실행하면 이어서

실행: app/.venv/bin/python training/data/eye/levels/review.py
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[2]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "eye" / "levels"   # 데이터 위치 (training/paths.py)
EYE = HERE.parent
BLEND = EYE / "closed_review" / "blendshapes.csv"
OUT = HERE / "review.csv"
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

COLS, ROWS, CELL, HEAD = 6, 4, 150, 56
NAMES = {1: "눈 뜸", 2: "웃는 눈", 3: "감기는 눈", 4: "감은 눈", 0: "선글라스"}
COLORS = {1: (60, 180, 60), 2: (0, 210, 255), 3: (0, 140, 255), 4: (60, 60, 230), 0: (150, 150, 150)}   # BGR
GROUPS = {"A": "웃는 눈 후보", "B": "감기는 눈 후보 (opened)", "C": "감기는 눈 후보 (closed)"}
KEYS = {ord(str(k)): k for k in NAMES}


def candidates():
    """[(folder, file, group, 처음 단계)]"""
    out = []
    for r in csv.DictReader(open(BLEND)):
        if r["blink"] == "":
            continue
        blink, smile = float(r["blink"]), float(r["smile"])
        if r["folder"] == "opened" and blink >= 0.3:
            out.append((r["folder"], r["file"], "A" if smile >= 0.5 else "B", 2 if smile >= 0.5 else 3))
        elif r["folder"] == "closed" and blink < 0.5:
            out.append((r["folder"], r["file"], "C", 3))
    return sorted(out, key=lambda x: (x[2], x[0], x[1]))


def load_levels(items):
    levels = {(f, n): lv for f, n, _, lv in items}
    if OUT.exists():
        for r in csv.DictReader(open(OUT)):
            levels[(r["folder"], r["file"])] = int(r["level"])
    return levels


def save(items, levels):
    with open(OUT, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["folder", "file", "group", "level"])
        for f, n, g, _ in items:
            w.writerow([f, n, g, levels[(f, n)]])


def render(items, levels, page, pages, brush, font, small):
    per = COLS * ROWS
    canvas = np.full((HEAD + ROWS * CELL, COLS * CELL, 3), 255, np.uint8)
    chunk = items[page * per:(page + 1) * per]
    for i, (f, n, g, _) in enumerate(chunk):
        img = cv2.resize(cv2.imread(str(EYE / "preprocessed" / f / n)), (CELL - 8, CELL - 26), interpolation=cv2.INTER_CUBIC)
        y, x = HEAD + (i // COLS) * CELL, (i % COLS) * CELL
        lv = levels[(f, n)]
        canvas[y:y + CELL, x:x + CELL] = COLORS[lv]
        canvas[y + 4:y + CELL - 22, x + 4:x + CELL - 4] = img
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    g = chunk[0][2] if chunk else ""
    done = sum(1 for f, n, _, _ in items[:(page + 1) * per])
    d.text((8, 4), f"[{GROUPS.get(g, '')}] 화면 {page + 1}/{pages} (검수 {done}/{len(items)})   칠할 단계: {brush} {NAMES[brush]}",
           font=font, fill=(0, 0, 0))
    d.text((8, 30), "숫자 1 뜸 2 웃는눈 3 감기는눈 4 감은눈 0 선글라스 / 클릭: 칠하기 / 스페이스: 다음 / delete: 이전 / ESC: 종료",
           font=small, fill=(90, 90, 90))
    for i, (f, n, _, _) in enumerate(chunk):
        y, x = HEAD + (i // COLS) * CELL, (i % COLS) * CELL
        lv = levels[(f, n)]
        d.text((x + 6, y + CELL - 21), f"{lv} {NAMES[lv]}", font=small, fill=(255, 255, 255) if lv in (3, 4, 0) else (0, 0, 0))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), chunk


def main():
    items = candidates()
    levels = load_levels(items)
    per = COLS * ROWS
    pages = (len(items) + per - 1) // per
    font, small = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 14)
    state = {"page": 0, "brush": 1, "chunk": []}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONUP or y < HEAD:
            return
        i = ((y - HEAD) // CELL) * COLS + x // CELL
        if i < len(state["chunk"]):
            f, n, _, _ = state["chunk"][i]
            levels[(f, n)] = state["brush"]

    cv2.namedWindow("eye levels", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("eye levels", on_mouse)
    while True:
        img, state["chunk"] = render(items, levels, state["page"], pages, state["brush"], font, small)
        cv2.imshow("eye levels", img)
        key = cv2.waitKey(50) & 0xFF
        if key == 27:
            break
        if key in KEYS:
            state["brush"] = KEYS[key]
        elif key == 32:
            save(items, levels)
            state["page"] = min(state["page"] + 1, pages - 1)
        elif key in (8, 127):
            save(items, levels)
            state["page"] = max(state["page"] - 1, 0)
    save(items, levels)
    cv2.destroyAllWindows()
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
