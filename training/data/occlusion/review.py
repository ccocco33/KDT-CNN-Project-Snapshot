"""COFW 가림 검수 도구 (확실히 안 보이는 가림만 양성으로)
- 검수 기준 (2026-09-27 thinkcat): 입이 안 보이거나 두 눈이 모두 안 보이면 가림. 한쪽 눈만 가림은 보임
- 분류
  - 1 가림: 입 또는 두 눈이 확실히 안 보임 (손, 물건, 마스크, 모자 등)
  - 2 보임: 입과 눈 한쪽 이상이 보임. 한쪽 눈만 가림(앞머리 등), 수염, 선글라스류 (요구사항: 선글라스류는 보이는 얼굴)
  - 3 애매: 머리카락 사이로 눈이 비침, 입에 문 작은 물건 등. 학습에서 제외
- 검수 대상: cofw/faces.csv 의 group (cofw_extract.py)
  - M 입 가림 후보 (처음 값 1) / E 눈 한쪽 가림 후보 (처음 값 1) / S 선글라스 후보 (처음 값 2)
- 조작 (한글 입력 상태에서도 동작하는 키만 사용)
  - 숫자 1 2 3: 칠할 분류 선택 / 사진 클릭: 그 사진을 선택한 분류로
  - 스페이스: 다음 화면 / 백스페이스(delete): 이전 화면 / ESC: 저장 후 종료
  - 테두리 색: 빨강 1 가림, 초록 2 보임, 회색 3 애매
- 저장: ./review.csv (file, group, label). 화면을 넘길 때마다 저장. 다시 실행하면 이어서

실행: app/.venv/bin/python training/data/occlusion/review.py
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[1]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "occlusion"   # 데이터 위치 (training/paths.py)
FACES = HERE / "cofw" / "faces"
OUT = HERE / "review.csv"
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

COLS, ROWS, CELL, HEAD = 6, 4, 170, 56
NAMES = {1: "가림", 2: "보임", 3: "애매"}
COLORS = {1: (60, 60, 230), 2: (60, 180, 60), 3: (150, 150, 150)}   # BGR
GROUPS = {"M": "입 가림 후보", "E": "눈 한쪽 가림 후보", "S": "선글라스 후보"}
START = {"M": 1, "E": 1, "S": 2}
KEYS = {ord(str(k)): k for k in NAMES}


def candidates():
    """[(file, group)] 묶음 순서 M, E, S"""
    rows = [r for r in csv.DictReader(open(HERE / "cofw" / "faces.csv")) if r["group"] in GROUPS]
    return sorted(((r["file"], r["group"]) for r in rows), key=lambda x: ("MES".index(x[1]), x[0]))


def load_labels(items):
    labels = {f: START[g] for f, g in items}
    if OUT.exists():
        for r in csv.DictReader(open(OUT)):
            labels[r["file"]] = int(r["label"])
    return labels


def save(items, labels):
    with open(OUT, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "group", "label"])
        for f, g in items:
            w.writerow([f, g, labels[f]])


def render(items, labels, page, pages, brush, font, small):
    per = COLS * ROWS
    canvas = np.full((HEAD + ROWS * CELL, COLS * CELL, 3), 255, np.uint8)
    chunk = items[page * per:(page + 1) * per]
    for i, (f, g) in enumerate(chunk):
        img = cv2.resize(cv2.imread(str(FACES / f)), (CELL - 8, CELL - 26), interpolation=cv2.INTER_AREA)
        y, x = HEAD + (i // COLS) * CELL, (i % COLS) * CELL
        canvas[y:y + CELL, x:x + CELL] = COLORS[labels[f]]
        canvas[y + 4:y + CELL - 22, x + 4:x + CELL - 4] = img
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    g = chunk[0][1] if chunk else ""
    done = min(len(items), (page + 1) * per)
    d.text((8, 4), f"[{GROUPS.get(g, '')}] 화면 {page + 1}/{pages} (검수 {done}/{len(items)})   칠할 분류: {brush} {NAMES[brush]}",
           font=font, fill=(0, 0, 0))
    d.text((8, 30), "숫자 1 가림 2 보임(수염, 선글라스) 3 애매 / 클릭: 칠하기 / 스페이스: 다음 / delete: 이전 / ESC: 종료",
           font=small, fill=(90, 90, 90))
    for i, (f, _) in enumerate(chunk):
        y, x = HEAD + (i // COLS) * CELL, (i % COLS) * CELL
        lb = labels[f]
        d.text((x + 6, y + CELL - 21), f"{lb} {NAMES[lb]}", font=small, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), chunk


def main():
    items = candidates()
    labels = load_labels(items)
    per = COLS * ROWS
    pages = (len(items) + per - 1) // per
    font, small = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 14)
    state = {"page": 0, "brush": 1, "chunk": []}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONUP or y < HEAD:
            return
        i = ((y - HEAD) // CELL) * COLS + x // CELL
        if x < COLS * CELL and i < len(state["chunk"]):
            labels[state["chunk"][i][0]] = state["brush"]

    cv2.namedWindow("cofw occlusion", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("cofw occlusion", on_mouse)
    while True:
        img, state["chunk"] = render(items, labels, state["page"], pages, state["brush"], font, small)
        cv2.imshow("cofw occlusion", img)
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
