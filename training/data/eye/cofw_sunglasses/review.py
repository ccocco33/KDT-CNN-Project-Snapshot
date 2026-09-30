"""COFW 선글라스 얼굴 검수 도구 (eye 학습 데이터에 실제 선글라스 추가)
- 대상: occlusion 검수(../../occlusion/review.csv) 에서 선글라스 후보(S) 중 보임(2) 으로 둔 얼굴
- 분류 (eye 라벨 기준: 선글라스류로 눈이 안 보이면 뜸, 눈이 보이면 실제 눈 상태)
  - 1 선글라스: 렌즈로 두 눈이 안 보임 (반사, 미러 포함) -> opened
  - 2 눈 보임, 뜸: 옅은 렌즈, 투명 안경 등으로 눈이 보이고 뜸 -> opened
  - 3 눈 보임, 감음: 눈이 보이고 감음 -> closed
  - 4 제외: 안대, 한쪽만 가림, 판단 어려움
- 조작 (한글 입력 상태에서도 동작하는 키만 사용)
  - 숫자 1 2 3 4: 칠할 분류 선택 / 사진 클릭: 그 사진을 선택한 분류로
  - 스페이스: 다음 화면 / 백스페이스(delete): 이전 화면 / ESC: 저장 후 종료
- 저장: ./review.csv (file, label). 화면을 넘길 때마다 저장. 다시 실행하면 이어서

실행: app/.venv/bin/python training/data/eye/cofw_sunglasses/review.py
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

HERE = DATASET / "eye" / "cofw_sunglasses"   # 데이터 위치 (training/paths.py)
OCC = HERE.parents[1] / "occlusion"
FACES = OCC / "cofw" / "faces"
OUT = HERE / "review.csv"
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

COLS, ROWS, CELL, HEAD = 5, 3, 220, 56
NAMES = {1: "선글라스", 2: "보임 뜸", 3: "보임 감음", 4: "제외"}
COLORS = {1: (60, 180, 60), 2: (200, 150, 40), 3: (60, 60, 230), 4: (150, 150, 150)}   # BGR
KEYS = {ord(str(k)): k for k in NAMES}


def candidates():
    return sorted(r["file"] for r in csv.DictReader(open(OCC / "review.csv")) if r["group"] == "S" and r["label"] == "2")


def load_labels(items):
    labels = {f: 1 for f in items}
    if OUT.exists():
        for r in csv.DictReader(open(OUT)):
            labels[r["file"]] = int(r["label"])
    return labels


def save(items, labels):
    with open(OUT, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "label"])
        for f in items:
            w.writerow([f, labels[f]])


def render(items, labels, page, pages, brush, font, small):
    per = COLS * ROWS
    canvas = np.full((HEAD + ROWS * CELL, COLS * CELL, 3), 255, np.uint8)
    chunk = items[page * per:(page + 1) * per]
    for i, f in enumerate(chunk):
        img = cv2.resize(cv2.imread(str(FACES / f)), (CELL - 8, CELL - 26), interpolation=cv2.INTER_AREA)
        y, x = HEAD + (i // COLS) * CELL, (i % COLS) * CELL
        canvas[y:y + CELL, x:x + CELL] = COLORS[labels[f]]
        canvas[y + 4:y + CELL - 22, x + 4:x + CELL - 4] = img
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    done = min(len(items), (page + 1) * per)
    d.text((8, 4), f"화면 {page + 1}/{pages} (검수 {done}/{len(items)})   칠할 분류: {brush} {NAMES[brush]}",
           font=font, fill=(0, 0, 0))
    d.text((8, 30), "숫자 1 선글라스(눈 안 보임) 2 눈 보임 뜸 3 눈 보임 감음 4 제외 / 클릭: 칠하기 / 스페이스: 다음 / delete: 이전 / ESC: 종료",
           font=small, fill=(90, 90, 90))
    for i, f in enumerate(chunk):
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
    state = {"page": 0, "brush": 2, "chunk": []}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONUP or y < HEAD:
            return
        i = ((y - HEAD) // CELL) * COLS + x // CELL
        if x < COLS * CELL and i < len(state["chunk"]):
            labels[state["chunk"][i]] = state["brush"]

    cv2.namedWindow("cofw sunglasses", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("cofw sunglasses", on_mouse)
    while True:
        img, state["chunk"] = render(items, labels, state["page"], pages, state["brush"], font, small)
        cv2.imshow("cofw sunglasses", img)
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
