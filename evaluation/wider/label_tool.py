"""단체 사진 얼굴 라벨링 도구 (OpenCV)
- 실행 (저장소 루트에서): app/.venv/bin/python evaluation/wider/label_tool.py
- 입력, 출력
  - labels.csv: 얼굴 1개 = 1행. visible (1 / 0), hidden_reason, smile, eyes_open (1 / 0 / -1 판단 불가) 을 채움
    - visible, hidden_reason 칸이 없는 이전 파일은 빈 칸으로 추가 (값을 추측해 채우지 않음)
  - images.csv: 사진 1장 = 1행. unusable (1 사용 부적절), reason (사유)
- 사진을 넘길 때, 종료할 때 자동 저장. 다시 실행하면 처음 미완료 얼굴부터
- 키
  - 7 / 8: 보임 = 1 입과 눈(한쪽 이상)이 보임 / 0 보이지 않음 (옆모습, 입 가림, 두 눈 가림)
    - 0 이면 웃음, 눈을 -1 로 채우고 이유 선택 (1 side 옆모습 / 2 hand 손 / 3 hair 머리카락 / 4 mask 마스크 / 5 person 다른 사람, 물건 / 6 other 기타)
    - 보임 0 이고 이유가 빈 칸이면 이유 선택 중: 1~6 은 이유 키 (보임을 바꾸려면 7)
    - 보임이 빈 칸이거나 0 일 때 웃음이나 눈을 입력하면 보임은 1 로 채움 (0 에서 바꾸면 자동으로 채운 -1 을 비움)
  - 1 / 2 / 3: 웃음 = 1 웃음 / 0 웃지 않음 / -1 판단 불가
  - 4 / 5 / 6 / 0: 눈 = 1 뜸 / 0 감음 / -1 판단 불가 / 2 선글라스류로 눈이 보이지 않음 (조건 충족으로 계산. 투명한 안경은 1/0)
  - n / p: 다음 / 이전 얼굴
  - ] / [: 다음 / 이전 사진
  - x: 사진 사용 부적절 토글 (켤 때 터미널에서 사유 입력)
  - u: 다음 미완료 얼굴로 이동
  - r: 현재 얼굴 라벨 초기화 (보임, 이유, 웃음, 눈)
  - t: 현재 사진 전체 초기화 (모든 얼굴 라벨 + 사용 부적절 표시). 첫 얼굴로 이동
  - q, ESC: 저장 후 종료
- 한글 입력 상태에서도 동작 (두벌식 자모를 영문 키로 변환. 예: ㅜ -> n)
- 키가 안 먹으면 LABEL_DEBUG_KEYS=1 로 실행해 터미널에 찍히는 키 코드 확인
- 얼굴 라벨이 완료되면 (보임 0 + 이유, 또는 보임 1 + 웃음 + 눈) 같은 사진의 다음 얼굴로 이동. 마지막 얼굴이면 머묾 (다음 사진은 ])
- 보임 기준 (2026-09-27 thinkcat 변경): 입이 안 보이거나 두 눈이 모두 안 보이면 보이지 않음. 한쪽 눈만 가림은 보임
- 재검수 모드: LABEL_RECHECK=1 로 실행
  - 대상: 보이지 않음 + 이유가 옆모습(side)이 아닌 얼굴 (예전 기준 "하나라도 안 보임" 으로 매긴 가림)
  - 그 얼굴이 있는 사진과 그 얼굴만 보여 줌. u 는 확인하지 않은 다음 얼굴로
  - 여전히 가림: 8 을 누르고 이유 다시 선택 / 새 기준으로 보임: 웃음, 눈 입력 (보임 1 로 바뀜)
  - 라벨이 완료되면 확인 처리. 확인 목록은 recheck_occluded.csv (image, face_id) 에 저장, 다시 실행하면 이어서
"""
from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, RESULTS, WIDER  # noqa: E402

HERE = RESULTS   # 평가 결과 위치 (evaluation/paths.py)
ROOT = REPO
LABELS = LABELS / "labels.csv"
IMAGES = LABELS / "images.csv"
RECHECK = os.environ.get("LABEL_RECHECK") == "1"
RECHECK_FILE = LABELS / "recheck_occluded.csv"
RECHECK_REASONS = ("hand", "hair", "mask", "person", "other", "")   # 옆모습(side) 제외

VIEW_W, VIEW_H = 1100, 800   # 사진 영역 최대 크기
PANEL_W = 320                # 오른쪽 패널 최소 너비 (설명 줄이 길면 늘어남)
CROP_SIZE = 300              # 현재 얼굴 확대 이미지 크기 (긴 변)
PAD = 10                     # 패널 여백
LINE_H = 20                  # 설명 줄 간격
FONT, FONT_SCALE = cv2.FONT_HERSHEY_SIMPLEX, 0.45
CROP_MARGIN = 0.3            # 확대 이미지 여백 (얼굴 크기 비율)

KEY_SMILE = {ord("1"): "1", ord("2"): "0", ord("3"): "-1"}
KEY_EYES = {ord("4"): "1", ord("5"): "0", ord("6"): "-1", ord("0"): "2"}
KEY_VISIBLE = {ord("7"): "1", ord("8"): "0"}

# 두벌식 한글 자모 -> 같은 자리의 영문 키 (한글 입력 상태에서 누른 키 복원)
HANGUL_TO_LATIN = dict(zip("ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔㅁㄴㅇㄹㅎㅗㅓㅏㅣㅋㅌㅊㅍㅠㅜㅡ", "qwertyuiopasdfghjklzxcvbnm"))
HANGUL_TO_LATIN.update(zip("ㅃㅉㄸㄲㅆㅒㅖ", "qwertop"))   # Shift 자모


def normalize_key(code: int) -> int:
    """waitKeyEx 코드 -> 영문 키 코드
    - 한글 자모면 같은 자리의 영문 키로 변환
    - 그 외는 하위 8비트
    """
    if code > 0xFF:
        try:
            ch = chr(code)
        except (ValueError, OverflowError):
            return code & 0xFF
        if ch in HANGUL_TO_LATIN:
            return ord(HANGUL_TO_LATIN[ch])
    return code & 0xFF
KEY_REASON = {ord("1"): "side", ord("2"): "hand", ord("3"): "hair", ord("4"): "mask", ord("5"): "person", ord("6"): "other"}

# 색 (BGR)
GRAY, GREEN, RED, YELLOW, CYAN, WHITE = (160, 160, 160), (80, 200, 80), (60, 60, 230), (0, 210, 255), (255, 200, 0), (255, 255, 255)
MAGENTA = (200, 80, 200)
VALUE_TEXT = {"1": "yes", "0": "no", "-1": "unknown", "2": "eyewear", "": "-"}


@dataclass
class State:
    rows: list[dict]                                   # labels.csv 전체 행
    images: list[str]                                  # 사진 경로 순서
    flags: dict[str, dict]                             # 사진 -> {"unusable": "0"/"1", "reason": str}
    img_idx: int = 0
    face_idx: int = 0
    dirty: bool = False
    faces_by_image: dict[str, list[int]] = field(default_factory=dict)   # 사진 -> rows 인덱스 목록
    checked: set[tuple[str, str]] | None = None   # 재검수 모드: 확인한 (사진, face_id). 재검수가 아니면 None

    @property
    def image(self) -> str:
        return self.images[self.img_idx]

    @property
    def faces(self) -> list[int]:
        return self.faces_by_image[self.image]

    @property
    def row(self) -> dict | None:
        return self.rows[self.faces[self.face_idx]] if self.faces else None


def load() -> State:
    rows = list(csv.DictReader(open(LABELS, encoding="utf-8")))
    for r in rows:
        r.setdefault("visible", "")
        r.setdefault("hidden_reason", "")
    images, by_image = [], {}
    for i, r in enumerate(rows):
        if r["image"] not in by_image:
            images.append(r["image"])
            by_image[r["image"]] = []
        if r["target"] == "1":
            by_image[r["image"]].append(i)
    flags = {img: {"unusable": "0", "reason": ""} for img in images}
    if IMAGES.exists():
        for r in csv.DictReader(open(IMAGES, encoding="utf-8")):
            if r["image"] in flags:
                flags[r["image"]] = {"unusable": r["unusable"], "reason": r["reason"]}
    checked = None
    if RECHECK:
        by_image = {img: [i for i in idx if rows[i]["visible"] == "0" and rows[i]["hidden_reason"] in RECHECK_REASONS]
                    for img, idx in by_image.items()}
        images = [img for img in images if by_image[img]]
        checked = set()
        if RECHECK_FILE.exists():
            checked = {(r["image"], r["face_id"]) for r in csv.DictReader(open(RECHECK_FILE, encoding="utf-8"))}
    state = State(rows, images, flags, faces_by_image=by_image, checked=checked)
    goto_next_unlabeled(state, start_here=True)
    return state


def save(state: State) -> None:
    with open(LABELS, "w", newline="", encoding="utf-8") as f:
        tail = ["visible", "hidden_reason", "smile", "eyes_open"]
        fields = [k for k in state.rows[0] if k not in tail] + tail
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(state.rows)
    with open(IMAGES, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "unusable", "reason"])
        w.writeheader()
        w.writerows({"image": img, **state.flags[img]} for img in state.flags)
    if state.checked is not None:
        with open(RECHECK_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "face_id"])
            w.writerows(sorted(state.checked))
    state.dirty = False


def picking_reason(row: dict | None) -> bool:
    """보이지 않음 이유 선택 중: 보임 0 이고 이유가 빈 칸"""
    return row is not None and row["visible"] == "0" and row["hidden_reason"] == ""


def is_done(row: dict) -> bool:
    if row["visible"] == "0":
        return row["hidden_reason"] != ""
    return row["visible"] == "1" and row["smile"] != "" and row["eyes_open"] != ""


def face_done(state: State, row: dict) -> bool:
    """이동, 진행 표시용 완료. 재검수 모드면 확인 여부"""
    if state.checked is not None:
        return (row["image"], row["face_id"]) in state.checked
    return is_done(row)


def mark_checked(state: State, row: dict | None) -> None:
    """재검수 모드: 라벨이 완료된 얼굴을 확인 처리"""
    if state.checked is not None and row is not None and is_done(row):
        state.checked.add((row["image"], row["face_id"]))


def image_done(state: State, img: str) -> bool:
    """사진 라벨 완료: 사용 부적절이거나 모든 얼굴 완료"""
    return state.flags[img]["unusable"] == "1" or all(is_done(state.rows[i]) for i in state.faces_by_image[img])


def goto_next_unlabeled(state: State, start_here: bool = False) -> bool:
    """다음 미완료 얼굴로 이동. 없으면 False (위치 유지)"""
    order = [(ii, fi) for ii, img in enumerate(state.images) for fi in range(len(state.faces_by_image[img]))]
    cur = (state.img_idx, state.face_idx)
    start = next((k for k, pos in enumerate(order) if pos == cur), 0)
    for k in range(len(order)):
        ii, fi = order[(start + k + (0 if start_here else 1)) % len(order)]
        img = state.images[ii]
        if state.flags[img]["unusable"] == "1":
            continue
        if not face_done(state, state.rows[state.faces_by_image[img][fi]]):
            state.img_idx, state.face_idx = ii, fi
            return True
    return False


def reset_face(row: dict) -> None:
    for k in ("visible", "hidden_reason", "smile", "eyes_open"):
        row[k] = ""


def next_face_in_image(state: State) -> None:
    """같은 사진의 다음 얼굴로. 마지막 얼굴이면 그대로"""
    if state.face_idx < len(state.faces) - 1:
        state.face_idx += 1


def move_image(state: State, step: int) -> None:
    state.img_idx = (state.img_idx + step) % len(state.images)
    state.face_idx = 0


def handle_key(state: State, key: int, ask_reason=input) -> bool:
    """키 처리. 종료 키면 False"""
    if key in (ord("q"), 27):
        return False
    row = state.row
    if picking_reason(row) and key in KEY_REASON:   # 보이지 않음 이유 선택
        row["hidden_reason"] = KEY_REASON[key]
        state.dirty = True
        mark_checked(state, row)
        next_face_in_image(state)
        return True
    if row is not None and key in KEY_VISIBLE:
        row["visible"] = KEY_VISIBLE[key]
        if row["visible"] == "0":
            row["smile"] = row["eyes_open"] = "-1"
            if state.checked is not None:   # 재검수: 예전 이유가 남아 바로 넘어가지 않도록 다시 고르게 함
                row["hidden_reason"] = ""
        else:
            row["hidden_reason"] = ""
            if row["smile"] == row["eyes_open"] == "-1":   # 보이지 않음 -> 보임으로 바꾼 경우 다시 입력
                row["smile"] = row["eyes_open"] = ""
        state.dirty = True
        if picking_reason(row):   # 이유를 고를 때까지 이동하지 않음
            return True
    elif row is not None and (key in KEY_SMILE or key in KEY_EYES):
        if row["visible"] == "0":   # 보이지 않음 -> 웃음, 눈 입력: 보임으로 바꾸고 자동으로 채운 -1, 이유를 비움
            row["smile"] = row["eyes_open"] = row["hidden_reason"] = ""
        row["visible"] = "1"
        if key in KEY_SMILE:
            row["smile"] = KEY_SMILE[key]
        else:
            row["eyes_open"] = KEY_EYES[key]
        state.dirty = True
    elif key == ord("n") and state.faces:
        state.face_idx = (state.face_idx + 1) % len(state.faces)
    elif key == ord("p") and state.faces:
        state.face_idx = (state.face_idx - 1) % len(state.faces)
    elif key in (ord("]"), ord("[")):
        if state.dirty:
            save(state)
        move_image(state, 1 if key == ord("]") else -1)
    elif key == ord("u"):
        goto_next_unlabeled(state)
    elif key == ord("r") and row is not None:
        reset_face(row)
        state.dirty = True
    elif key == ord("t"):
        for i in state.faces:
            reset_face(state.rows[i])
        state.flags[state.image] = {"unusable": "0", "reason": ""}
        state.face_idx = 0
        state.dirty = True
    elif key == ord("x"):
        flag = state.flags[state.image]
        if flag["unusable"] == "1":
            flag.update(unusable="0", reason="")
        else:
            flag.update(unusable="1", reason=ask_reason("사용 부적절 사유: ").strip())
            print(f"부적절: {state.image} ({flag['reason']})")
        state.dirty = True
        save(state)
        return True
    else:
        return True
    # 현재 얼굴을 다 채우면 같은 사진의 다음 얼굴로
    if key in KEY_SMILE or key in KEY_EYES or key in KEY_VISIBLE:
        if row is not None and is_done(row):
            mark_checked(state, row)
            next_face_in_image(state)
    return True


def face_color(row: dict) -> tuple[int, int, int]:
    if not is_done(row):
        return GRAY
    if row["visible"] == "0":
        return MAGENTA
    if "-1" in (row["smile"], row["eyes_open"]):
        return YELLOW
    return GREEN if row["smile"] == "1" and row["eyes_open"] in ("1", "2") else RED   # 2: 선글라스류로 눈이 보이지 않음 = 눈 조건 충족


def render(state: State, image: np.ndarray) -> np.ndarray:
    """사진 + 오른쪽 패널"""
    h, w = image.shape[:2]
    scale = min(VIEW_W / w, VIEW_H / h, 1.0)
    view = cv2.resize(image, (int(w * scale), int(h * scale)))
    unusable = state.flags[state.image]["unusable"] == "1"

    for k, ri in enumerate(state.faces):
        r = state.rows[ri]
        x, y, fw, fh = (int(int(r[c]) * scale) for c in ("x", "y", "w", "h"))
        current = k == state.face_idx
        color = CYAN if current else face_color(r)
        cv2.rectangle(view, (x, y), (x + fw, y + fh), color, 3 if current else 2)
        cv2.putText(view, str(k), (x, max(y - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if unusable:
        cv2.putText(view, "UNUSABLE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 3, cv2.LINE_AA)

    row = state.row
    if state.checked is not None:
        targets = [state.rows[i] for img in state.images for i in state.faces_by_image[img]]
        done_faces, total_faces = sum(face_done(state, r) for r in targets), len(targets)
    else:
        done_faces = sum(is_done(r) for r in state.rows if r["target"] == "1")
        total_faces = sum(r["target"] == "1" for r in state.rows)
    done_images = sum(image_done(state, img) for img in state.images)
    lines = [
        f"image {state.img_idx + 1}/{len(state.images)}",
        f"face {state.face_idx + 1 if row else 0}/{len(state.faces)}",
    ]
    if row is not None:
        lines += [
            f"visible: {VALUE_TEXT[row['visible']]}" + (f" ({row['hidden_reason'] or '?'})" if row["visible"] == "0" else ""),
            f"smile: {VALUE_TEXT[row['smile']]}",
            f"eyes_open: {VALUE_TEXT[row['eyes_open']]}",
            f"occlusion(WIDER): {row['occlusion']}  blur: {row['blur']}",
        ]
    lines += [
        f"unusable: {'YES' if unusable else 'no'}",   # 사유(한글)는 putText 로 못 그려 images.csv, 터미널에서 확인
        "",
        f"{'RECHECK occluded ' if state.checked is not None else ''}done faces {done_faces}/{total_faces}",
        f"done images {done_images}/{len(state.images)}",
        "",
        "7/8 visible yes/no (side, mouth or both eyes covered)",
        *(["PICK REASON: 1 side 2 hand 3 hair", "  4 mask 5 person 6 other"] if picking_reason(row) else []),
        "1/2/3 smile yes/no/unknown",
        "4/5/6/0 eyes open/closed/?/eyewear",
        "n/p face  ]/[ image",
        "u jump to next unlabeled face",
        "r reset face  t reset image",
        "x unusable  q quit",
    ]

    # 패널 크기: 설명 줄이 잘리지 않도록 가장 긴 줄의 너비, 줄 수에 맞춤
    text_w = max(cv2.getTextSize(t, FONT, FONT_SCALE, 1)[0][0] for t in lines if t)
    panel_w = max(PANEL_W, text_w + 2 * PAD)
    text_y0 = PAD + CROP_SIZE + PAD + LINE_H
    panel_h = max(view.shape[0], text_y0 + (len(lines) - 1) * LINE_H + PAD)
    panel = np.full((panel_h, panel_w, 3), 40, np.uint8)

    if row is not None:
        x, y, fw, fh = (int(row[c]) for c in ("x", "y", "w", "h"))
        m = int(max(fw, fh) * CROP_MARGIN)
        crop = image[max(y - m, 0):y + fh + m, max(x - m, 0):x + fw + m]
        if crop.size:
            s = CROP_SIZE / max(crop.shape[:2])
            crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
            panel[PAD:PAD + crop.shape[0], PAD:PAD + crop.shape[1]] = crop

    for i, text in enumerate(lines):
        cv2.putText(panel, text, (PAD, text_y0 + i * LINE_H), FONT, FONT_SCALE, WHITE, 1, cv2.LINE_AA)

    canvas = np.full((panel_h, view.shape[1] + panel_w, 3), 30, np.uint8)
    canvas[:view.shape[0], :view.shape[1]] = view
    canvas[:, view.shape[1]:] = panel
    return canvas


def main():
    if not LABELS.exists():
        sys.exit(f"{LABELS} 없음. 먼저 make_label_sheet.py 실행")
    state = load()
    print(__doc__)
    cache = {}
    try:
        while True:
            path = WIDER / state.image
            if path not in cache:
                cache.clear()
                cache[path] = cv2.imread(str(path))
            cv2.imshow("label_tool", render(state, cache[path]))
            code = cv2.waitKeyEx(0)
            key = normalize_key(code)
            if os.environ.get("LABEL_DEBUG_KEYS"):
                print(f"key code {code} -> {chr(key)!r}")
            if not handle_key(state, key):
                break
    finally:
        save(state)
        cv2.destroyAllWindows()
        print(f"저장: {LABELS}, {IMAGES}")


if __name__ == "__main__":
    main()
