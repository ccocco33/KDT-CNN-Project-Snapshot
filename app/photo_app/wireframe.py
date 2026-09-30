"""와이어프레임 이미지 생성
- 카메라 없이 상태별 화면을 PNG 로 저장
- 화면을 모은 grid.png (화면 아래에 상태 이름). README 화면 예시용
- 실행: python -m photo_app.wireframe [출력 폴더]
- 기본 출력 폴더: 저장소의 docs/wireframe/
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from . import config
from .model import NOT_VISIBLE, TOO_SMALL, Face
from .states import State
from .ui import View, _draw_texts, render

OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "wireframe"
GRID_COLS = 4
GRID_SCALE = 0.5     # grid 안 화면 크기 비율
LABEL_H = 34         # 화면 아래 상태 이름 칸 높이


def dummy_frame():
    """더미 카메라 프레임: 회색 배경 + 얼굴 자리 3개"""
    w, h = config.SCREEN_W, config.SCREEN_H
    frame = np.full((h, w, 3), (120, 120, 120), np.uint8)
    faces = [
        Face((120, 160, 100, 120), smile=True, eyes_open=True),
        Face((270, 150, 100, 120), smile=False, eyes_open=True),
        Face((420, 160, 100, 120), smile=True, eyes_open=True),
    ]
    for f in faces:
        x, y, fw, fh = f.bbox
        cv2.ellipse(frame, (x + fw // 2, y + fh // 2), (fw // 2, fh // 2), 0, 0, 360, (200, 200, 200), -1)
    return frame, faces


def screens():
    """(파일 이름, grid 이름, 상태, View) 목록"""
    frame, faces = dummy_frame()
    photo = frame.copy()
    return [
        ("1_home", "HOME 초기화면", State.HOME, View()),
        ("2_preview", "PREVIEW 미리보기", State.PREVIEW, View(frame=frame, faces=faces)),
        ("2_preview_guide", "PREVIEW 안내 (작은 얼굴, 가린 얼굴)", State.PREVIEW,
         View(frame=frame, faces=faces, guides=[(faces[0], TOO_SMALL), (faces[2], NOT_VISIBLE)])),
        ("2_preview_cancel", "PREVIEW 촬영 취소 안내", State.PREVIEW, View(frame=frame, faces=faces, show_cancel_msg=True)),
        ("2_preview_camera_lost", "PREVIEW 카메라 끊김", State.PREVIEW, View(camera_lost=True)),
        ("3_prepare", "PREPARE 준비 (미소)", State.PREPARE, View(frame=frame, faces=faces, remaining_sec=3)),
        ("4_capture", "CAPTURE 촬영", State.CAPTURE, View(frame=frame, faces=faces, remaining_sec=1.4)),
        ("5_checking", "CHECKING 사진 고르기", State.CHECKING, View()),
        ("6_result", "RESULT 결과", State.RESULT, View(results=[photo] * 5)),
        ("6_result_closest", "RESULT 가장 가까운 사진", State.RESULT, View(results=[photo] * 3, results_closest=True)),
        ("6_result_empty", "RESULT 사진 없음", State.RESULT, View()),
    ]


def grid(tiles: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """화면을 GRID_COLS 열로 모으고 화면마다 아래에 이름"""
    w, h = round(config.SCREEN_W * GRID_SCALE), round(config.SCREEN_H * GRID_SCALE)
    rows = (len(tiles) + GRID_COLS - 1) // GRID_COLS
    img = np.full((rows * (h + LABEL_H), GRID_COLS * w, 3), 255, np.uint8)
    texts = []
    for i, (label, screen) in enumerate(tiles):
        x, y = (i % GRID_COLS) * w, (i // GRID_COLS) * (h + LABEL_H)
        img[y:y + h, x:x + w] = cv2.resize(screen, (w, h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (200, 200, 200), 1)
        texts.append((label, (x + w // 2, y + h + LABEL_H // 2), 16, (40, 40, 40), "mm"))
    return _draw_texts(img, texts)


def main(out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for name, label, state, view in screens():
        path = out_dir / f"{name}.png"
        screen = render(state, view)
        cv2.imwrite(str(path), screen)
        tiles.append((label, screen))
        print(path)
    path = out_dir / "grid.png"
    cv2.imwrite(str(path), grid(tiles))
    print(path)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR)
