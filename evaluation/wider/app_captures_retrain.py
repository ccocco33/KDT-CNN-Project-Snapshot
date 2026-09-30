"""앱 촬영 사진 (dev 모드 모델 평가 찍기, 연속 촬영) 에 앱 eye, smile 과 재학습 모델 점수 비교 (증상 확인용. 기준값 근거 아님)
- 사진: app/photos/eval/**/*.jpg 와 같은 이름의 json (앱이 저장한 얼굴 bbox)
- 자르기: 앱과 같음 (square_crop. smile 얼굴 전체 128, eye 위쪽 0.6 160). 새 smile 은 inference_config 의 input.bottom
- 출력: app_captures_retrain.csv (사진, 얼굴 높이, 점수 4개), app_captures_retrain.png (촬영 순서 그리드)

실행: app/.venv/bin/python evaluation/wider/app_captures_retrain.py EYE_DIR SMILE_DIR [OLD_DIR]
- OLD_DIR: 비교 기준 모델 폴더 (eye_fp32.tflite, smile_fp32.tflite, smile_inference_config.json). 없으면 앱 config 의 모델
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, RESULTS, WIDER  # noqa: E402

HERE = RESULTS   # 평가 결과 위치 (evaluation/paths.py)
ROOT = REPO
sys.path.insert(0, str(ROOT / "app"))
from photo_app.models.yunet_cnn import config as c  # noqa: E402
from photo_app.models.yunet_cnn.model import Classifier, square_crop, upper  # noqa: E402

EYE_DIR, SMILE_DIR = Path(sys.argv[1]), Path(sys.argv[2])
OLD_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else None
SRC = ROOT / "app" / "photos" / "eval"
def bottom(path):
    return json.loads(path.read_text())["input"].get("bottom", 1.0)


SMILE_BOTTOM = bottom(SMILE_DIR / ("smile_inference_config.json" if (SMILE_DIR / "smile_inference_config.json").exists() else "inference_config.json"))
OLD_BOTTOM = bottom(OLD_DIR / "smile_inference_config.json") if OLD_DIR else c.SMILE_BOTTOM


def lower(face, ratio):
    """정사각형 얼굴의 아래쪽 ratio 비율"""
    return face[face.shape[0] - int(round(face.shape[0] * ratio)):]


def main():
    old_eye, old_smile = (OLD_DIR / "eye_fp32.tflite", OLD_DIR / "smile_fp32.tflite") if OLD_DIR else (c.EYE_PATH, c.SMILE_PATH)
    eye_old, eye_new = Classifier(old_eye, c.EYE_SIZE, 4), Classifier(EYE_DIR / "eye_fp32.tflite", c.EYE_SIZE, 4)
    sm_old, sm_new = (Classifier(old_smile, c.SMILE_SIZE, 4),
                      Classifier(SMILE_DIR / "smile_fp32.tflite", c.SMILE_SIZE, 4))
    rows = []
    for jpg in sorted(SRC.rglob("*.jpg"), key=lambda p: p.stem):
        meta = json.loads(jpg.with_suffix(".json").read_text())
        pred = meta.get("prediction") or {}
        frame = cv2.imread(str(jpg))
        for f in pred.get("faces", []):
            crop = square_crop(frame, tuple(f["bbox"]))   # RGB
            rows.append({"image": str(jpg.relative_to(SRC)), "h": f["bbox"][3], "crop": crop,
                         "eye_old": eye_old(upper(crop, 0.6)), "eye_new": eye_new(upper(crop, 0.6)),
                         "smile_old": sm_old(lower(crop, OLD_BOTTOM)), "smile_new": sm_new(lower(crop, SMILE_BOTTOM))})
    keys = ("eye_old", "eye_new", "smile_old", "smile_new")
    with open(HERE / "app_captures_retrain.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["image", "h", *keys])
        for r in rows:
            w.writerow([r["image"], r["h"], *(round(r[k], 4) for k in keys)])
    tiles = []
    for r in rows:
        t = cv2.resize(cv2.cvtColor(r["crop"], cv2.COLOR_RGB2BGR), (120, 120), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 90), (120, 120), (0, 0, 0), -1)
        cv2.putText(t, f"eye {r['eye_old']:.2f}>{r['eye_new']:.2f}", (2, 102), 0, 0.36,
                    (0, 255, 255) if r["eye_new"] >= c.EYE_TH else (0, 0, 255), 1)
        cv2.putText(t, f"sm {r['smile_old']:.2f}>{r['smile_new']:.2f} h{r['h']}", (2, 116), 0, 0.36,
                    (255, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 12:
        tiles.append(np.zeros((120, 120, 3), np.uint8))
    out = HERE / "app_captures_retrain.png"
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 12]) for i in range(0, len(tiles), 12)]))
    print(len(rows), "얼굴", out)


if __name__ == "__main__":
    main()
