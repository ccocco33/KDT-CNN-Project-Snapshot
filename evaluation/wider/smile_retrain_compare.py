"""smile 재학습 모델 vs 앱 smile (v0.0.5) 비교
- 대상: WIDER dev, holdout. 가로 640, YuNet (SCORE_TH) 검출, 정답과 짝 (evaluate.match). 보이는 얼굴(visible=1)만
- 자르기: 앱과 같음 (square_crop, 얼굴 전체, 128). new 는 inference_config 의 input.bottom (아래쪽 비율, 없으면 1.0)
- 웃음 라벨: 1 웃음, 0 안 웃음
- 선글라스 얼굴: 눈 라벨 2
- 출력
  - AUC (웃음 1 vs 안 웃음 0), 전체와 선글라스 얼굴
  - 기준값별 웃음 recall, 안 웃음을 웃음으로 본 비율 (전체, 선글라스 얼굴)
    - 앱 기준 SMILE_TH, 노트북 기준, dev 에서 고른 기준 (안 웃음을 웃음으로 <= old 의 SMILE_TH 비율 중 recall 최대)
  - 선글라스 얼굴 그리드 (새 점수 순, 옛 점수 병기, 라벨): smile_retrain_sunglasses_{TAG}.png
  - 얼굴별 점수: smile_retrain_{TAG}.csv

실행: app/.venv/bin/python evaluation/wider/smile_retrain_compare.py MODEL_DIR TAG
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]
CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, RESULTS, WIDER  # noqa: E402

HERE = RESULTS   # 평가 결과 위치 (evaluation/paths.py)
sys.path.insert(0, str(CODE))
import evaluate  # noqa: E402

sys.path.insert(0, str(evaluate.ROOT / "app"))
from photo_app.models.yunet_cnn import config as c  # noqa: E402
from photo_app.models.yunet_cnn.model import Classifier, square_crop  # noqa: E402

NEW_DIR, TAG = Path(ARGS[0]), ARGS[1]
NB_CFG = json.loads((NEW_DIR / "inference_config.json").read_text())
NB_TH = NB_CFG["threshold"]
NEW_BOTTOM = NB_CFG["input"].get("bottom", 1.0)


def lower(face, ratio):
    """정사각형 얼굴의 아래쪽 ratio 비율"""
    return face[face.shape[0] - int(round(face.shape[0] * ratio)):]


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def collect():
    old = Classifier(c.SMILE_PATH, c.SMILE_SIZE, 4)
    new = Classifier(NEW_DIR / "smile_fp32.tflite", c.SMILE_SIZE, 4)
    det = cv2.FaceDetectorYN.create(str(c.YUNET_PATH), "", (320, 320), c.SCORE_TH)
    rows = []
    for subset in ("dev", "holdout"):
        evaluate.SUBSET = subset
        for path, labels in evaluate.load().items():
            src = cv2.imread(str(evaluate.WIDER / path))
            s = 640 / src.shape[1]
            img = cv2.resize(src, (640, round(src.shape[0] * s)), interpolation=cv2.INTER_AREA)
            det.setInputSize((img.shape[1], img.shape[0]))
            _, d = det.detect(img)
            if d is None:
                continue
            boxes = [tuple(int(v) for v in x[:4]) for x in d]
            gts = [tuple(int(r[k]) * s for k in "xywh") for r in labels]
            for gi, (pi, _) in evaluate.match(gts, boxes).items():
                g = labels[gi]
                if g["visible"] != "1" or g["smile"] not in ("0", "1"):
                    continue
                crop = square_crop(img, boxes[pi])
                rows.append({"subset": subset, "image": path, "h": boxes[pi][3], "label": g["smile"],
                             "sun": g["eyes_open"] == "2", "old": old(crop), "new": new(lower(crop, NEW_BOTTOM)), "crop": crop})
    return rows


def pick_th(rows, key, fpr):
    """안 웃음을 웃음으로 본 비율 <= fpr 인 기준 중 가장 낮은 값 (웃음 recall 최대)"""
    neg = np.array([r[key] for r in rows if r["label"] == "0"])
    for th in np.unique([r[key] for r in rows]):
        if (neg >= th).mean() <= fpr:
            return float(th)
    return 1.0


def report(rows, key, th):
    pos = [r[key] for r in rows if r["label"] == "1"]
    neg = [r[key] for r in rows if r["label"] == "0"]
    tp = sum(v >= th for v in pos)
    fp = sum(v >= th for v in neg)
    return (f"웃음 recall {tp}/{len(pos)} ({tp / max(len(pos), 1):.0%})  "
            f"안 웃음을 웃음으로 {fp}/{len(neg)} ({fp / max(len(neg), 1):.1%})")


def grid(rows):
    items = sorted([r for r in rows if r["sun"]], key=lambda r: r["new"])
    tiles = []
    for r in items:
        t = cv2.resize(cv2.cvtColor(r["crop"], cv2.COLOR_RGB2BGR), (120, 120), interpolation=cv2.INTER_CUBIC)
        wrong = (r["new"] >= c.SMILE_TH) != (r["label"] == "1")
        cv2.rectangle(t, (0, 0), (119, 119), (0, 0, 255) if wrong else (0, 200, 0), 2)
        cv2.rectangle(t, (0, 92), (120, 120), (0, 0, 0), -1)
        cv2.putText(t, f"new {r['new']:.2f} lab{r['label']}", (3, 104), 0, 0.38, (0, 255, 255), 1)
        cv2.putText(t, f"old {r['old']:.2f} h{r['h']}", (3, 117), 0, 0.38, (255, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 10:
        tiles.append(np.zeros((120, 120, 3), np.uint8))
    out = HERE / f"smile_retrain_sunglasses_{TAG}.png"
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))
    return out


def main():
    rows = collect()
    with open(HERE / f"smile_retrain_{TAG}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "image", "h", "label", "sun", "old", "new"])
        for r in rows:
            w.writerow([r["subset"], r["image"], r["h"], r["label"], int(r["sun"]), round(r["old"], 4), round(r["new"], 4)])
    print(f"비교: 앱 smile {c.MODEL_VERSION} (old) vs {NEW_DIR.name} (new, 아래쪽 {NEW_BOTTOM}), 가로 640, 보이는 얼굴")
    dev = [r for r in rows if r["subset"] == "dev"]
    old_fpr = np.mean([r["old"] >= c.SMILE_TH for r in dev if r["label"] == "0"])
    th_new = pick_th(dev, "new", old_fpr)
    print(f"new 의 dev 기준: 안 웃음을 웃음으로 <= old 기준 {c.SMILE_TH} 의 dev 비율 {old_fpr:.1%} -> {th_new:.3f}")
    for subset in ("dev", "holdout"):
        for name, sub in ((subset, [r for r in rows if r["subset"] == subset]),
                          (f"{subset} 선글라스", [r for r in rows if r["subset"] == subset and r["sun"]])):
            print(f"\n[{name}] 웃음 {sum(r['label'] == '1' for r in sub)}, 안 웃음 {sum(r['label'] == '0' for r in sub)}")
            for key in ("old", "new"):
                a = auc([r[key] for r in sub if r["label"] == "1"], [r[key] for r in sub if r["label"] == "0"])
                print(f"  {key}  AUC {a:.3f}")
                for t in [c.SMILE_TH] + ([th_new, NB_TH] if key == "new" else []):
                    print(f"    기준 {t:.3f}: {report(sub, key, t)}")
    print(grid(rows))


if __name__ == "__main__":
    main()
