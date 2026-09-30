"""eye 재학습 모델 vs 앱 eye (v0.0.5) 비교
- 대상: WIDER dev, holdout. 가로 640, YuNet (SCORE_TH) 검출, 정답과 짝 (evaluate.match). 보이는 얼굴(visible=1)만
- 자르기: 앱과 같음 (square_crop, 위쪽 0.6, 160)
- 눈 라벨: 1 뜸, 0 감음, 2 선글라스 (뜸으로 봐야 함)
- 출력
  - AUC (뜸 1 vs 감음 0)
  - 기준값별 감은 눈 recall, 뜬 눈을 감음으로 본 비율, 선글라스를 뜸으로 본 비율
    - 앱 기준 0.8, 노트북 기준, dev 에서 고른 기준 (감은 눈 recall >= 95% 중 뜬 눈 recall 최대)
  - 선글라스 얼굴 그리드 (새 모델 점수 순, 옛 점수 병기): eye_retrain_sunglasses_{TAG}.png
  - 얼굴별 점수: eye_retrain_{TAG}.csv

실행: app/.venv/bin/python evaluation/wider/eye_retrain_compare.py MODEL_DIR TAG
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
from photo_app.models.yunet_cnn.model import Classifier, square_crop, upper  # noqa: E402

NEW_DIR, TAG = Path(ARGS[0]), ARGS[1]
NB_TH = json.loads((NEW_DIR / "inference_config.json").read_text())["threshold"]


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def collect():
    old = Classifier(c.EYE_PATH, c.EYE_SIZE, 4)
    new = Classifier(NEW_DIR / "eye_fp32.tflite", c.EYE_SIZE, 4)
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
                if g["visible"] != "1" or g["eyes_open"] not in ("0", "1", "2"):
                    continue
                eye = upper(square_crop(img, boxes[pi]), 0.6)
                rows.append({"subset": subset, "image": path, "h": boxes[pi][3], "label": g["eyes_open"],
                             "old": old(eye), "new": new(eye), "crop": square_crop(img, boxes[pi])})
    return rows


def pick_th(rows, key):
    """dev 에서 감은 눈 recall >= 95% 인 기준 중 뜬 눈 recall 최대"""
    closed = np.array([r[key] for r in rows if r["label"] == "0"])
    best = None
    for th in np.unique([r[key] for r in rows if r["label"] in "01"]):
        if (closed < th).mean() >= 0.95:
            best = th   # 오름차순이므로 조건을 만족하는 가장 작은 값이 뜬 눈 recall 최대
            break
    return float(best)


def report(rows, key, th):
    closed = [r[key] for r in rows if r["label"] == "0"]
    opened = [r[key] for r in rows if r["label"] == "1"]
    sun = [r[key] for r in rows if r["label"] == "2"]
    cr = sum(v < th for v in closed)
    oc = sum(v < th for v in opened)
    so = sum(v >= th for v in sun)
    return (f"기준 {th:.3f}: 감은 눈 recall {cr}/{len(closed)} ({cr / len(closed):.0%})  "
            f"뜬 눈을 감음으로 {oc}/{len(opened)} ({oc / len(opened):.1%})  "
            f"선글라스를 뜸으로 {so}/{len(sun)} ({so / len(sun):.0%})")


def grid(rows):
    items = sorted([r for r in rows if r["label"] == "2"], key=lambda r: r["new"])
    tiles = []
    for r in items:
        t = cv2.resize(cv2.cvtColor(r["crop"], cv2.COLOR_RGB2BGR), (120, 120), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(t, (0, 0), (119, 119), (0, 200, 0) if r["new"] >= c.EYE_TH else (0, 0, 255), 2)
        cv2.rectangle(t, (0, 92), (120, 120), (0, 0, 0), -1)
        cv2.putText(t, f"new {r['new']:.2f} h{r['h']}", (3, 104), 0, 0.38, (0, 255, 255), 1)
        cv2.putText(t, f"old {r['old']:.2f}", (3, 117), 0, 0.38, (255, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 10:
        tiles.append(np.zeros((120, 120, 3), np.uint8))
    out = HERE / f"eye_retrain_sunglasses_{TAG}.png"
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))
    return out


def main():
    rows = collect()
    with open(HERE / f"eye_retrain_{TAG}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "image", "h", "label", "old", "new"])
        for r in rows:
            w.writerow([r["subset"], r["image"], r["h"], r["label"], round(r["old"], 4), round(r["new"], 4)])
    print(f"비교: 앱 eye {c.MODEL_VERSION} (old) vs {NEW_DIR.name} (new), 가로 640, 보이는 얼굴")
    dev = [r for r in rows if r["subset"] == "dev"]
    th = {k: pick_th(dev, k) for k in ("old", "new")}
    for subset in ("dev", "holdout"):
        sub = [r for r in rows if r["subset"] == subset]
        print(f"\n[{subset}] 뜸 {sum(r['label'] == '1' for r in sub)}, 감음 {sum(r['label'] == '0' for r in sub)}, "
              f"선글라스 {sum(r['label'] == '2' for r in sub)}")
        for key in ("old", "new"):
            a = auc([r[key] for r in sub if r["label"] == "1"], [r[key] for r in sub if r["label"] == "0"])
            print(f"  {key}  AUC {a:.3f}")
            ths = [c.EYE_TH, th[key]] + ([NB_TH] if key == "new" else [])
            for t in ths:
                print(f"    {report(sub, key, t)}")
    sun = [r for r in rows if r["label"] == "2"]
    for key in ("old", "new"):
        bad = [r for r in sun if r[key] < c.EYE_TH]
        print(f"\n[선글라스 {len(sun)}명, 기준 {c.EYE_TH}] {key}: 감음으로 {len(bad)}명, 높이 중앙값 "
              f"{np.median([r['h'] for r in bad]) if bad else '-'}")
    print(grid(rows))


if __name__ == "__main__":
    main()
