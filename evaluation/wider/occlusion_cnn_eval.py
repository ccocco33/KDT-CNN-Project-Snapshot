"""가림 CNN 을 WIDER 단체 사진으로 평가 (YuNet 신뢰도 규칙과 비교)
- 대상: evaluate.load() (EVAL_SUBSET=dev|holdout). 가로 640 (카메라 해상도)
- 라벨 (보임 기준 2026-09-27 thinkcat, 재검수한 labels.csv)
  - 보임: visible 1 / 가림: visible 0 이고 이유가 옆모습(side)이 아님 / 옆모습은 따로 셈 (yaw 규칙 대상)
- 얼굴: YuNet (신뢰도 0.7 이상 = 앱 기준) 검출과 정답을 IoU 매칭 (evaluate.match)
- CNN 입력: 앱과 같음 (yunet_cnn.model.square_crop: 박스 긴 변 정사각형, 가장자리 복제, RGB)
  - 모델 폴더의 occlusion_fp32.tflite, inference_config.json (입력 크기, 임계값)
- 출력: results_occ_{TAG}_{SUBSET}/faces.csv, 터미널 요약
  - AUC (가림 vs 보임), 보임 오판 1, 2, 5% 에서의 가림 recall, 노트북 임계값에서의 값
  - 비교: YuNet 신뢰도 규칙 (신뢰도가 낮을수록 가림)
  - yaw 규칙과 합친 결과 (보임 판정 = yaw < YAW_TH 이고 가림 아님)

실행: EVAL_SUBSET=dev app/.venv/bin/python evaluation/wider/occlusion_cnn_eval.py MODEL_DIR TAG
"""
import csv
import json
import math
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
from evaluate import ROOT, SUBSET, load, match  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
from photo_app.models.yunet_cnn import config as cnn_config  # noqa: E402
from photo_app.models.yunet_cnn.model import Classifier, square_crop  # noqa: E402

MODEL_DIR = Path(ARGS[0])
TAG = ARGS[1]
WIDTH = 640
OUT = HERE / f"results_occ_{TAG}_{SUBSET}"
FPRS = (0.01, 0.02, 0.05)


def kind(r):
    if r["visible"] == "1":
        return "visible"
    return "side" if r["hidden_reason"] == "side" else "occluded"


def run(cfg):
    det = cv2.FaceDetectorYN.create(str(cnn_config.YUNET_PATH), "", (320, 320), cnn_config.SCORE_TH)
    clf = Classifier(MODEL_DIR / "occlusion_fp32.tflite", cfg["input"]["size"], cnn_config.NUM_THREADS)
    rows = []
    for n, (path, labels) in enumerate(load().items(), 1):
        src = cv2.imread(str(WIDER / path))
        s = WIDTH / src.shape[1]
        img = cv2.resize(src, (WIDTH, round(src.shape[0] * s)), interpolation=cv2.INTER_AREA)
        det.setInputSize((img.shape[1], img.shape[0]))
        _, dets = det.detect(img)
        dets = [] if dets is None else list(dets)
        boxes = [tuple(int(v) for v in d[:4]) for d in dets]
        gts = [tuple(int(r[k]) * s for k in ("x", "y", "w", "h")) for r in labels]
        for gi, (pi, _) in match(gts, boxes).items():
            r = labels[gi]
            if r["visible"] not in ("0", "1"):
                continue
            d = dets[pi]
            (rx, ry), (lx, ly), (nx, _) = d[4:6], d[6:8], d[8:10]
            yaw = abs(nx - (rx + lx) / 2) / (math.hypot(lx - rx, ly - ry) or 1.0)
            rows.append({"image": path, "face_id": r["face_id"], "kind": kind(r), "hidden_reason": r["hidden_reason"],
                         "h_px": boxes[pi][3], "yunet_score": round(float(d[14]), 4), "yaw": round(float(yaw), 3),
                         "occ_score": round(clf(square_crop(img, boxes[pi])), 4)})
        print(f"\r{n}", end="", flush=True)
    print()
    OUT.mkdir(exist_ok=True)
    with open(OUT / "faces.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def auc(pos, neg):
    p, n = np.array(pos)[:, None], np.array(neg)[None, :]
    return float(np.mean(p > n) + 0.5 * np.mean(p == n))


def at_fpr(pos, neg, fpr):
    """보임 오판 <= fpr 인 가장 낮은 기준 -> (기준, 가림 recall, 실제 보임 오판)"""
    th = np.quantile(neg, 1 - fpr, method="higher") + 1e-9
    return th, float(np.mean(np.array(pos) >= th)), float(np.mean(np.array(neg) >= th))


def pct(v):
    return f"{v:.0%}"


def summary(cfg):
    rows = list(csv.DictReader(open(OUT / "faces.csv")))
    occ = [r for r in rows if r["kind"] == "occluded"]
    vis = [r for r in rows if r["kind"] == "visible"]
    side = [r for r in rows if r["kind"] == "side"]
    print(f"대상: {SUBSET}, 가로 {WIDTH}, YuNet >= {cnn_config.SCORE_TH}, 모델 {MODEL_DIR}")
    print(f"얼굴: 가림 {len(occ)}, 보임 {len(vis)}, 옆모습 {len(side)} (옆모습은 AUC 에서 제외)")
    methods = {
        "가림 CNN": lambda r: float(r["occ_score"]),
        "YuNet 신뢰도 (낮을수록 가림)": lambda r: -float(r["yunet_score"]),
    }
    print(f"\n{'방식':<26} {'AUC':>6}  " + "  ".join(f"보임 오판 {pct(f)}: 가림 recall" for f in FPRS))
    for name, f in methods.items():
        pos, neg = [f(r) for r in occ], [f(r) for r in vis]
        cells = [pct(at_fpr(pos, neg, fpr)[1]) for fpr in FPRS]
        print(f"{name:<26} {auc(pos, neg):>6.3f}  " + "  ".join(f"{c:>22}" for c in cells))

    th = cfg["threshold"]
    yaw_th = cnn_config.YAW_TH
    o = lambda r: float(r["occ_score"]) >= th  # noqa: E731
    y = lambda r: float(r["yaw"]) >= yaw_th  # noqa: E731
    print(f"\n[노트북 임계값 {th:.3f}]")
    print(f"  가림 CNN만: 가림 recall {sum(map(o, occ))}/{len(occ)} ({np.mean([o(r) for r in occ]):.0%}), "
          f"보임 오판 {sum(map(o, vis))}/{len(vis)} ({np.mean([o(r) for r in vis]):.1%}), "
          f"옆모습을 가림으로 {sum(map(o, side))}/{len(side)}")
    hidden = lambda r: o(r) or y(r)  # noqa: E731
    print(f"  yaw 규칙과 합침: 가림 {sum(map(hidden, occ))}/{len(occ)} ({np.mean([hidden(r) for r in occ]):.0%}), "
          f"옆모습 {sum(map(hidden, side))}/{len(side)} ({np.mean([hidden(r) for r in side]):.0%}), "
          f"보임 오판 {sum(map(hidden, vis))}/{len(vis)} ({np.mean([hidden(r) for r in vis]):.1%})")
    print(f"  (비교) yaw 규칙만: 가림 {sum(map(y, occ))}/{len(occ)}, 옆모습 {sum(map(y, side))}/{len(side)}, "
          f"보임 오판 {sum(map(y, vis))}/{len(vis)} ({np.mean([y(r) for r in vis]):.1%})")
    print("\n[가림 이유별, 노트북 임계값]")
    for reason in sorted({r["hidden_reason"] for r in occ}):
        rs = [r for r in occ if r["hidden_reason"] == reason]
        print(f"  {reason:<7} {sum(map(o, rs))}/{len(rs)}")
    print("\n[얼굴 높이별 보임 오판, 노트북 임계값]")
    for lo, hi in ((0, 40), (40, 60), (60, 100), (100, 10**9)):
        rs = [r for r in vis if lo <= int(r["h_px"]) < hi]
        if rs:
            print(f"  {lo:>3}~{hi if hi < 10**9 else '':<4}px  {sum(map(o, rs))}/{len(rs)} ({np.mean([o(r) for r in rs]):.1%})")


if __name__ == "__main__":
    cfg = json.load(open(MODEL_DIR / "inference_config.json"))
    if not (OUT / "faces.csv").exists():
        run(cfg)
    summary(cfg)
