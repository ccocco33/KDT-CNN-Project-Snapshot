"""YuNet 박스 높이와 WIDER 정답 박스 높이 비교 (MIN_FACE_H 기준 확인)
- 대상: evaluate.py 와 같음 (EVAL_SUBSET=dev 권장). 보이는 얼굴(visible=1) 중 검출된 얼굴
- 사진 가로 640, YuNet score >= 0.7, IoU >= 0.3 으로 짝
- yaw 계산 시간도 함께 측정 (얼굴당)
- 출력: results_box_height[_SUBSET]/faces.csv, 터미널 요약

실행: EVAL_SUBSET=dev app/.venv/bin/python evaluation/wider/box_height_check.py
"""
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.argv = sys.argv[:1]
CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, RESULTS, WIDER  # noqa: E402

HERE = RESULTS   # 평가 결과 위치 (evaluation/paths.py)
sys.path.insert(0, str(CODE))
from evaluate import ROOT, SUBSET, load, match  # noqa: E402

YUNET = ROOT / "app/models/face_detection_yunet_2023mar.onnx"
WIDTH = 640
OUT = HERE / ("results_box_height" + (f"_{SUBSET}" if SUBSET else ""))
BINS = ((0, 40), (40, 50), (50, 60), (60, 80), (80, 10**9))


def yaw(d):
    (rx, ry), (lx, ly), (nx, _) = d[4:6], d[6:8], d[8:10]
    return abs(nx - (rx + lx) / 2) / (math.hypot(lx - rx, ly - ry) or 1.0)


def main():
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), score_threshold=0.7)
    rows, yaw_ns, detect_ms = [], [], []
    for path, labels in load().items():
        src = cv2.imread(str(WIDER / path))
        s = WIDTH / src.shape[1]
        img = cv2.resize(src, (WIDTH, round(src.shape[0] * s)), interpolation=cv2.INTER_AREA)
        det.setInputSize((img.shape[1], img.shape[0]))
        t = time.perf_counter()
        _, dets = det.detect(img)
        detect_ms.append((time.perf_counter() - t) * 1000)
        dets = [] if dets is None else list(dets)
        for d in dets:
            t = time.perf_counter_ns()
            yaw(d)
            yaw_ns.append(time.perf_counter_ns() - t)
        gts = [tuple(int(r[k]) * s for k in ("x", "y", "w", "h")) for r in labels]
        for gi, (pi, v) in match(gts, [tuple(d[:4]) for d in dets]).items():
            if labels[gi]["visible"] != "1":
                continue
            rows.append({"image": path, "face_id": labels[gi]["face_id"], "gt_h": round(gts[gi][3], 1),
                         "yunet_h": round(float(dets[pi][3]), 1), "iou": round(v, 3)})
    OUT.mkdir(exist_ok=True)
    with open(OUT / "faces.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    gt = np.array([r["gt_h"] for r in rows])
    yn = np.array([r["yunet_h"] for r in rows])
    ratio = yn / gt
    print(f"대상: {SUBSET or '전체'}, 보이는 얼굴 중 검출 {len(rows)}개, 가로 {WIDTH}px")
    print("YuNet 높이 / 정답 높이 (정답 높이 구간별)")
    print(f"  {'정답 높이':>9} {'n':>5} {'중앙값':>7} {'25%':>6} {'75%':>6}")
    for lo, hi in BINS:
        m = (gt >= lo) & (gt < hi)
        q = np.percentile(ratio[m], [25, 50, 75])
        print(f"  {lo:>4}~{hi if hi < 10**9 else '':<4} {m.sum():>5} {q[1]:>7.2f} {q[0]:>6.2f} {q[2]:>6.2f}")
    print("정답 높이 50px 근처 (45~55) 얼굴의 YuNet 높이")
    m = (gt >= 45) & (gt < 55)
    print(f"  n {m.sum()}, 중앙값 {np.median(yn[m]):.1f}, 25% {np.percentile(yn[m], 25):.1f}, 75% {np.percentile(yn[m], 75):.1f}")
    print("기준별 판정 일치 (정답 높이 < 50 을 '작음'으로 볼 때)")
    for th in (40, 45, 50, 55, 60, 65):
        small_gt, small_yn = gt < 50, yn < th
        print(f"  YuNet < {th}: 정답 작음 중 잡음 {np.sum(small_gt & small_yn)}/{small_gt.sum()}, "
              f"정답 50 이상을 작음으로 {np.sum(~small_gt & small_yn)}/{(~small_gt).sum()}")
    print(f"yaw 계산 시간 (얼굴당, 파이썬): 중앙값 {np.median(yaw_ns) / 1000:.1f} us / YuNet 검출 (사진당): 중앙값 {np.median(detect_ms):.1f} ms")


if __name__ == "__main__":
    main()
