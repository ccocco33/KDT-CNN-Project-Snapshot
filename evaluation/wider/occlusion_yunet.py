"""가려진 얼굴 판정에 YuNet 신뢰도를 쓸 수 있는지 측정
- 대상: evaluate.load() (EVAL_SUBSET=dev|holdout). 가로 640 (카메라 해상도)
- YuNet 을 신뢰도 DETECT_TH 이상으로 한 번 검출 -> 정답 얼굴과 IoU 매칭 (evaluate.match)
- 얼굴 분류 (라벨)
  - 보임: visible 1
  - 옆모습: visible 0, hidden_reason side (yaw 규칙으로 잡는 대상)
  - 가림: visible 0, hidden_reason other | person | hand | hair | mask
- 저장: 얼굴별 신뢰도, 높이, yaw, 랜드마크 5점 (박스 기준 0~1)
- 요약: 분류별 검출 수, 신뢰도 분포, 신뢰도 기준별 가림 recall 과 보임 오판 (앱 검출 기준 0.7 이상 얼굴만)

실행: EVAL_SUBSET=dev app/.venv/bin/python evaluation/wider/occlusion_yunet.py
"""
import csv
import math
import sys
from pathlib import Path

import cv2

sys.argv = sys.argv[:1]
CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, RESULTS, WIDER  # noqa: E402

HERE = RESULTS   # 평가 결과 위치 (evaluation/paths.py)
sys.path.insert(0, str(CODE))
from evaluate import ROOT, SUBSET, load, match  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
from photo_app.models.yunet_cnn import config as cnn_config  # noqa: E402

WIDTH = 640
DETECT_TH = 0.3
APP_TH = cnn_config.SCORE_TH
OCCLUDED = {"other", "person", "hand", "hair", "mask"}
OUT = HERE / (f"results_occlusion" + (f"_{SUBSET}" if SUBSET else ""))
BINS = ((0, 40), (40, 60), (60, 100), (100, 10**9))


def kind(r):
    if r["visible"] == "1":
        return "보임"
    return "옆모습" if r["hidden_reason"] == "side" else "가림"


def run():
    det = cv2.FaceDetectorYN.create(str(cnn_config.YUNET_PATH), "", (320, 320), DETECT_TH)
    rows = []
    for n, (path, labels) in enumerate(load().items(), 1):
        src = cv2.imread(str(WIDER / path))
        s = WIDTH / src.shape[1]
        img = cv2.resize(src, (WIDTH, round(src.shape[0] * s)), interpolation=cv2.INTER_AREA)
        det.setInputSize((img.shape[1], img.shape[0]))
        _, dets = det.detect(img)
        dets = [] if dets is None else list(dets)
        boxes = [tuple(float(v) for v in d[:4]) for d in dets]
        gts = [tuple(int(r[k]) * s for k in ("x", "y", "w", "h")) for r in labels]
        m = match(gts, boxes)
        for gi, r in enumerate(labels):
            if r["visible"] not in ("0", "1"):
                continue
            row = {"image": path, "face_id": r["face_id"], "kind": kind(r), "hidden_reason": r["hidden_reason"],
                   "occlusion": r["occlusion"], "gt_h": round(gts[gi][3])}
            if gi in m:
                d = dets[m[gi][0]]
                x, y, w, h = boxes[m[gi][0]]
                (rx, ry), (lx, ly), (nx, _) = d[4:6], d[6:8], d[8:10]
                eye_dist = math.hypot(lx - rx, ly - ry) or 1.0
                row.update({"score": round(float(d[14]), 4), "h_px": round(h),
                            "yaw": round(abs(nx - (rx + lx) / 2) / eye_dist, 3)})
                for i, name in enumerate(("re", "le", "nose", "rm", "lm")):
                    row[f"{name}_x"] = round((float(d[4 + 2 * i]) - x) / w, 3)
                    row[f"{name}_y"] = round((float(d[5 + 2 * i]) - y) / h, 3)
            rows.append(row)
        print(f"\r{n}", end="", flush=True)
    print()
    OUT.mkdir(exist_ok=True)
    keys = list(max(rows, key=len))
    with open(OUT / "faces.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "-"


def q(v, p):
    v = sorted(v)
    return v[int(p * (len(v) - 1))] if v else float("nan")


def summary():
    rows = list(csv.DictReader(open(OUT / "faces.csv")))
    print(f"대상: {SUBSET or '전체'}, 가로 {WIDTH}, YuNet 검출 {DETECT_TH} 이상, 앱 기준 {APP_TH}")
    print("\n[분류별 검출] 앱 기준 이상 = 앱이 얼굴로 다루는 얼굴")
    for k in ("보임", "옆모습", "가림"):
        rs = [r for r in rows if r["kind"] == k]
        sc = [float(r["score"]) for r in rs if r.get("score")]
        app = [x for x in sc if x >= APP_TH]
        print(f"  {k:<4} 라벨 {len(rs):>5}  검출 {len(sc):>5}  앱 기준 이상 {pct(len(app), len(rs)):<16} "
              f"신뢰도 중앙값 {q(app, .5):.3f}  하위 10% {q(app, .1):.3f}  하위 25% {q(app, .25):.3f}")
    app_rows = [r for r in rows if r.get("score") and float(r["score"]) >= APP_TH]
    print("\n[가림 이유별] 앱 기준 이상 얼굴")
    for reason in sorted(OCCLUDED):
        sc = [float(r["score"]) for r in app_rows if r["hidden_reason"] == reason]
        if sc:
            print(f"  {reason:<7} {len(sc):>4}  신뢰도 중앙값 {q(sc, .5):.3f}")
    print("\n[신뢰도 기준별] 앱 기준 이상 얼굴에서 신뢰도 < 기준이면 보이지 않음으로 볼 때 (yaw 규칙과 별도, 함께)")
    print(f"  {'기준':<5} {'가림 recall':<16} {'보임 오판':<18} {'가림 recall (+yaw)':<18} {'보임 오판 (+yaw)':<18}")
    vis = [r for r in app_rows if r["kind"] == "보임"]
    occ = [r for r in app_rows if r["kind"] == "가림"]
    yaw_hidden = lambda r: float(r["yaw"]) >= cnn_config.YAW_TH  # noqa: E731
    for th in (0.75, 0.8, 0.85, 0.88, 0.9, 0.92):
        low = lambda r: float(r["score"]) < th  # noqa: E731
        print(f"  {th:<5} {pct(sum(map(low, occ)), len(occ)):<16} {pct(sum(map(low, vis)), len(vis)):<18} "
              f"{pct(sum(low(r) or yaw_hidden(r) for r in occ), len(occ)):<18} "
              f"{pct(sum(low(r) or yaw_hidden(r) for r in vis), len(vis)):<18}")
    print("\n[얼굴 높이 구간별 신뢰도 중앙값] 앱 기준 이상")
    for lo, hi in BINS:
        cells = []
        for k in ("보임", "가림"):
            sc = [float(r["score"]) for r in app_rows if r["kind"] == k and lo <= int(r["h_px"]) < hi]
            cells.append(f"{k} {len(sc):>4}개 {q(sc, .5):.3f}")
        print(f"  {lo:>3}~{hi if hi < 10**9 else '':<4}px  " + "  /  ".join(cells))


if __name__ == "__main__":
    if not (OUT / "faces.csv").exists():
        run()
    summary()
