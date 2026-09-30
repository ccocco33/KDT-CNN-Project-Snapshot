"""거리(최소 얼굴 높이), 가림(얼굴이 보이지 않음), 안경류 요구사항 검토용 측정
- 대상: evaluate.py 와 같음. EVAL_SUBSET=dev 권장 (holdout 은 최종 확인용)
- 사진 가로 640 (카메라 해상도), YuNet score >= 0.7, 정답과 IoU >= 0.3 으로 짝
- 얼굴별 측정
  - YuNet: 검출 여부, 신뢰도, 고개 돌림(yaw: 코가 두 눈 가운데에서 벗어난 정도 / 눈 사이 거리, 절댓값)
  - FaceLandmarker (여백 0.5 로 잘라 넣음): 검출 여부, 웃음 점수, 눈 뜸 점수 (1 - 눈 감음 최대)
  - CNN: smile (SMILE_TFLITE, 출력 1 = 웃음), eye (EYE_TFLITE, 출력 1 = 뜸, 얼굴 위쪽 60%)
- 출력: results_requirements[_SUBSET]/faces.csv, 터미널 요약

실행: EVAL_SUBSET=dev app/.venv/bin/python evaluation/wider/requirements_check.py SMILE_TFLITE EYE_TFLITE
"""
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]   # evaluate.py 는 import 할 때 sys.argv 를 읽음

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, RESULTS, WIDER  # noqa: E402

HERE = RESULTS   # 평가 결과 위치 (evaluation/paths.py)
sys.path.insert(0, str(CODE))
from compare_cnn import Classifier, auc, eye_top, square_crop  # noqa: E402
from evaluate import ROOT, SUBSET, load, match  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
from photo_app.models.landmark import rules  # noqa: E402
from photo_app.models.yunet_landmark import YunetLandmarkModel  # noqa: E402

SMILE_PATH, EYE_PATH = Path(ARGS[0]), Path(ARGS[1])
EYE_TOP = eye_top(EYE_PATH)
WIDTH = 640
OUT = HERE / ("results_requirements" + (f"_{SUBSET}" if SUBSET else ""))
H_BINS = ((0, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 10**9))


def main():
    lm = YunetLandmarkModel(score_th=0.7)
    smile_cnn, eye_cnn = Classifier(SMILE_PATH, 128), Classifier(EYE_PATH, 160)
    rows = []
    for path, labels in load().items():
        src = cv2.imread(str(WIDER / path))
        s = WIDTH / src.shape[1]
        img = cv2.resize(src, (WIDTH, round(src.shape[0] * s)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lm._detector.setInputSize((img.shape[1], img.shape[0]))
        _, dets = lm._detector.detect(img)
        dets = [] if dets is None else list(dets)
        boxes = [tuple(int(v) for v in d[:4]) for d in dets]
        gts = [tuple(int(r[k]) * s for k in ("x", "y", "w", "h")) for r in labels]
        m = match(gts, boxes)
        for gi, r in enumerate(labels):
            row = {"image": path, "face_id": r["face_id"], "h_px": round(gts[gi][3]), "visible": r["visible"],
                   "hidden_reason": r["hidden_reason"], "smile": r["smile"], "eyes_open": r["eyes_open"],
                   "detected": 0, "yunet_score": "", "yaw": "", "lm_found": "", "bs_smile": "", "bs_eye": "",
                   "cnn_smile": "", "cnn_eye": ""}
            if gi in m:
                d = dets[m[gi][0]]
                box = boxes[m[gi][0]]
                (rx, ry), (lx, ly), (nx, ny) = d[4:6], d[6:8], d[8:10]
                eye_dist = math.hypot(lx - rx, ly - ry) or 1.0
                face = cv2.cvtColor(square_crop(img, box), cv2.COLOR_BGR2RGB)
                bs = lm._blendshapes(rgb, box)
                row.update({
                    "detected": 1, "yunet_score": round(float(d[14]), 3),
                    "yaw": round(abs(nx - (rx + lx) / 2) / eye_dist, 3),
                    "lm_found": int(bs is not None),
                    "bs_smile": "" if bs is None else round(rules.smile_score(bs), 4),
                    "bs_eye": "" if bs is None else round(1 - rules.blink_score(bs), 4),
                    "cnn_smile": round(smile_cnn(face), 4),
                    "cnn_eye": round(eye_cnn(face[:max(1, int(face.shape[0] * EYE_TOP))]), 4),
                })
            rows.append(row)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "faces.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary(rows)


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "-"


def fmt_auc(rs, label, pos, key):
    p = [float(r[key]) for r in rs if r[label] in pos and r[key] != ""]
    n = [float(r[key]) for r in rs if r[label] not in pos and r[key] != ""]
    return f"{auc(p, n):.2f}" if len(p) >= 5 and len(n) >= 5 else "-"


def summary(rows):
    vis = [r for r in rows if r["visible"] == "1"]
    print(f"대상: {SUBSET or '전체'}, 얼굴 {len(rows)} (보임 {len(vis)}), 가로 {WIDTH}px")

    print("\n[거리] 보이는 얼굴, 얼굴 높이별 (AUC 는 검출된 얼굴. 판단 불가 라벨 제외)")
    print(f"  {'높이':>8} {'얼굴':>5} {'검출률':>11} {'웃음 CNN':>8} {'웃음 BS':>7} {'눈 n(감음)':>10} {'눈 CNN':>6} {'눈 BS':>5} "
          f"{'뜬 눈->감음 CNN':>14} {'BS':>5}")
    for lo, hi in H_BINS:
        rs = [r for r in vis if lo <= r["h_px"] < hi]
        det = [r for r in rs if r["detected"] == 1]
        sm = [r for r in det if r["smile"] in ("0", "1")]
        ey = [r for r in det if r["eyes_open"] in ("0", "1")]
        opened = [r for r in ey if r["eyes_open"] == "1"]
        fc = sum(float(r["cnn_eye"]) < 0.5 for r in opened)
        fb = [r for r in opened if r["bs_eye"] != ""]
        name = f"{lo}~{hi if hi < 10**9 else ''}"
        print(f"  {name:>8} {len(rs):>5} {pct(len(det), len(rs)):>11} "
              f"{fmt_auc(sm, 'smile', ('1',), 'cnn_smile'):>8} {fmt_auc(sm, 'smile', ('1',), 'bs_smile'):>7} "
              f"{len(ey):>5}({sum(r['eyes_open'] == '0' for r in ey):>2}) "
              f"{fmt_auc(ey, 'eyes_open', ('1',), 'cnn_eye'):>6} {fmt_auc(ey, 'eyes_open', ('1',), 'bs_eye'):>5} "
              f"{pct(fc, len(opened)):>14} {pct(sum(float(r['bs_eye']) < 0.5 for r in fb), len(fb)):>5}")

    print("\n[가림] 보이지 않는 얼굴(visible=0)")
    hid = [r for r in rows if r["visible"] == "0"]
    print(f"  YuNet 검출: {pct(sum(r['detected'] for r in hid), len(hid))} (못 찾으면 안내도 판정도 없이 사라짐)")
    for reason in sorted({r["hidden_reason"] for r in hid}):
        rs = [r for r in hid if r["hidden_reason"] == reason]
        print(f"    {reason:<7} 검출 {pct(sum(r['detected'] for r in rs), len(rs))}, "
              f"검출 중 FaceLandmarker 미검출 {pct(sum(r['lm_found'] == 0 for r in rs if r['detected']), sum(r['detected'] for r in rs))}")
    dv = [r for r in vis if r["detected"]]
    dh = [r for r in hid if r["detected"]]
    print("  검출된 얼굴에서 '보이지 않음' 판정 방법 비교 (양성 = 보이지 않음)")
    tp = sum(r["lm_found"] == 0 for r in dh)
    fp = sum(r["lm_found"] == 0 for r in dv)
    print(f"    FaceLandmarker 미검출 = 보이지 않음: 보이지 않는 얼굴 중 잡음 {pct(tp, len(dh))}, "
          f"보이는 얼굴을 잘못 잡음 {pct(fp, len(dv))}")
    for key, name, sign in (("yaw", "고개 돌림 yaw (클수록 보이지 않음)", 1), ("yunet_score", "YuNet 신뢰도 (낮을수록 보이지 않음)", -1)):
        p = [sign * float(r[key]) for r in dh]
        n = [sign * float(r[key]) for r in dv]
        print(f"    {name}: AUC {auc(p, n):.3f}")
    side = [r for r in dh if r["hidden_reason"] == "side"]
    print(f"    yaw 만으로 옆모습 구분 AUC: {auc([float(r['yaw']) for r in side], [float(r['yaw']) for r in dv]):.3f} "
          f"(옆모습 {len(side)}, 보임 {len(dv)})")
    for th in (0.3, 0.4, 0.5):
        print(f"      yaw >= {th}: 보이지 않는 얼굴 중 잡음 {pct(sum(float(r['yaw']) >= th for r in dh), len(dh))}, "
              f"보이는 얼굴을 잘못 잡음 {pct(sum(float(r['yaw']) >= th for r in dv), len(dv))}")

    print("\n[안경류] eyes_open = 2 (안경류로 눈이 가려짐, 조건 충족으로 처리해야 함)")
    gl = [r for r in rows if r["eyes_open"] == "2" and r["detected"]]
    print(f"  검출 {len(gl)}개: 눈 뜸 판정 CNN {pct(sum(float(r['cnn_eye']) >= 0.5 for r in gl), len(gl))}, "
          f"BS {pct(sum(r['bs_eye'] != '' and float(r['bs_eye']) >= 0.5 for r in gl), sum(r['bs_eye'] != '' for r in gl))}")


if __name__ == "__main__":
    main()
