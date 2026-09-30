"""yunet_landmark (베이스라인) vs yunet_cnn 비교 (앱 판정 규칙, 앱 config 값 그대로)
- 대상: evaluate.load() (EVAL_SUBSET=dev|holdout). 가로 640 (카메라 해상도). 라벨은 재검수한 labels.csv (보임 기준 2026-09-27)
- 사진마다 YuNet (신뢰도 >= SCORE_TH) 한 번 검출 -> 얼굴마다 두 모델의 재료를 모두 계산해 저장 (faces.csv)
  - 공통: yaw (YuNet 랜드마크), 얼굴 높이
  - yunet_cnn: 가림, smile, eye CNN 점수 (앱과 같은 자르기)
  - yunet_landmark: 여백을 두고 자른 얼굴의 FaceLandmarker 결과 (찾았는지, blendshape 웃음, 눈 감음)
  - 얼굴당 처리 시간 (이 PC)
- 판정 규칙 (config 값은 실행할 때 읽음. 요약만 다시 내면 바뀐 값이 반영됨)
  - yunet_landmark: 보임 = yaw < YAW_TH. 웃음 = mouthSmile 평균 >= SMILE_TH. 눈 뜸 = eyeBlink 최대 < BLINK_TH. FaceLandmarker 미검출이면 웃음, 눈 None
  - yunet_landmark+miss: 위 + FaceLandmarker 미검출이면 보이지 않음 (보임 판정 보강 후보)
  - yunet_cnn: 보임 = yaw < YAW_TH 이고 가림 < OCC_TH. 웃음 = smile >= SMILE_TH. 눈 뜸 = eye >= EYE_TH
- 사진 조건 충족 (앱 meets_condition): 검출 얼굴 1명 이상, 모든 얼굴이 보이고 웃고 눈 뜸 (None 은 미충족)
  - 정답: 모든 정답 얼굴이 보임, 웃음, 눈 뜸 (evaluate.label_ok). 판단 불가(-1) 사진은 미충족으로 / 제외 두 가지
- 얼굴 단위 (정답과 짝지어진 얼굴)
  - 보임: 가림, 옆모습을 보이지 않음으로 본 비율 / 보이는 얼굴을 보이지 않음으로 본 비율
  - 웃음, 눈: 정답이 보이는 얼굴 중 판정 가능한 얼굴. 판정 불가(None) 비율은 따로
- 출력: results_models_{SUBSET}/faces.csv, 터미널 요약

실행: EVAL_SUBSET=dev app/.venv/bin/python evaluation/wider/compare_models.py
"""
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.argv = sys.argv[:1]
CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, RESULTS, WIDER  # noqa: E402

HERE = RESULTS   # 평가 결과 위치 (evaluation/paths.py)
sys.path.insert(0, str(CODE))
from evaluate import ROOT, SUBSET, has_unknown, label_ok, load, match  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
from photo_app.models.landmark import rules  # noqa: E402
from photo_app.models.yunet_cnn import config as cnn_cfg  # noqa: E402
from photo_app.models.yunet_cnn.model import Classifier, lower, square_crop, upper  # noqa: E402
from photo_app.models.yunet_landmark import YunetLandmarkModel  # noqa: E402
from photo_app.models.yunet_landmark import config as lm_cfg  # noqa: E402

WIDTH = 640
OUT = HERE / f"results_models_{SUBSET}"
MODELS = ("yunet_landmark", "yunet_landmark+miss", "yunet_cnn")


def kind(r):
    if r["visible"] == "1":
        return "visible"
    return "side" if r["hidden_reason"] == "side" else "occluded"


def run():
    lm = YunetLandmarkModel(score_th=cnn_cfg.SCORE_TH)
    occ = Classifier(cnn_cfg.OCC_PATH, cnn_cfg.OCC_SIZE, cnn_cfg.NUM_THREADS)
    smile = Classifier(cnn_cfg.SMILE_PATH, cnn_cfg.SMILE_SIZE, cnn_cfg.NUM_THREADS)
    eye = Classifier(cnn_cfg.EYE_PATH, cnn_cfg.EYE_SIZE, cnn_cfg.NUM_THREADS)
    rows = []
    for n, (path, labels) in enumerate(load().items(), 1):
        src = cv2.imread(str(WIDER / path))
        s = WIDTH / src.shape[1]
        img = cv2.resize(src, (WIDTH, round(src.shape[0] * s)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lm._detector.setInputSize((img.shape[1], img.shape[0]))
        _, dets = lm._detector.detect(img)
        dets = [] if dets is None else list(dets)
        boxes = [tuple(int(v) for v in d[:4]) for d in dets]
        gts = [tuple(int(r[k]) * s for k in ("x", "y", "w", "h")) for r in labels]
        pred_to_gt = {pi: gi for gi, (pi, _) in match(gts, boxes).items()}
        base = {"image": path, "photo_label_ok": int(all(label_ok(r) for r in labels)),
                "photo_unknown": int(has_unknown(labels))}
        for pi, (d, box) in enumerate(zip(dets, boxes)):
            yaw = rules.yaw((d[4], d[5]), (d[6], d[7]), (d[8], d[9]))
            crop = square_crop(img, box)
            t0 = time.perf_counter()
            bs = lm._blendshapes(rgb, box)
            t1 = time.perf_counter()
            o = occ(crop)
            t2 = time.perf_counter()
            sm = smile(lower(crop, cnn_cfg.SMILE_BOTTOM))
            t3 = time.perf_counter()
            ey = eye(upper(crop, cnn_cfg.EYE_TOP))
            t4 = time.perf_counter()
            gi = pred_to_gt.get(pi)
            g = labels[gi] if gi is not None else None
            rows.append({**base, "h_px": box[3], "yaw": round(yaw, 3),
                         "kind": "" if g is None else kind(g),
                         "smile_label": "" if g is None else g["smile"], "eye_label": "" if g is None else g["eyes_open"],
                         "lm_found": int(bs is not None),
                         "bs_smile": "" if bs is None else round(rules.smile_score(bs), 4),
                         "bs_blink": "" if bs is None else round(rules.blink_score(bs), 4),
                         "occ": round(o, 4), "cnn_smile": round(sm, 4), "cnn_eye": round(ey, 4),
                         "lm_ms": round((t1 - t0) * 1000, 2), "occ_ms": round((t2 - t1) * 1000, 2),
                         "smile_ms": round((t3 - t2) * 1000, 2), "eye_ms": round((t4 - t3) * 1000, 2)})
        if not dets:
            rows.append({**base, "h_px": ""})
        print(f"\r{n}", end="", flush=True)
    print()
    OUT.mkdir(exist_ok=True)
    keys = list(max(rows, key=len))
    with open(OUT / "faces.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def judge(r, model):
    """얼굴 하나 -> (보임, 웃음, 눈 뜸). 판정 안 함은 None"""
    yaw = float(r["yaw"])
    if model == "yunet_cnn":
        vis = yaw < cnn_cfg.YAW_TH and float(r["occ"]) < cnn_cfg.OCC_TH
        if not vis:
            return False, None, None
        return True, float(r["cnn_smile"]) >= cnn_cfg.SMILE_TH, float(r["cnn_eye"]) >= cnn_cfg.EYE_TH
    vis = yaw < lm_cfg.YAW_TH
    found = r["lm_found"] == "1"
    if model == "yunet_landmark+miss":
        vis = vis and found
    if not vis:
        return False, None, None
    if not found:
        return True, None, None
    return True, float(r["bs_smile"]) >= lm_cfg.SMILE_TH, float(r["bs_blink"]) < lm_cfg.BLINK_TH


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "-"


def summary():
    rows = list(csv.DictReader(open(OUT / "faces.csv")))
    by_photo = defaultdict(list)
    for r in rows:
        by_photo[r["image"]].append(r)
    faces = [r for r in rows if r["h_px"] != ""]
    matched = [r for r in faces if r["kind"]]
    print(f"대상: {SUBSET}, 사진 {len(by_photo)}장, 검출 얼굴 {len(faces)} (정답과 짝 {len(matched)}), 가로 {WIDTH}")
    print(f"설정: yunet_landmark YAW_TH {lm_cfg.YAW_TH}, SMILE_TH {lm_cfg.SMILE_TH}, BLINK_TH {lm_cfg.BLINK_TH} / "
          f"yunet_cnn {cnn_cfg.MODEL_VERSION} YAW_TH {cnn_cfg.YAW_TH}, OCC_TH {cnn_cfg.OCC_TH}, SMILE_TH {cnn_cfg.SMILE_TH}, EYE_TH {cnn_cfg.EYE_TH}")

    print("\n[사진 조건 충족] precision (판단 불가 사진 미충족 / 제외), recall")
    for model in MODELS:
        out = []
        for drop_unknown in (False, True):
            tp = fp = fn = 0
            for rs in by_photo.values():
                if drop_unknown and rs[0]["photo_unknown"] == "1":
                    continue
                fs = [r for r in rs if r["h_px"] != ""]
                pred = bool(fs) and all(all(v is True for v in judge(r, model)) for r in fs)
                lab = rs[0]["photo_label_ok"] == "1"
                tp += pred and lab
                fp += pred and not lab
                fn += (not pred) and lab
            out.append((tp, fp, fn))
        (tp, fp, fn), (tp2, fp2, _) = out
        print(f"  {model:<20} precision {pct(tp, tp + fp):<14} / {pct(tp2, tp2 + fp2):<14} recall {pct(tp, tp + fn)}")

    print("\n[얼굴 보임] 보이지 않음으로 판정한 비율")
    for model in MODELS:
        cells = []
        for k in ("occluded", "side", "visible"):
            rs = [r for r in matched if r["kind"] == k]
            cells.append(f"{k} {pct(sum(judge(r, model)[0] is False for r in rs), len(rs))}")
        print(f"  {model:<20} " + "  ".join(cells))

    vis_faces = [r for r in matched if r["kind"] == "visible"]
    print("\n[얼굴 웃음, 눈] 정답이 보이는 얼굴, 모델이 보임으로 판정하고 판정 가능한 얼굴만. 판정 불가는 따로")
    for model in MODELS:
        judged = [(r, judge(r, model)) for r in vis_faces]
        usable = [(r, j) for r, j in judged if j[0] and j[1] is not None]
        none = sum(1 for r, j in judged if j[0] and j[1] is None)
        sm1 = [j[1] for r, j in usable if r["smile_label"] == "1"]
        sm0 = [j[1] for r, j in usable if r["smile_label"] == "0"]
        op = [j[2] for r, j in usable if r["eye_label"] in ("1", "2")]
        cl = [j[2] for r, j in usable if r["eye_label"] == "0"]
        print(f"  {model:<20} 웃음 recall {pct(sum(sm1), len(sm1)):<16} 안 웃음을 웃음으로 {pct(sum(sm0), len(sm0)):<14} "
              f"감은 눈 recall {pct(sum(not v for v in cl), len(cl)):<14} 뜬 눈을 감음으로 {pct(sum(not v for v in op), len(op)):<16} "
              f"판정 불가 {none}")

    med = lambda k: float(np.median([float(r[k]) for r in faces]))  # noqa: E731
    print(f"\n[얼굴당 처리 시간, 이 PC, 중앙값] FaceLandmarker {med('lm_ms'):.1f} ms / "
          f"CNN 가림 {med('occ_ms'):.1f}, smile {med('smile_ms'):.1f}, eye {med('eye_ms'):.1f} ms")


if __name__ == "__main__":
    if not (OUT / "faces.csv").exists():
        run()
    summary()
