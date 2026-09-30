"""사진 단위 비교: blendshape(yunet_landmark) vs CNN(smile, eye) vs 섞음
- 대상: evaluate.py 와 같음 (EVAL_SUBSET=dev 권장)
- 사진 가로 640 (카메라 해상도). YuNet 은 신뢰도 0.5 이상을 한 번 검출해 두고 기준(0.7, 0.9)별로 걸러 씀
- 얼굴마다 한 번씩 계산해 저장
  - blendshape: yunet_landmark 와 같음 (여백 0.5 로 잘라 FaceLandmarker). 못 찾으면 빈칸
  - CNN: compare_cnn 과 같은 입력 (박스 긴 변 정사각형, eye 는 위쪽 EYE_TOP. 모델 폴더의 설정에서 읽음). 모델 안에 좌우 뒤집기 평균 포함
  - yaw: 코가 두 눈 가운데에서 벗어난 정도 / 눈 사이 거리
  - 처리 시간 (이 PC): FaceLandmarker, CNN smile, CNN eye 각각
- 사진 판정 (앱 meets_condition 과 같음): 검출 얼굴 1명 이상, 모든 얼굴이 웃고 눈 뜸
  - blendshape 판정 불가(FaceLandmarker 미검출)는 조건 미충족
  - yaw 규칙을 켜면 yaw >= YAW_TH 인 얼굴은 "보이지 않음" -> 조건 미충족
- 정답: 모든 정답 얼굴이 보임, 웃음, 눈 뜸(선글라스류 포함) (evaluate.label_ok)
  - 보이는 얼굴 중 판단 불가(-1)가 있는 사진은 정답을 모름 -> 두 가지로 계산: 미충족으로 / 제외
- 출력: results_photo_{TAG}[_SUBSET]/faces.csv (검출 얼굴별 점수), 터미널 요약

실행: EVAL_SUBSET=dev app/.venv/bin/python evaluation/wider/photo_compare.py MODEL_DIR TAG
"""
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]   # evaluate.py 는 import 할 때 sys.argv 를 읽음

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, RESULTS, WIDER  # noqa: E402

HERE = RESULTS   # 평가 결과 위치 (evaluation/paths.py)
sys.path.insert(0, str(CODE))
from compare_cnn import Classifier, eye_top, square_crop  # noqa: E402
from evaluate import ROOT, SUBSET, has_unknown, label_ok, load, match  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
from photo_app.models.landmark import config as lm_config  # noqa: E402
from photo_app.models.landmark import rules  # noqa: E402
from photo_app.models.yunet_landmark import YunetLandmarkModel  # noqa: E402

MODEL_DIR = Path(ARGS[0])
TAG = ARGS[1]
EYE_TOP = eye_top(MODEL_DIR / "eye_fp32.tflite")
WIDTH = 640
DETECT_TH = 0.5
YAW_TH = 0.5
OUT = HERE / (f"results_photo_{TAG}" + (f"_{SUBSET}" if SUBSET else ""))


def run():
    lm = YunetLandmarkModel(score_th=DETECT_TH)
    smile_cnn, eye_cnn = Classifier(MODEL_DIR / "smile_fp32.tflite", 128), Classifier(MODEL_DIR / "eye_fp32.tflite", 160)
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
        photo_ok = int(all(label_ok(r) for r in labels))
        for pi, (d, box) in enumerate(zip(dets, boxes)):
            (rx, ry), (lx, ly), (nx, _) = d[4:6], d[6:8], d[8:10]
            eye_dist = math.hypot(lx - rx, ly - ry) or 1.0
            face = cv2.cvtColor(square_crop(img, box), cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            bs = lm._blendshapes(rgb, box)
            t1 = time.perf_counter()
            cs = smile_cnn(face)
            t2 = time.perf_counter()
            ce = eye_cnn(face[:max(1, int(face.shape[0] * EYE_TOP))])
            t3 = time.perf_counter()
            gi = pred_to_gt.get(pi)
            rows.append({
                "image": path, "photo_label_ok": photo_ok, "gt_faces": len(labels),
                "face_label_ok": "" if gi is None else int(label_ok(labels[gi])),
                "h_px": box[3], "yunet_score": round(float(d[14]), 3), "yaw": round(abs(nx - (rx + lx) / 2) / eye_dist, 3),
                "bs_smile": "" if bs is None else round(rules.smile_score(bs), 4),
                "bs_eye": "" if bs is None else round(1 - rules.blink_score(bs), 4),
                "cnn_smile": round(cs, 4), "cnn_eye": round(ce, 4),
                "lm_ms": round((t1 - t0) * 1000, 2), "cnn_smile_ms": round((t2 - t1) * 1000, 2),
                "cnn_eye_ms": round((t3 - t2) * 1000, 2),
            })
        if not dets:   # 검출 얼굴이 없는 사진도 사진 목록에 남김
            rows.append({"image": path, "photo_label_ok": photo_ok, "gt_faces": len(labels), "h_px": ""})
        print(f"\r{n}", end="", flush=True)
    print()
    OUT.mkdir(exist_ok=True)
    keys = list(max(rows, key=len))
    with open(OUT / "faces.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def face_ok(r, method, smile_th, eye_th, use_yaw):
    """얼굴 하나의 조건 충족 (웃음 + 눈 뜸)"""
    if use_yaw and float(r["yaw"]) >= YAW_TH:
        return False
    bs_ok = r["bs_smile"] != ""
    if method == "blendshape":
        return bs_ok and float(r["bs_smile"]) >= lm_config.SMILE_TH and float(r["bs_eye"]) > 1 - lm_config.BLINK_TH
    if method == "cnn":
        return float(r["cnn_smile"]) >= smile_th and float(r["cnn_eye"]) >= eye_th
    if method == "bs_smile+cnn_eye":
        return bs_ok and float(r["bs_smile"]) >= lm_config.SMILE_TH and float(r["cnn_eye"]) >= eye_th
    if method == "cnn_smile+bs_eye":
        return bs_ok and float(r["cnn_smile"]) >= smile_th and float(r["bs_eye"]) > 1 - lm_config.BLINK_TH
    raise ValueError(method)


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else f"{a}/0"


def photo_table(title, by_photo, cfg):
    """사진 조건 충족 precision, recall 표. by_photo: {사진: [검출 얼굴 행]}"""
    positives = sum(rs[0]["photo_label_ok"] == "1" for rs in by_photo.values())
    print(f"\n[사진 조건 충족, {title}] 사진 {len(by_photo)}장 (정답 충족 {positives}). 양성 = 충족")
    print(f"  {'YuNet':>5} {'yaw':>4} {'방식':<18} {'기준값':<14} {'precision':>14} {'recall':>14} {'정확도':>14}")
    thresholds = {"0.5": (0.5, 0.5), "노트북": (cfg["smile"], cfg["eye"])}
    for det_th in (0.9, 0.7):
        for use_yaw in (False, True):
            for method in ("blendshape", "cnn", "bs_smile+cnn_eye", "cnn_smile+bs_eye"):
                for tname, (sth, eth) in thresholds.items():
                    if method == "blendshape" and tname != "0.5":
                        continue
                    tp = fp = fn = tn = 0
                    for rs in by_photo.values():
                        faces = [r for r in rs if r["h_px"] != "" and float(r["yunet_score"]) >= det_th]
                        pred = len(faces) > 0 and all(face_ok(r, method, sth, eth, use_yaw) for r in faces)
                        label = rs[0]["photo_label_ok"] == "1"
                        tp += pred and label
                        fp += pred and not label
                        fn += (not pred) and label
                        tn += (not pred) and not label
                    tl = "-" if method == "blendshape" else tname
                    print(f"  {det_th:>5} {'o' if use_yaw else 'x':>4} {method:<18} {tl:<14} {pct(tp, tp + fp):>14} "
                          f"{pct(tp, tp + fn):>14} {pct(tp + tn, len(by_photo)):>14}")



def summary():
    rows = list(csv.DictReader(open(OUT / "faces.csv")))
    by_photo = defaultdict(list)
    for r in rows:
        by_photo[r["image"]].append(r)
    unknown = {img for img, faces in load().items() if has_unknown(faces)}
    positives = sum(rs[0]["photo_label_ok"] == "1" for rs in by_photo.values())
    print(f"대상: {SUBSET or '전체'}, 사진 {len(by_photo)}장 (정답 충족 {positives}, 판단 불가 얼굴이 있는 사진 {len(unknown)}), "
          f"가로 {WIDTH}px, 모델 {MODEL_DIR}, eye 입력 위쪽 {EYE_TOP}")
    cfg = {k: json.load(open(MODEL_DIR / f"{k}_inference_config.json"))["threshold"] for k in ("smile", "eye")}
    print(f"CNN 노트북 기준값: smile {cfg['smile']:.3f}, eye {cfg['eye']:.3f} / blendshape: 웃음 >= {lm_config.SMILE_TH}, "
          f"눈 감음 < {lm_config.BLINK_TH}")

    for title, photos in (("판단 불가 얼굴이 있는 사진은 미충족", by_photo),
                          ("판단 불가 얼굴이 있는 사진 제외", {k: v for k, v in by_photo.items() if k not in unknown})):
        photo_table(title, photos, cfg)

    faces = [r for r in rows if r["h_px"] != ""]
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    lm_ms = [float(r["lm_ms"]) for r in faces]
    cnn_ms = [float(r["cnn_smile_ms"]) + float(r["cnn_eye_ms"]) for r in faces]
    print(f"\n[얼굴당 처리 시간, 이 PC, 중앙값] FaceLandmarker {med(lm_ms):.1f} ms / CNN smile+eye {med(cnn_ms):.1f} ms "
          f"(smile {med([float(r['cnn_smile_ms']) for r in faces]):.1f}, eye {med([float(r['cnn_eye_ms']) for r in faces]):.1f})")


if __name__ == "__main__":
    if not (OUT / "faces.csv").exists():
        run()
    summary()
