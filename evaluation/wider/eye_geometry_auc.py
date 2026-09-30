"""눈 뜸 판정 AUC: 베이스라인 blendshape vs FaceLandmarker 기하 지표 vs eye CNN (같은 얼굴에서)
- 대상: WIDER dev, holdout. 가로 640, YuNet (yunet_cnn SCORE_TH) 검출, 정답과 짝. 보이는 얼굴, 눈 라벨 1 (뜸) vs 0 (감음)
- 베이스라인 쪽: yunet_landmark 와 같은 자르기 (여백 CROP_MARGIN) 로 FaceLandmarker 1회 -> 두 가지 점수
  - blendshape: 1 - eyeBlink 최대 (앱 yunet_landmark 규칙과 같음)
  - 기하: 눈 열림 = 위 눈꺼풀 중앙 ~ 아래 눈꺼풀 중앙 거리 / 눈 가로 길이 (training/data/eye/levels/geometry_metrics.py 와 같은 계산)
    - 두 눈 평균 (geo_mean), 작은 쪽 (geo_min)
- CNN: 앱과 같은 자르기 (square_crop, 위쪽 EYE_TOP). 모델은 인자로 (기본 앱 eye)
- AUC 는 FaceLandmarker 가 찾은 얼굴만으로 모든 점수를 비교 (같은 얼굴 집합). CNN 은 전체 얼굴 AUC 도 따로
- 출력: results/eye_geometry_auc.csv (얼굴별 점수), 터미널 표 (전체, 얼굴 높이 구간)

실행: app/.venv/bin/python evaluation/wider/eye_geometry_auc.py [이름=eye_fp32.tflite 경로 ...]
"""
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]
CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import REPO, RESULTS, WIDER  # noqa: E402

sys.path.insert(0, str(CODE))
import evaluate  # noqa: E402

sys.path.insert(0, str(REPO / "app"))
import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402
from photo_app.models.landmark import rules  # noqa: E402
from photo_app.models.yunet_cnn import config as c  # noqa: E402
from photo_app.models.yunet_cnn.model import Classifier, square_crop, upper  # noqa: E402
from photo_app.models.yunet_landmark import config as lm_cfg  # noqa: E402

EYES = ((33, 133, 159, 145), (362, 263, 386, 374))   # (눈꼬리, 눈꼬리, 위 눈꺼풀 중앙, 아래 눈꺼풀 중앙)
BINS = ((0, 40), (40, 60), (60, 100), (100, 10000))
OUT = RESULTS / "eye_geometry_auc.csv"
CNNS = dict(a.split("=", 1) for a in ARGS) or {"app": str(c.EYE_PATH)}


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def eye_open(landmarks, w, h) -> tuple[float, float]:
    """두 눈의 눈 열림 (평균, 작은 쪽). 두 눈 중심 선을 수평으로 돌린 뒤 계산"""
    p = np.array([[q.x * w, q.y * h] for q in landmarks])
    centers = [(p[a] + p[b]) / 2 for a, b, _, _ in EYES]
    angle = math.atan2(centers[1][1] - centers[0][1], centers[1][0] - centers[0][0])
    rot = np.array([[math.cos(-angle), -math.sin(-angle)], [math.sin(-angle), math.cos(-angle)]])
    p = p @ rot.T
    opens = [np.linalg.norm(p[top] - p[bottom]) / (np.linalg.norm(p[a] - p[b]) or 1e-6) for a, b, top, bottom in EYES]
    return float(np.mean(opens)), float(min(opens))


def collect():
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(lm_cfg.FACE_MODEL_PATH)), running_mode=vision.RunningMode.IMAGE,
        num_faces=1, output_face_blendshapes=True))
    cnns = {name: Classifier(Path(p), c.EYE_SIZE, 4) for name, p in CNNS.items()}
    det = cv2.FaceDetectorYN.create(str(c.YUNET_PATH), "", (320, 320), c.SCORE_TH)
    rows = []
    for subset in ("dev", "holdout"):
        evaluate.SUBSET = subset
        for path, labels in evaluate.load().items():
            src = cv2.imread(str(WIDER / path))
            s = 640 / src.shape[1]
            img = cv2.resize(src, (640, round(src.shape[0] * s)), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            det.setInputSize((img.shape[1], img.shape[0]))
            _, d = det.detect(img)
            if d is None:
                continue
            boxes = [tuple(int(v) for v in x[:4]) for x in d]
            gts = [tuple(int(r[k]) * s for k in "xywh") for r in labels]
            for gi, (pi, _) in evaluate.match(gts, boxes).items():
                g = labels[gi]
                if g["visible"] != "1" or g["eyes_open"] not in ("0", "1"):
                    continue
                x, y, bw, bh = boxes[pi]
                mx, my = bw * lm_cfg.CROP_MARGIN, bh * lm_cfg.CROP_MARGIN
                x1, y1 = max(int(x - mx), 0), max(int(y - my), 0)
                x2, y2 = min(int(x + bw + mx), img.shape[1]), min(int(y + bh + my), img.shape[0])
                crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
                res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
                row = {"subset": subset, "image": path, "h": bh, "label": g["eyes_open"], "found": int(bool(res.face_landmarks))}
                if res.face_landmarks:
                    bs = {cat.category_name: cat.score for cat in res.face_blendshapes[0]}
                    row["blendshape"] = round(1 - rules.blink_score(bs), 4)
                    row["geo_mean"], row["geo_min"] = (round(v, 4) for v in eye_open(res.face_landmarks[0], crop.shape[1], crop.shape[0]))
                eye_in = upper(square_crop(img, boxes[pi]), c.EYE_TOP)
                for name, clf in cnns.items():
                    row[f"cnn_{name}"] = round(clf(eye_in), 4)
                rows.append(row)
    keys = ["subset", "image", "h", "label", "found", "blendshape", "geo_mean", "geo_min"] + [f"cnn_{n}" for n in cnns]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    return rows, keys[5:]


def main():
    rows, keys = collect()
    cnn_keys = [k for k in keys if k.startswith("cnn_")]
    print("눈 뜸 AUC (뜸 1 vs 감음 0). 가로 640, 보이는 얼굴. FaceLandmarker 가 찾은 얼굴 기준 (CNN 전체는 마지막 열)")
    for subset in ("dev", "holdout"):
        sub = [r for r in rows if r["subset"] == subset]
        found = [r for r in sub if r["found"]]
        print(f"\n[{subset}] 전체 뜸/감음 {sum(r['label'] == '1' for r in sub)}/{sum(r['label'] == '0' for r in sub)}, "
              f"FaceLandmarker 찾음 {len(found)}/{len(sub)}")
        print(f"  {'구간':<12}{'n':>9}" + "".join(f"{k:>12}" for k in keys) + "".join(f"{k + '(전체)':>18}" for k in cnn_keys))
        for name, lo, hi in [("전체", 0, 10000)] + [(f"{lo}~{hi if hi < 10000 else ''}px", lo, hi) for lo, hi in BINS]:
            g = [r for r in found if lo <= r["h"] < hi]
            ga = [r for r in sub if lo <= r["h"] < hi]
            n = f"{sum(r['label'] == '1' for r in g)}/{sum(r['label'] == '0' for r in g)}"
            vals = [auc([r[k] for r in g if r["label"] == "1"], [r[k] for r in g if r["label"] == "0"]) for k in keys]
            alls = [auc([r[k] for r in ga if r["label"] == "1"], [r[k] for r in ga if r["label"] == "0"]) for k in cnn_keys]
            print(f"  {name:<12}{n:>9}" + "".join(f"{v:>12.3f}" for v in vals) + "".join(f"{v:>18.3f}" for v in alls))
    print(f"\n{OUT}")


if __name__ == "__main__":
    main()
