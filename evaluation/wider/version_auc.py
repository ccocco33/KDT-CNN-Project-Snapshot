"""CNN 버전별 smile, eye AUC 를 같은 조건으로 (발표, 진단용)
- 대상: WIDER dev, holdout (evaluate.load). 가로 640, YuNet (yunet_cnn SCORE_TH) 검출, 정답과 짝. 보이는 얼굴(visible=1)만
- 모델 (파일이 바뀐 버전만. 같은 파일을 쓴 버전은 묶음)
  - smile: v0.0.1 (출력 1 = 안 웃음 -> 뒤집음), v0.0.2 (~v0.0.5), v0.0.6 (얼굴 아래쪽 0.45)
  - eye: v0.0.1, v0.0.2, v0.0.3 (~v0.0.6). 모두 얼굴 위쪽 0.6 을 160 으로
  - 베이스라인 blendshape (yunet_landmark 와 같은 자르기): 웃음 mouthSmile 평균, 눈 뜸 1 - eyeBlink 최대
  - FaceLandmarker 미검출 얼굴은 blendshape 만 빠짐 (CNN 은 모든 얼굴)
- 자르기: 앱과 같음 (yunet_cnn.model.square_crop, upper, lower)
- 라벨: smile 1 웃음 / 0 안 웃음, eyes_open 1 뜸 / 0 감음 (선글라스 2 는 AUC 에서 뺌)
- 구간: 얼굴 높이 (px), 흑백 여부 (얼굴 채도 평균 < GRAY_SAT)
- 출력: version_auc.csv (얼굴별 점수), version_auc.log (표)

실행: app/.venv/bin/python evaluation/wider/version_auc.py
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.argv = sys.argv[:1]
CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, RESULTS, WIDER  # noqa: E402

HERE = RESULTS   # 평가 결과 위치 (evaluation/paths.py)
sys.path.insert(0, str(CODE))
import evaluate  # noqa: E402

ROOT = evaluate.ROOT
sys.path.insert(0, str(ROOT / "app"))
from photo_app.models.landmark import rules  # noqa: E402
from photo_app.models.yunet_cnn import config as c  # noqa: E402
from photo_app.models.yunet_cnn.model import Classifier, lower, square_crop, upper  # noqa: E402
from photo_app.models.yunet_landmark import YunetLandmarkModel  # noqa: E402

M = ROOT / "app" / "models"
SMILE = {"v0.0.1": (M / "v0.0.1/smile_fp32.tflite", 1.0, True),   # (파일, 아래쪽 비율, 출력 뒤집기)
         "v0.0.2": (M / "v0.0.2/smile_fp32.tflite", 1.0, False),
         "v0.0.6": (M / "v0.0.6/smile_fp32.tflite", 0.45, False)}
EYE = {"v0.0.1": M / "v0.0.1/eye_float32.tflite",
       "v0.0.2": M / "v0.0.2/eye_fp32.tflite",
       "v0.0.3": M / "v0.0.3/eye_fp32.tflite"}
GRAY_SAT = 25
BINS = ((0, 40), (40, 60), (60, 100), (100, 10000))
OUT = HERE / "version_auc.csv"


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def collect():
    lm = YunetLandmarkModel(score_th=c.SCORE_TH)
    smile = {k: (Classifier(p, c.SMILE_SIZE, 4), b, inv) for k, (p, b, inv) in SMILE.items()}
    eye = {k: Classifier(p, c.EYE_SIZE, 4) for k, p in EYE.items()}
    rows = []
    for subset in ("dev", "holdout"):
        evaluate.SUBSET = subset
        for path, labels in evaluate.load().items():
            src = cv2.imread(str(WIDER / path))
            s = 640 / src.shape[1]
            img = cv2.resize(src, (640, round(src.shape[0] * s)), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lm._detector.setInputSize((img.shape[1], img.shape[0]))
            _, d = lm._detector.detect(img)
            if d is None:
                continue
            boxes = [tuple(int(v) for v in x[:4]) for x in d]
            gts = [tuple(int(r[k]) * s for k in "xywh") for r in labels]
            for gi, (pi, _) in evaluate.match(gts, boxes).items():
                g = labels[gi]
                if g["visible"] != "1":
                    continue
                crop = square_crop(img, boxes[pi])
                sat = float(cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)[..., 1].mean())
                bs = lm._blendshapes(rgb, boxes[pi])
                row = {"subset": subset, "image": path, "h": boxes[pi][3], "gray": int(sat < GRAY_SAT),
                       "smile_label": g["smile"], "eye_label": g["eyes_open"],
                       "bs_smile": "" if bs is None else round(rules.smile_score(bs), 4),
                       "bs_eye": "" if bs is None else round(1 - rules.blink_score(bs), 4)}
                for k, (clf, b, inv) in smile.items():
                    v = clf(lower(crop, b) if b < 1 else crop)
                    row[f"smile_{k}"] = round(1 - v if inv else v, 4)
                for k, clf in eye.items():
                    row[f"eye_{k}"] = round(clf(upper(crop, c.EYE_TOP)), 4)
                rows.append(row)
    with open(OUT, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def table(rows, task, keys, label):
    pos_l, neg_l = "1", "0"
    print(f"\n[{task}] AUC (라벨 {label}: 1 vs 0). n = 양성/음성. blendshape 는 FaceLandmarker 검출 얼굴만")
    head = f"  {'구간':<14}{'n':>10}" + "".join(f"{k:>10}" for k in keys)
    for subset in ("dev", "holdout"):
        print(f" {subset}")
        print(head)
        sub = [r for r in rows if r["subset"] == subset and r[label] in (pos_l, neg_l)]
        groups = [("전체", sub)] + [(f"{lo}~{hi if hi < 10000 else ''}px", [r for r in sub if lo <= int(r["h"]) < hi])
                                     for lo, hi in BINS] + [("컬러", [r for r in sub if r["gray"] == 0]),
                                                            ("흑백", [r for r in sub if r["gray"] == 1])]
        for name, g in groups:
            n = f"{sum(r[label] == pos_l for r in g)}/{sum(r[label] == neg_l for r in g)}"
            vals = []
            for k in keys:
                gg = [r for r in g if r[k] != ""]
                vals.append(auc([float(r[k]) for r in gg if r[label] == pos_l], [float(r[k]) for r in gg if r[label] == neg_l]))
            print(f"  {name:<14}{n:>10}" + "".join(f"{v:>10.3f}" for v in vals))


def main():
    rows = collect()
    print("같은 조건: WIDER 가로 640, YuNet >= 0.7, 보이는 얼굴, 앱과 같은 자르기")
    table(rows, "smile", ["bs_smile"] + [f"smile_{k}" for k in SMILE], "smile_label")
    table(rows, "eye", ["bs_eye"] + [f"eye_{k}" for k in EYE], "eye_label")


if __name__ == "__main__":
    main()
