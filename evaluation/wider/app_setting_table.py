"""앱 설정 그대로의 사진 단위 표 (photo_compare 결과 faces.csv 재사용)
- YuNet 0.7, yaw >= 0.5 는 보이지 않음, CNN smile >= SMILE_TH, eye >= EYE_TH
- 판단 불가(-1) 사진은 미충족으로 / 제외 두 가지
실행: python3 evaluation/wider/app_setting_table.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.argv = sys.argv[:1]
CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, RESULTS, WIDER  # noqa: E402

HERE = RESULTS   # 평가 결과 위치 (evaluation/paths.py)
sys.path.insert(0, str(CODE))
from evaluate import has_unknown, load  # noqa: E402

SMILE_TH, EYE_TH, DET_TH, YAW_TH = 0.5, 0.8, 0.7, 0.5
BLINK_TH, BS_SMILE_TH = 0.5, 0.5


def ok(r, method):
    if float(r["yaw"]) >= YAW_TH:
        return False
    if method == "blendshape":
        return r["bs_smile"] != "" and float(r["bs_smile"]) >= BS_SMILE_TH and float(r["bs_eye"]) > 1 - BLINK_TH
    return float(r["cnn_smile"]) >= SMILE_TH and float(r["cnn_eye"]) >= EYE_TH


def table(subset, tag, method):
    by = defaultdict(list)
    for r in csv.DictReader(open(HERE / f"results_photo_{tag}_{subset}" / "faces.csv")):
        by[r["image"]].append(r)
    import os
    os.environ["EVAL_SUBSET"] = subset
    out = []
    for name, photos in (("-1 미충족", by), ("-1 제외", {k: v for k, v in by.items() if k not in UNK[subset]})):
        tp = fp = fn = 0
        for rs in photos.values():
            faces = [r for r in rs if r["h_px"] != "" and float(r["yunet_score"]) >= DET_TH]
            pred = bool(faces) and all(ok(r, method) for r in faces)
            lab = rs[0]["photo_label_ok"] == "1"
            tp += pred and lab
            fp += pred and not lab
            fn += (not pred) and lab
        out.append(f"{tp}/{tp + fp} ({tp / (tp + fp):.0%})" if tp + fp else "0/0")
    print(f"{subset:<8} {method:<11} {tag:<8} {len(by):>4}장 정답충족 {tp + fn:>3}  precision {out[0]:<14} / {out[1]:<14} recall {tp}/{tp + fn} ({tp / (tp + fn):.0%})")


UNK = {}
if __name__ == "__main__":
    import importlib
    import os
    import evaluate
    for s in ("dev", "holdout"):
        os.environ["EVAL_SUBSET"] = s
        importlib.reload(evaluate)
        UNK[s] = {img for img, faces in evaluate.load().items() if evaluate.has_unknown(faces)}
        print(f"{s}: -1 사진 {len(UNK[s])}")
        table(s, "v0.0.2", "blendshape")
        table(s, "v0.0.2", "cnn")
        table(s, "eye1209", "cnn")
        table(s, "eye1307", "cnn")
