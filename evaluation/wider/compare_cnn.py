"""CNN 분류기(smile, eye tflite)와 FaceLandmarker blendshape 비교
- 대상: evaluate.py 와 같음 (라벨 완료, 부적절 제외). 보이는 얼굴(visible=1)만
- 얼굴: 가로 WIDTH 로 줄인 사진에서 YuNet(score >= 0.7) 검출, 정답과 IoU >= 0.3 으로 짝
- CNN 입력 (예시 이미지, 모델 입력 이름, 라벨 데이터로 확인)
  - YuNet 박스를 긴 변 기준 정사각형으로 자름 (여백 없음). 화면 밖은 가장자리 복제
  - smile: 얼굴 RGB 128x128, 0~255. 출력 방향은 SMILE_OUTPUT (not_smile: 클수록 안 웃음 (v0.0.1), smile: 클수록 웃음)
  - eye: 얼굴 위쪽 EYE_TOP 비율 RGB 160x160, 0~255. 출력이 클수록 눈 뜸
    - EYE_TOP: eye 모델 옆 inference_config 에서 읽음 (eye_top(), 없으면 0.6)
- blendshape: yunet_landmark 와 같음 (여백 0.5 로 잘라 FaceLandmarker)
- 얼굴 높이 구간별 AUC (기준값과 무관한 분리 성능), 기준 0.5 에서의 precision, recall
- 사진 높이 기준: WIDTHS 로 줄인 결과 합산 (같은 얼굴을 여러 크기로)
- 평가 대상 사진: 환경변수 EVAL_SUBSET=dev|holdout (evaluate.py 참고)
- 환경변수 SMILE_TTA_FLIP=1 이면 smile 점수를 원본과 좌우 뒤집은 얼굴의 평균으로
- 출력: results_cnn[_TAG][_SUBSET]/faces.csv (얼굴별 점수), 터미널 요약

실행: [EVAL_SUBSET=dev] app/.venv/bin/python evaluation/wider/compare_cnn.py [SMILE_TFLITE] [EYE_TFLITE] [SMILE_OUTPUT] [TAG]
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]   # evaluate.py 는 import 할 때 sys.argv 를 읽으므로 비워 둠

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, RESULTS, WIDER  # noqa: E402

HERE = RESULTS   # 평가 결과 위치 (evaluation/paths.py)
sys.path.insert(0, str(CODE))
from evaluate import ROOT, SUBSET, load, match  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
from photo_app.models.landmark import rules  # noqa: E402
from photo_app.models.yunet_landmark import YunetLandmarkModel  # noqa: E402

SMILE_PATH = Path(ARGS[0]) if len(ARGS) > 0 else ROOT / "app/models/v0.0.1/smile_fp32.tflite"
EYE_PATH = Path(ARGS[1]) if len(ARGS) > 1 else ROOT / "app/models/v0.0.1/eye_float32.tflite"
SMILE_OUTPUT = ARGS[2] if len(ARGS) > 2 else "not_smile"   # smile 모델 출력 의미: not_smile | smile
SMILE_TTA_FLIP = os.environ.get("SMILE_TTA_FLIP") == "1"
TAG = ARGS[3] if len(ARGS) > 3 else ""                      # 결과 폴더 이름표 (비우면 results_cnn)
WIDTHS = (640, 1024)
DEFAULT_EYE_TOP = 0.6
BINS = ((0, 40), (40, 60), (60, 100), (100, 10**9))


def eye_top(eye_path):
    """eye 모델 입력의 위쪽 비율
    - 모델 파일 옆 eye_inference_config.json (앱 폴더) 또는 inference_config.json (학습 결과 폴더)
    - input.eye_top 값, 없으면 input.crop 문구의 "위쪽 N"
    - 설정 파일이 없으면 DEFAULT_EYE_TOP (v0.0.1)
    """
    eye_path = Path(eye_path)
    for name in ("eye_inference_config.json", "inference_config.json"):
        cfg_path = eye_path.parent / name
        if cfg_path.exists():
            inp = json.load(open(cfg_path))["input"]
            if "eye_top" in inp:
                return float(inp["eye_top"])
            m = re.search(r"위쪽 ([0-9.]+)", inp.get("crop", ""))
            if m:
                return float(m.group(1))
    return DEFAULT_EYE_TOP


EYE_TOP = eye_top(EYE_PATH)
OUT = HERE / ("results_cnn" + (f"_{TAG}" if TAG else "") + (f"_{SUBSET}" if SUBSET else ""))


class Classifier:
    def __init__(self, path, size):
        self.it = Interpreter(model_path=str(path))
        self.it.allocate_tensors()
        self.size = size
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]

    def __call__(self, rgb):
        x = cv2.resize(rgb, (self.size, self.size)).astype(np.float32)[None]
        self.it.set_tensor(self.inp, x)
        self.it.invoke()
        return float(self.it.get_tensor(self.out)[0, 0])


def smile_out(model, face):
    """smile 모델 출력. SMILE_TTA_FLIP 이면 원본과 좌우 뒤집은 얼굴의 평균"""
    if not SMILE_TTA_FLIP:
        return model(face)
    return (model(face) + model(np.ascontiguousarray(face[:, ::-1]))) / 2


def smile_score(out):
    """smile 모델 출력 -> 웃음 점수 (클수록 웃음)"""
    if SMILE_OUTPUT not in ("smile", "not_smile"):
        raise ValueError(f"SMILE_OUTPUT 은 smile 또는 not_smile: {SMILE_OUTPUT}")
    return out if SMILE_OUTPUT == "smile" else 1 - out


def square_crop(img, box):
    """박스 중심 기준 긴 변 정사각형. 화면 밖은 가장자리 복제"""
    x, y, w, h = box
    c = max(w, h)
    x1, y1 = int(x + w / 2 - c / 2), int(y + h / 2 - c / 2)
    H, W = img.shape[:2]
    top, left = max(0, -y1), max(0, -x1)
    pad = cv2.copyMakeBorder(img, top, max(0, y1 + c - H), left, max(0, x1 + c - W), cv2.BORDER_REPLICATE)
    return pad[y1 + top:y1 + top + c, x1 + left:x1 + left + c]


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    p, n = np.array(pos)[:, None], np.array(neg)[None, :]
    return float(np.mean(p > n) + 0.5 * np.mean(p == n))


def main():
    smile_cnn, eye_cnn = Classifier(SMILE_PATH, 128), Classifier(EYE_PATH, 160)
    lm = YunetLandmarkModel(score_th=0.7)
    rows = []
    for path, labels in load().items():
        src = cv2.imread(str(WIDER / path))
        for width in WIDTHS:
            s = width / src.shape[1]
            img = cv2.resize(src, (width, round(src.shape[0] * s)), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = [box for box, _ in lm._detect_faces(img)]   # (bbox, yaw) 중 bbox
            gts = [tuple(int(r[k]) * s for k in ("x", "y", "w", "h")) for r in labels]
            for gi, (pi, _) in match(gts, boxes).items():
                r = labels[gi]
                if r["visible"] != "1":
                    continue
                face = square_crop(rgb, boxes[pi])
                bs = lm._blendshapes(rgb, boxes[pi])
                rows.append({
                    "image": path, "face_id": r["face_id"], "width": width, "h_px": boxes[pi][3],
                    "smile": r["smile"], "eyes_open": r["eyes_open"],
                    "cnn_smile": round(smile_score(smile_out(smile_cnn, face)), 4),           # 클수록 웃음
                    "cnn_eye": round(eye_cnn(face[:max(1, int(face.shape[0] * EYE_TOP))]), 4),   # 클수록 뜸
                    "bs_smile": "" if bs is None else round(rules.smile_score(bs), 4),
                    "bs_eye": "" if bs is None else round(1 - rules.blink_score(bs), 4),       # 클수록 뜸으로 뒤집음
                })
    OUT.mkdir(exist_ok=True)
    with open(OUT / "faces.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary(rows)


def summary(rows):
    print(f"보이는 얼굴, 사진 가로 {WIDTHS} 합산, YuNet score >= 0.7, eye 입력 위쪽 {EYE_TOP}")
    print("AUC: 1 이면 완벽 분리, 0.5 면 무작위. blendshape 이 None(FaceLandmarker 미검출)인 얼굴은 두 방식 모두 제외")
    for task, label, pos in (("smile", "smile", ("1",)), ("eye", "eyes_open", ("1",))):
        print(f"\n[{task}] 라벨 {label}: 양성 {pos}, 음성 0")
        print(f"  {'높이(px)':>10} {'n(양성/음성)':>14} {'CNN AUC':>8} {'blendshape AUC':>15}")
        groups = defaultdict(list)
        for r in rows:
            if r[label] not in ("0", "1") or r[f"bs_{task}"] == "":
                continue
            for lo, hi in BINS:
                if lo <= r["h_px"] < hi:
                    groups[(lo, hi)].append(r)
            groups["전체"].append(r)
        for k in [*BINS, "전체"]:
            rs = groups[k]
            p = [r for r in rs if r[label] in pos]
            n = [r for r in rs if r[label] not in pos]
            name = k if k == "전체" else f"{k[0]}~{k[1] if k[1] < 10**9 else ''}"
            print(f"  {name:>10} {f'{len(p)}/{len(n)}':>14} "
                  f"{auc([r[f'cnn_{task}'] for r in p], [r[f'cnn_{task}'] for r in n]):>8.3f} "
                  f"{auc([r[f'bs_{task}'] for r in p], [r[f'bs_{task}'] for r in n]):>15.3f}")
        rs = groups["전체"]
        for name, key, th in (("CNN", f"cnn_{task}", 0.5), ("blendshape", f"bs_{task}", 0.5)):
            tp = sum(1 for r in rs if r[label] in pos and r[key] >= th)
            fp = sum(1 for r in rs if r[label] not in pos and r[key] >= th)
            fn = sum(1 for r in rs if r[label] in pos and r[key] < th)
            print(f"  {name} 기준 0.5: 양성 precision {tp / max(tp + fp, 1):.1%}, recall {tp / max(tp + fn, 1):.1%}")


if __name__ == "__main__":
    main()
