"""yunet_cnn CNN(smile, eye, occlusion) 양자화 형식별 비교: fp32 / dynamic / int8
- 모델: yunet_cnn config 의 SMILE_PATH 와 같은 폴더의 {smile,eye,occlusion}_{fp32,dynamic,int8}.tflite
- 대상: WIDER dev + holdout (evaluate.load), 가로 640, YuNet (SCORE_TH) 검출과 정답 IoU 매칭한 얼굴
  - 입력은 앱과 같음 (yunet_cnn.model.square_crop, upper, lower)
- AUC (기준값과 무관)
  - 웃음: 정답이 보이는 얼굴, 웃음 1 vs 0 / 눈 뜸: 뜸(1, 선글라스 2) vs 감음(0) / 가림: 가림 vs 보임
- 속도: CNN 1회 호출 시간 (얼굴 1명, 리사이즈 포함). 처음 WARMUP 회는 버림. 스레드 NUM_THREADS
  - int8: 입출력 int8 (양자화 scale, zero point 로 변환). XNNPACK 준비에 실패하면 XNNPACK 을 끄고 실행 (노트북 검증과 같음)
- 형식별로 fp32 와의 점수 차이 (최대, 평균) 도 적음
- 출력: results_quant_{TAG}/scores.csv (얼굴별 점수), 터미널 요약. TAG 는 실행 환경 이름 (mac, colima 등)

실행: app/.venv/bin/python evaluation/wider/bench_quant.py mac
     컨테이너: python evaluation/wider/bench_quant.py colima (저장소 루트에서)
"""
import csv
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

TAG = sys.argv[1] if len(sys.argv) > 1 else "mac"
sys.argv = sys.argv[:1]
CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, RESULTS, WIDER  # noqa: E402

HERE = RESULTS   # 평가 결과 위치 (evaluation/paths.py)
sys.path.insert(0, str(CODE))
from evaluate import ROOT, load, match  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
from photo_app.models.yunet_cnn import config as cfg  # noqa: E402
from photo_app.models.yunet_cnn.model import lower, square_crop, upper  # noqa: E402

WIDTH, WARMUP = 640, 20
KINDS = ("fp32", "dynamic", "int8")
TASKS = {"smile": cfg.SMILE_SIZE, "eye": cfg.EYE_SIZE, "occlusion": cfg.OCC_SIZE}
OUT = HERE / f"results_quant_{TAG}"


class QClassifier:
    """LiteRT 분류 CNN (fp32, dynamic, int8 모두). 입력 RGB 0~255, 출력 점수 0~1"""

    def __init__(self, path, size):
        self.size = size
        self.xnnpack = True
        try:
            self._open(path, OpResolverType.AUTO)
            self(np.zeros((size, size, 3), np.uint8))
        except RuntimeError:   # MobileNetV3 int8 은 XNNPACK 준비 실패 (allocate_tensors 또는 첫 실행)
            self.xnnpack = False
            self._open(path, OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)

    def _open(self, path, resolver):
        self.it = Interpreter(model_path=str(path), num_threads=cfg.NUM_THREADS, experimental_op_resolver_type=resolver)
        self.it.allocate_tensors()
        self.inp, self.out = self.it.get_input_details()[0], self.it.get_output_details()[0]

    def __call__(self, rgb):
        x = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype(np.float32)[None]
        if np.issubdtype(self.inp["dtype"], np.integer):
            scale, zero = self.inp["quantization"]
            x = np.clip(np.rint(x / scale + zero), -128, 127).astype(self.inp["dtype"])
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"]).astype(np.float32)
        if np.issubdtype(self.out["dtype"], np.integer):
            scale, zero = self.out["quantization"]
            y = (y - zero) * scale
        return float(y.ravel()[0])


def kind(r):
    if r["visible"] == "1":
        return "visible"
    return "side" if r["hidden_reason"] == "side" else "occluded"


def run():
    det = cv2.FaceDetectorYN.create(str(cfg.YUNET_PATH), "", (320, 320), cfg.SCORE_TH)
    models = {(t, k): QClassifier(cfg.SMILE_PATH.parent / f"{t}_{k}.tflite", size)
              for t, size in TASKS.items() for k in KINDS}
    for (t, k), m in models.items():
        print(f"{t}_{k}: XNNPACK {'사용' if m.xnnpack else '끔'}")
    rows, times = [], {key: [] for key in models}
    images = {}
    for subset in ("dev", "holdout"):
        os.environ["EVAL_SUBSET"] = subset
        import evaluate
        evaluate.SUBSET = subset
        images.update({p: (subset, lab) for p, lab in evaluate.load().items()})
    for n, (path, (subset, labels)) in enumerate(images.items(), 1):
        src = cv2.imread(str(WIDER / path))
        s = WIDTH / src.shape[1]
        img = cv2.resize(src, (WIDTH, round(src.shape[0] * s)), interpolation=cv2.INTER_AREA)
        det.setInputSize((img.shape[1], img.shape[0]))
        _, dets = det.detect(img)
        if dets is None:
            continue
        boxes = [tuple(int(v) for v in d[:4]) for d in dets]
        gts = [tuple(int(r[k]) * s for k in ("x", "y", "w", "h")) for r in labels]
        for gi, (pi, _) in match(gts, boxes).items():
            g = labels[gi]
            if g["visible"] not in ("0", "1"):
                continue
            crop = square_crop(img, boxes[pi])
            inputs = {"smile": lower(crop, cfg.SMILE_BOTTOM), "eye": upper(crop, cfg.EYE_TOP), "occlusion": crop}
            row = {"subset": subset, "image": path, "face_id": g["face_id"], "kind": kind(g),
                   "smile_label": g["smile"], "eye_label": g["eyes_open"]}
            for (t, k), m in models.items():
                t0 = time.perf_counter()
                row[f"{t}_{k}"] = round(m(inputs[t]), 5)
                times[(t, k)].append((time.perf_counter() - t0) * 1000)
            rows.append(row)
        print(f"\r{n}/{len(images)}", end="", flush=True)
    print()
    OUT.mkdir(exist_ok=True)
    with open(OUT / "scores.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "times.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["model", "kind", "xnnpack", "mean_ms", "median_ms", "p90_ms", "calls"])
        for (t, k), v in times.items():
            v = np.array(v[WARMUP:])
            w.writerow([t, k, int(models[(t, k)].xnnpack), round(v.mean(), 3), round(np.median(v), 3),
                        round(np.percentile(v, 90), 3), len(v)])


def auc(pos, neg):
    p, n = np.array(pos)[:, None], np.array(neg)[None, :]
    return float(np.mean(p > n) + 0.5 * np.mean(p == n))


def summary():
    rows = list(csv.DictReader(open(OUT / "scores.csv")))
    times = {(r["model"], r["kind"]): r for r in csv.DictReader(open(OUT / "times.csv"))}
    print(f"환경: {TAG} ({platform.system()} {platform.machine()}, CPU {os.cpu_count()}), 모델 {cfg.MODEL_VERSION}, "
          f"스레드 {cfg.NUM_THREADS}, 얼굴 {len(rows)}")
    groups = {
        "smile": lambda sub: ([r for r in sub if r["kind"] == "visible" and r["smile_label"] == "1"],
                              [r for r in sub if r["kind"] == "visible" and r["smile_label"] == "0"]),
        "eye": lambda sub: ([r for r in sub if r["kind"] == "visible" and r["eye_label"] in ("1", "2")],
                            [r for r in sub if r["kind"] == "visible" and r["eye_label"] == "0"]),
        "occlusion": lambda sub: ([r for r in sub if r["kind"] == "occluded"], [r for r in sub if r["kind"] == "visible"]),
    }
    print(f"\n{'모델':<10} {'형식':<8} {'AUC dev':>8} {'holdout':>8} {'fp32 와 점수 차 최대/평균':>24} "
          f"{'1회 평균':>9} {'중앙값':>8} {'p90':>8}  XNNPACK")
    for t in TASKS:
        for k in KINDS:
            aucs = []
            for subset in ("dev", "holdout"):
                pos, neg = groups[t]([r for r in rows if r["subset"] == subset])
                aucs.append(auc([float(r[f"{t}_{k}"]) for r in pos], [float(r[f"{t}_{k}"]) for r in neg]))
            diff = np.abs(np.array([float(r[f"{t}_{k}"]) for r in rows]) - np.array([float(r[f"{t}_fp32"]) for r in rows]))
            tm = times[(t, k)]
            print(f"{t:<10} {k:<8} {aucs[0]:>8.3f} {aucs[1]:>8.3f} {diff.max():>14.3f} / {diff.mean():.4f} "
                  f"{float(tm['mean_ms']):>7.2f}ms {float(tm['median_ms']):>6.2f}ms {float(tm['p90_ms']):>6.2f}ms  "
                  f"{'o' if tm['xnnpack'] == '1' else 'x'}")


if __name__ == "__main__":
    if not (OUT / "scores.csv").exists():
        run()
    summary()
