"""yunet_landmark vs yunet_cnn 단계별 비교: AUC (기준값과 무관한 분리 성능) + 추론 속도
- AUC: compare_models.py 의 faces.csv (dev, holdout) 로 계산. 정답과 짝지어진 얼굴만
  - 웃음 (PREPARE, CHECKING): 정답이 보이는 얼굴, 웃음 라벨 1 vs 0. landmark 는 mouthSmile 평균, cnn 은 smile CNN
  - 눈 뜸 (CHECKING): 정답이 보이는 얼굴, 뜸(1, 선글라스 2) vs 감음(0). landmark 는 1 - eyeBlink 최대, cnn 은 eye CNN
  - 가림 (PREVIEW, CHECKING): 가림 vs 보임. landmark 는 가림 점수가 없음 -> FaceLandmarker 미검출(0/1) 로 계산 (참고), cnn 은 가림 CNN
  - 옆모습 (PREVIEW, CHECKING): 옆모습 vs 보임. 두 모델 모두 YuNet yaw (같은 값)
  - landmark 의 FaceLandmarker 미검출 얼굴: 웃음, 눈 AUC 에서 제외 (수는 따로 적음)
- 속도: 앱의 단계별 요청 그대로 Model.predict 호출 (app.py AGENT_REQUESTS, CHECK_REQUEST)
  - WIDER dev 사진 앞 N_SPEED 장, 가로 640. 처음 WARMUP 장은 버림
  - 사진당 시간 중앙값, 평균, 판정 FPS (1000 / 평균), 얼굴 1명당 시간 (사진당 시간 / 얼굴 수 의 중앙값, 얼굴 있는 사진만)
  - 단계별 세부 시간 (Prediction.timings) 평균
- 이 PC (Mac) 기준. 라즈베리파이는 따로 측정

실행: app/.venv/bin/python evaluation/wider/compare_models_stage.py
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.argv = sys.argv[:1]
CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, RESULTS, WIDER  # noqa: E402

HERE = RESULTS   # 평가 결과 위치 (evaluation/paths.py)
os.environ["EVAL_SUBSET"] = "dev"
sys.path.insert(0, str(CODE))
from evaluate import ROOT, load  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
from photo_app.app import AGENT_REQUESTS, CHECK_REQUEST  # noqa: E402
from photo_app.models import create_model  # noqa: E402
from photo_app.states import State  # noqa: E402

N_SPEED, WARMUP, WIDTH = 200, 5, 640
STAGES = {"PREVIEW": AGENT_REQUESTS[State.PREVIEW], "PREPARE": AGENT_REQUESTS[State.PREPARE], "CHECKING": CHECK_REQUEST}


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    p, n = np.array(pos, float)[:, None], np.array(neg, float)[None, :]
    return float(np.mean(p > n) + 0.5 * np.mean(p == n))


def auc_table():
    print("[AUC] 1 이면 완벽 분리, 0.5 면 무작위")
    print(f"  {'과제 (단계)':<24} {'':<8} {'yunet_landmark':>22} {'yunet_cnn':>16}")
    for subset in ("dev", "holdout"):
        rows = [r for r in csv.DictReader(open(HERE / f"results_models_{subset}" / "faces.csv")) if r.get("kind")]
        vis = [r for r in rows if r["kind"] == "visible"]
        found = [r for r in vis if r["lm_found"] == "1"]
        s1, s0 = [r for r in vis if r["smile_label"] == "1"], [r for r in vis if r["smile_label"] == "0"]
        e1, e0 = [r for r in vis if r["eye_label"] in ("1", "2")], [r for r in vis if r["eye_label"] == "0"]
        f = lambda rs, lab: [r for r in rs if r["lm_found"] == "1"]  # noqa: E731
        occ, side = [r for r in rows if r["kind"] == "occluded"], [r for r in rows if r["kind"] == "side"]
        lines = [
            ("웃음 (PREPARE, CHECKING)",
             auc([float(r["bs_smile"]) for r in f(s1, 1)], [float(r["bs_smile"]) for r in f(s0, 0)]),
             auc([float(r["cnn_smile"]) for r in s1], [float(r["cnn_smile"]) for r in s0]),
             f"웃음 {len(s1)} / 안 웃음 {len(s0)}, landmark 미검출 {len(vis) - len(found)}"),
            ("눈 뜸 (CHECKING)",
             auc([1 - float(r["bs_blink"]) for r in f(e1, 1)], [1 - float(r["bs_blink"]) for r in f(e0, 0)]),
             auc([float(r["cnn_eye"]) for r in e1], [float(r["cnn_eye"]) for r in e0]),
             f"뜸 {len(e1)} / 감음 {len(e0)}"),
            ("가림 (PREVIEW, CHECKING)",
             auc([1 - int(r["lm_found"]) for r in occ], [1 - int(r["lm_found"]) for r in vis]),
             auc([float(r["occ"]) for r in occ], [float(r["occ"]) for r in vis]),
             f"가림 {len(occ)} / 보임 {len(vis)}. landmark 는 미검출 여부 (참고)"),
            ("옆모습 (PREVIEW, CHECKING)",
             auc([float(r["yaw"]) for r in side], [float(r["yaw"]) for r in vis]),
             auc([float(r["yaw"]) for r in side], [float(r["yaw"]) for r in vis]),
             f"옆모습 {len(side)} / 보임 {len(vis)}. 두 모델 모두 YuNet yaw"),
        ]
        for name, a_lm, a_cnn, note in lines:
            print(f"  {name:<24} {subset:<8} {a_lm:>22.3f} {a_cnn:>16.3f}   {note}")


def speed_table():
    images = []
    for path in list(load())[:N_SPEED]:
        src = cv2.imread(str(WIDER / path))
        s = WIDTH / src.shape[1]
        images.append(cv2.resize(src, (WIDTH, round(src.shape[0] * s)), interpolation=cv2.INTER_AREA))
    print(f"\n[추론 속도] WIDER dev 사진 {len(images) - WARMUP}장, 가로 {WIDTH}, 이 PC. Model.predict 전체 (얼굴 검출 ~ 판정)")
    print(f"  {'단계':<9} {'모델':<15} {'사진당 중앙값':>12} {'평균':>8} {'판정 FPS':>9} {'얼굴당':>8}   세부 평균 (ms)")
    for name in ("yunet_landmark", "yunet_cnn"):
        model = create_model(name)
        for stage, req in STAGES.items():
            ms, per_face, parts = [], [], defaultdict(list)
            for i, img in enumerate(images):
                pred = model.predict(img, **req)
                if i < WARMUP:
                    continue
                ms.append(pred.elapsed_ms)
                if pred.faces:
                    per_face.append(pred.elapsed_ms / len(pred.faces))
                for k, v in pred.timings.items():
                    parts[k].append(v)
            detail = ", ".join(f"{k} {np.mean(v):.1f}" for k, v in parts.items())
            print(f"  {stage:<9} {name:<15} {np.median(ms):>10.1f}ms {np.mean(ms):>6.1f}ms {1000 / np.mean(ms):>9.1f} "
                  f"{np.median(per_face):>6.1f}ms   {detail}")


if __name__ == "__main__":
    auc_table()
    speed_table()
