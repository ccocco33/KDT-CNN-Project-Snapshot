"""기하 지표 가중 평균 교사 점수
- 입력: ./geometry_metrics.csv (geometry_metrics.py)
- 지표마다 학습 데이터 안의 백분위(0~1)로 바꾼 뒤 WEIGHTS 로 가중 평균. 방향은 클수록 눈을 뜬 쪽 (DIRECTION)
  - 백분위: 척도를 맞추는 용도. 손으로 고른 상수 없음
- 출력: ./teacher_geometry.csv (folder, file, level, teacher_score). 지표가 없는 이미지는 빈칸
- 터미널: 단계별 중앙값, 이웃 단계 AUC. 인자로 앱 촬영 사진을 주면 그 사진의 점수도

실행: app/.venv/bin/python training/data/eye/levels/geometry_teacher.py [사진 ...]
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry_metrics as gm  # noqa: E402

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[2]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "eye" / "levels"   # 데이터 위치 (training/paths.py)
OUT = HERE / "teacher_geometry.csv"
WEIGHTS = {"upper_arch": 0.4, "eye_open": 0.4, "mouth_lift": 0.1, "pitch": 0.1}   # 눈 모양, 눈 열림, 입, 아래봄
DIRECTION = {"upper_arch": 1, "eye_open": 1, "mouth_lift": 1, "pitch": -1}       # pitch 는 숙일수록 낮게
LABELS = {"upper_arch": "눈 모양", "eye_open": "눈 열림", "mouth_lift": "입", "pitch": "아래봄"}


class Teacher:
    def __init__(self, rows):
        self.ref = {k: np.sort(np.array([float(r[k]) for r in rows]) * DIRECTION[k]) for k in WEIGHTS}

    def percentiles(self, m):
        return {k: float(np.searchsorted(self.ref[k], m[k] * DIRECTION[k], side="right") / len(self.ref[k])) for k in WEIGHTS}

    def __call__(self, m):
        p = self.percentiles(m)
        return sum(WEIGHTS[k] * p[k] for k in WEIGHTS) / sum(WEIGHTS.values())


def photo_metrics(paths):
    """앱 촬영 사진 -> [(경로, 지표)]. 얼굴은 YuNet 박스 중심, 박스 긴 변 2배 정사각형으로 잘라 FaceLandmarker"""
    import cv2
    sys.path.insert(0, str(gm.REPO / "app"))
    from photo_app.models import create_model
    det, lm = create_model("yunet_landmark"), gm.make_landmarker()
    out = []
    for p in paths:
        img = cv2.imread(p)
        x, y, w, h = det._detect_faces(img)[0][0]
        c = int(max(w, h) * 2)
        cx, cy = x + w // 2, y + h // 2
        pad = cv2.copyMakeBorder(img, c, c, c, c, cv2.BORDER_CONSTANT)
        crop = cv2.cvtColor(pad[cy + c - c // 2:cy + c + c // 2, cx + c - c // 2:cx + c + c // 2], cv2.COLOR_BGR2RGB)
        out.append((p, gm.measure_rgb(lm, crop)))
    return out


def main():
    all_rows = list(csv.DictReader(open(gm.OUT)))
    rows = [r for r in all_rows if r["eye_open"] != ""]
    teacher = Teacher(rows)
    lv = np.array([int(r["level"]) for r in rows])
    score = np.array([teacher({k: float(r[k]) for k in WEIGHTS}) for r in rows])
    by_key = {(r["folder"], r["file"]): s for r, s in zip(rows, score)}
    with open(OUT, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["folder", "file", "level", "teacher_score"])
        for r in all_rows:
            s = by_key.get((r["folder"], r["file"]))
            w.writerow([r["folder"], r["file"], r["level"], "" if s is None else round(s, 4)])
    print(f"가중치: " + ", ".join(f"{LABELS[k]} {v}" for k, v in WEIGHTS.items()) + f"  -> {OUT}")
    print("단계별 중앙값: " + ", ".join(f"{gm.NAMES[l]} {np.median(score[lv == l]):.2f}" for l in (1, 2, 3, 4)))
    print("이웃 단계 AUC: " + " / ".join(f"{a}>{a + 1} {gm.auc(score[lv == a], score[lv == a + 1]):.2f}" for a in (1, 2, 3))
          + f", 뜸(1,2) vs 감음(3,4) {gm.auc(score[lv <= 2], score[lv >= 3]):.3f}")
    if len(sys.argv) > 1:
        print("촬영 사진:")
        for p, m in photo_metrics(sys.argv[1:]):
            ps = teacher.percentiles(m)
            print(f"  {Path(p).parent.name[:15]}/{Path(p).name}: {teacher(m):.2f}  ("
                  + ", ".join(f"{LABELS[k]} {v:.2f}" for k, v in ps.items()) + ")")


if __name__ == "__main__":
    main()
