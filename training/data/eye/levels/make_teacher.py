"""eye 눈 뜬 정도 교사 점수 (4단계 로지스틱 회귀)
- 입력: ../preprocessed/{opened,closed}/*.png 의 FaceLandmarker blendshape (눈, 표정 관련 값)
  - 192x192 로 키우고 둘레 32px 검은 여백 (teacher/make_teacher_scores.py 와 같음)
  - 사용 값: FEATURES (좌우 각각). 저장: ./blendshapes_all.csv (한 번 계산하면 재사용)
- 정답: ./levels.csv 의 단계 (1 뜸 / 2 웃는 눈 / 3 감기는 눈 / 4 감은 눈)
- 모델: 다항 로지스틱 회귀 (표준화 후). 단계별 확률 P(단계)
  - 교사 점수 = sum P(단계) x LEVEL_TARGETS[단계] (0~1, 클수록 뜸)
  - 5겹 교차 검증: 각 이미지의 점수는 그 이미지를 학습에 쓰지 않은 모델로 계산 (누수 방지)
  - FaceLandmarker 미검출 이미지는 빈칸 (학습에서는 단계 점수만 사용)
- 출력: ./teacher_levels.csv (folder, file, level, teacher_score, p1, p2, p3, p4)
- 요약: 단계별 교사 점수 중앙값, 이웃 단계 순서 AUC (교차 검증), 기존 교사(1 - 눈 감음)와 비교

실행
- 1단계 blendshape 측정: app/.venv/bin/python training/data/eye/levels/make_teacher.py (mediapipe 필요)
- 2단계 교사 모델: scikit-learn 이 있는 환경의 python 으로 같은 파일 실행 (blendshapes_all.csv 가 있으면 측정 생략)
"""
import csv
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[2]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "eye" / "levels"   # 데이터 위치 (training/paths.py)
EYE = HERE.parent
MODEL = REPO / "app/models/face_landmarker.task"
BLEND = HERE / "blendshapes_all.csv"
OUT = HERE / "teacher_levels.csv"
UPSCALE, PAD = 192, 32
LEVEL_TARGETS = {1: 1.0, 2: 0.7, 3: 0.3, 4: 0.0}
FEATURES = [f"{name}{side}" for name in ("eyeBlink", "eyeSquint", "eyeWide", "eyeLookDown", "eyeLookUp", "cheekSquint",
                                         "mouthSmile", "browDown", "browInnerUp") for side in ("Left", "Right")] + ["browInnerUp"]
FEATURES = list(dict.fromkeys(f for f in FEATURES if f != "browInnerUpLeft" and f != "browInnerUpRight"))


def measure():
    """blendshape 측정 (mediapipe 필요: app/.venv)"""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.IMAGE,
        num_faces=1, output_face_blendshapes=True))
    rows, names = [], None
    for folder in ("opened", "closed"):
        for path in sorted((EYE / "preprocessed" / folder).glob("*.png")):
            img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            big = cv2.copyMakeBorder(cv2.resize(img, (UPSCALE, UPSCALE), interpolation=cv2.INTER_CUBIC),
                                     PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT)
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=big))
            row = {"folder": folder, "file": path.name}
            if res.face_blendshapes:
                b = {c.category_name: round(c.score, 5) for c in res.face_blendshapes[0]}
                names = names or list(b)
                row.update(b)
            rows.append(row)
    with open(BLEND, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["folder", "file"] + names)
        w.writeheader()
        w.writerows(rows)


def auc(pos, neg):
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(np.mean(pos > neg) + 0.5 * np.mean(pos == neg))


def main():
    if not BLEND.exists():
        measure()
        print(f"blendshape 저장: {BLEND}")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("scikit-learn 이 없어 교사 모델은 건너뜀. scikit-learn 이 있는 환경으로 다시 실행")
        return
    blend = {(r["folder"], r["file"]): r for r in csv.DictReader(open(BLEND))}
    levels = list(csv.DictReader(open(HERE / "levels.csv")))
    keys = [(r["folder"], r["file"]) for r in levels]
    y = np.array([int(r["level"]) for r in levels])
    has = np.array([blend[k].get(FEATURES[0], "") != "" for k in keys])
    X = np.array([[float(blend[k][f]) for f in FEATURES] if h else [0.0] * len(FEATURES) for k, h in zip(keys, has)])
    targets = np.array([LEVEL_TARGETS[c] for c in (1, 2, 3, 4)])

    proba = np.full((len(y), 4), np.nan)
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    idx = np.flatnonzero(has)
    for tr, te in folds.split(idx, y[idx]):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
        clf.fit(X[idx[tr]], y[idx[tr]])
        proba[idx[te]] = clf.predict_proba(X[idx[te]])
    score = proba @ targets

    with open(OUT, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["folder", "file", "level", "teacher_score", "p1", "p2", "p3", "p4"])
        for (f, n), lv, s, p in zip(keys, y, score, proba):
            w.writerow([f, n, lv] + (["", "", "", "", ""] if np.isnan(s) else [round(s, 4)] + [round(v, 4) for v in p]))

    old = np.array([1 - max(float(blend[k]["eyeBlinkLeft"]), float(blend[k]["eyeBlinkRight"])) if h else np.nan
                    for k, h in zip(keys, has)])
    print(f"저장: {OUT} (blendshape 검출 {has.sum()} / {len(y)})")
    print(f"사용 값 {len(FEATURES)}개: {', '.join(FEATURES)}")
    print(f"  {'단계':<10}{'n':>6}{'새 교사 중앙값':>14}{'기존 교사 중앙값':>16}")
    names = {1: "1 뜸", 2: "2 웃는 눈", 3: "3 감기는 눈", 4: "4 감은 눈"}
    for lv in (1, 2, 3, 4):
        m = has & (y == lv)
        print(f"  {names[lv]:<10}{m.sum():>6}{np.median(score[m]):>14.3f}{np.median(old[m]):>16.3f}")
    print("이웃 단계 순서 AUC (교차 검증. 새 교사 / 기존 교사)")
    for lv in (1, 2, 3):
        a, b = has & (y == lv), has & (y == lv + 1)
        print(f"  {lv}>{lv + 1}: {auc(score[a], score[b]):.3f} / {auc(old[a], old[b]):.3f}")
    print(f"  뜸(1,2) vs 감음(3,4): {auc(score[has & (y <= 2)], score[has & (y >= 3)]):.3f} / "
          f"{auc(old[has & (y <= 2)], old[has & (y >= 3)]):.3f}")


if __name__ == "__main__":
    main()
