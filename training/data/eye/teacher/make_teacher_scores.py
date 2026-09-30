"""eye 지식 증류용 교사 점수 계산
- 교사: MediaPipe FaceLandmarker blendshape (앱 yunet_landmark 모델과 같은 파일)
- 입력: ../preprocessed/{opened,closed}/*.png (64x64 전처리 얼굴)
  - 192x192 로 키우고 둘레 32px 검은 여백 (작은 얼굴도 FaceLandmarker 가 찾도록)
- 교사 점수 teacher_score: 1 - max(eyeBlinkLeft, eyeBlinkRight) (0~1, 클수록 뜸. 앱 룰의 눈 뜸 판정과 같은 값. 한쪽만 감아도 낮음)
  - FaceLandmarker 미검출이면 빈칸 (학습에서는 정답 라벨만 사용)
- 웃음 확률 보정은 하지 않음 (학습 노트북에서 train 분할만으로 보정. val, test 누수 방지)
- 출력: ./teacher_scores.csv (folder, file, label, teacher_score, eye_blink_left, eye_blink_right)
  - folder, file 은 preprocessed.zip 안의 폴더 이름, 파일 이름과 같음
  - label: 1 = 뜸 (opened), 0 = 감음 (closed)
- 요약: 폴더별 미검출 수, 라벨별 점수 중앙값, 교사 AUC

실행: app/.venv/bin/python training/data/eye/teacher/make_teacher_scores.py
"""
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[2]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "eye" / "teacher"   # 데이터 위치 (training/paths.py)
DATA = HERE.parent / "preprocessed"
MODEL = REPO / "app/models/face_landmarker.task"
OUT = HERE / "teacher_scores.csv"

FOLDERS = {"opened": 1, "closed": 0}
UPSCALE = 192   # FaceLandmarker 입력용 확대 크기
PAD = 32        # 둘레 여백


def auc(pos, neg):
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(np.mean(pos > neg) + 0.5 * np.mean(pos == neg))


def main():
    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True))
    rows = []
    for folder, label in FOLDERS.items():
        for path in sorted((DATA / folder).glob("*.png")):
            img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            big = cv2.copyMakeBorder(cv2.resize(img, (UPSCALE, UPSCALE)), PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=big))
            row = {"folder": folder, "file": path.name, "label": label,
                   "teacher_score": "", "eye_blink_left": "", "eye_blink_right": ""}
            if result.face_blendshapes:
                bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
                left, right = bs["eyeBlinkLeft"], bs["eyeBlinkRight"]
                row.update({"teacher_score": round(1 - max(left, right), 5),
                            "eye_blink_left": round(left, 5), "eye_blink_right": round(right, 5)})
            rows.append(row)
    with open(OUT, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{OUT} ({len(rows)}행)")
    for folder in FOLDERS:
        rs = [r for r in rows if r["folder"] == folder]
        scores = [r["teacher_score"] for r in rs if r["teacher_score"] != ""]
        print(f"  {folder:<16} {len(rs):>5}장, 미검출 {len(rs) - len(scores):>3}, 점수 중앙값 {np.median(scores):.3f}")
    rs = [r for r in rows if r["teacher_score"] != ""]
    pos = [r["teacher_score"] for r in rs if r["label"] == 1]
    neg = [r["teacher_score"] for r in rs if r["label"] == 0]
    print(f"  교사 AUC (전체): {auc(pos, neg):.3f}")


if __name__ == "__main__":
    main()
