"""eye 학습용 웃는 눈 후보: GENKI 웃음 사진에서 눈이 가늘게 잡힌 얼굴 N 장
- 목적: eye 학습 데이터에 웃는 눈 (단계 2) 이 적음 (126장). 웃느라 가늘어진 눈을 감음으로 보는 오판을 줄이기 위해 추가
- 입력: ../../smile/preprocessed/{smile_train,smile_test}/*.png (GENKI 웃음, 64px, eye 와 같은 전처리), manifest.csv (채도)
  - 흑백 제외 (채도 평균 < GRAY_SAT. eye 전처리와 같은 기준)
- 측정: FaceLandmarker (192 로 키우고 둘레 32px 검은 여백. eye 교사와 같음)
  - blendshape 전부 (./blendshapes.csv 에 저장. 교사 점수 만들 때 재사용)
  - 기하 눈 열림: 위 눈꺼풀 중앙 ~ 아래 눈꺼풀 중앙 거리 / 눈 가로 길이. 두 눈 중 작은 쪽 (geo_min)
- 선택: FaceLandmarker 가 찾은 얼굴 중 geo_min 이 작은 순 N 장 (눈이 가장 가늘게 잡힌 얼굴)
- 처음 단계 (검수 전): blendshape 눈 감음 최대 < AUTO_OPEN_BLINK 이면 2 (웃는 눈), 아니면 3 (감기는 눈. 검수 필요)
- 출력 (데이터 폴더 eye/genki_smile/): candidates.csv (file, src, saturation, blink, squint, geo_min, level), blendshapes.csv, candidates_grid.png (geo_min 순)

실행: app/.venv/bin/python training/data/eye/genki_smile/select_candidates.py
"""
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[2]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "eye" / "genki_smile"   # 데이터 위치 (training/paths.py)
SMILE = DATASET / "smile" / "preprocessed"
MODEL = REPO / "app/models/face_landmarker.task"
FOLDERS = ("smile_train", "smile_test")
N = 300
GRAY_SAT = 25
AUTO_OPEN_BLINK = 0.35
UPSCALE, PAD = 192, 32
EYES = ((33, 133, 159, 145), (362, 263, 386, 374))   # (눈꼬리, 눈꼬리, 위 눈꺼풀 중앙, 아래 눈꺼풀 중앙)


def geo_min(landmarks, w, h) -> float:
    """두 눈 중 작은 눈 열림. 두 눈 중심 선을 수평으로 돌린 뒤 계산"""
    p = np.array([[q.x * w, q.y * h] for q in landmarks])
    centers = [(p[a] + p[b]) / 2 for a, b, _, _ in EYES]
    angle = math.atan2(centers[1][1] - centers[0][1], centers[1][0] - centers[0][0])
    rot = np.array([[math.cos(-angle), -math.sin(-angle)], [math.sin(-angle), math.cos(-angle)]])
    p = p @ rot.T
    return float(min(np.linalg.norm(p[t] - p[b2]) / (np.linalg.norm(p[a] - p[b]) or 1e-6) for a, b, t, b2 in EYES))


def main():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    sat = {r["dst"]: float(r["saturation"]) for r in csv.DictReader(open(SMILE / "manifest.csv")) if r["status"] == "ok"}
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.IMAGE,
        num_faces=1, output_face_blendshapes=True))
    rows, blends, names = [], [], None
    for folder in FOLDERS:
        for path in sorted((SMILE / folder).glob("*.png")):
            rel = f"{folder}/{path.name}"
            if sat.get(rel, 0) < GRAY_SAT:
                continue
            img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            big = cv2.copyMakeBorder(cv2.resize(img, (UPSCALE, UPSCALE), interpolation=cv2.INTER_CUBIC),
                                     PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT)
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=big))
            if not res.face_blendshapes:
                continue
            b = {c.category_name: round(c.score, 5) for c in res.face_blendshapes[0]}
            names = names or list(b)
            file = f"genki_{folder}_{path.stem}.png"   # eye 데이터에서 겹치지 않는 이름
            blends.append({"file": file, **b})
            rows.append({"file": file, "src": rel, "saturation": sat[rel],
                         "blink": round(max(b["eyeBlinkLeft"], b["eyeBlinkRight"]), 4),
                         "squint": round(max(b["eyeSquintLeft"], b["eyeSquintRight"]), 4),
                         "geo_min": round(geo_min(res.face_landmarks[0], big.shape[1], big.shape[0]), 4)})
    rows.sort(key=lambda r: r["geo_min"])
    picked = rows[:N]
    for r in picked:
        r["level"] = 2 if r["blink"] < AUTO_OPEN_BLINK else 3
    HERE.mkdir(parents=True, exist_ok=True)
    with open(HERE / "candidates.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(picked[0]))
        w.writeheader()
        w.writerows(picked)
    keep = {r["file"] for r in picked}
    with open(HERE / "blendshapes.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["file"] + names)
        w.writeheader()
        w.writerows(b for b in blends if b["file"] in keep)
    tiles = []
    for r in picked:
        t = cv2.resize(cv2.imread(str(SMILE / r["src"])), (128, 128), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(t, (0, 104), (128, 128), (0, 0, 0), -1)
        cv2.putText(t, f"g{r['geo_min']:.2f} b{r['blink']:.2f}", (3, 120), 0, 0.4,
                    (0, 255, 255) if r["level"] == 2 else (0, 128, 255), 1)
        tiles.append(t)
    while len(tiles) % 20:
        tiles.append(np.zeros((128, 128, 3), np.uint8))
    cv2.imwrite(str(HERE / "candidates_grid.png"), np.vstack([np.hstack(tiles[i:i + 20]) for i in range(0, len(tiles), 20)]))
    print(f"컬러 + 검출 {len(rows)}장 중 {len(picked)}장. 처음 단계 2 (웃는 눈) {sum(r['level'] == 2 for r in picked)}, "
          f"3 (검수) {sum(r['level'] == 3 for r in picked)}. geo_min {picked[0]['geo_min']} ~ {picked[-1]['geo_min']}")
    print(HERE / "candidates_grid.png")


if __name__ == "__main__":
    main()
