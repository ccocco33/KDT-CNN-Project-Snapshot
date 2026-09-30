"""eye 학습 데이터에 GENKI 웃는 눈 검수분 추가 + 베이스라인 교사 점수
- 입력 (데이터 폴더 eye/)
  - preprocessed_cofw.zip, levels/levels_cofw.csv: 지금 학습 데이터 (원래 2,607장 + COFW 선글라스 85장)
  - genki_smile/review.csv (review.py 검수), candidates.csv: GENKI 사진 (../smile/preprocessed/)
  - levels/blendshapes_all.csv (원래 2,607장), genki_smile/blendshapes.csv: FaceLandmarker blendshape (192 확대 + 32 여백)
- GENKI 검수 단계 -> 폴더, 단계
  - 0 선글라스, 1 뜸 -> opened, 1 / 2 웃는 눈 -> opened, 2 / 3 감기는 눈 -> closed, 3 / 4 감은 눈 -> closed, 4 / 5 제외 -> 넣지 않음
  - 이미지는 그대로 복사 (64px, eye 와 같은 전처리)
- 교사 점수 (베이스라인): 1 - max(eyeBlinkLeft, eyeBlinkRight) (앱 yunet_landmark 의 눈 판정과 같은 값)
  - 노트북이 train 에서 확률로 보정해 씀 (10번 셀)
  - COFW 선글라스는 빈칸 (선글라스 위 blendshape 눈 감음은 믿을 수 없음. 정답만으로 학습)
- 출력 (eye/)
  - preprocessed_genki.zip: 드라이브 Team-11/dataset/eye/preprocessed.zip 으로 올림
  - levels/levels_genki.csv: 드라이브 eye/levels.csv 로 올림 (GENKI 는 source genki_review)
  - levels/teacher_baseline.csv (folder, file, teacher_score): 드라이브 eye/teacher_baseline.csv 로 올림

실행: python3 training/data/eye/genki_smile/build_dataset.py
"""
import csv
import sys
import zipfile
from pathlib import Path

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[2]))   # training/
from paths import DATASET  # noqa: E402

EYE = DATASET / "eye"
HERE = EYE / "genki_smile"   # 데이터 위치 (training/paths.py)
SMILE = DATASET / "smile" / "preprocessed"
FOLDER_LEVEL = {0: ("opened", 1), 1: ("opened", 1), 2: ("opened", 2), 3: ("closed", 3), 4: ("closed", 4)}


def teacher(b: dict) -> float:
    return round(1 - max(float(b["eyeBlinkLeft"]), float(b["eyeBlinkRight"])), 4)


def main():
    src = {r["file"]: r["src"] for r in csv.DictReader(open(HERE / "candidates.csv"))}
    review = [(r["file"], int(r["level"])) for r in csv.DictReader(open(HERE / "review.csv"))]
    genki = [(f, *FOLDER_LEVEL[lv]) for f, lv in review if lv in FOLDER_LEVEL]   # (file, folder, level)

    with zipfile.ZipFile(EYE / "preprocessed_cofw.zip") as base, \
            zipfile.ZipFile(EYE / "preprocessed_genki.zip", "w", zipfile.ZIP_DEFLATED) as out:
        names = set()
        for info in base.infolist():
            out.writestr(info, base.read(info))
            names.add(info.filename)
        for f, folder, _ in genki:
            arc = f"preprocessed/{folder}/{f}"
            assert arc not in names, arc
            out.write(SMILE / src[f], arc)
        n_img = sum(1 for n in out.namelist() if n.endswith(".png"))

    levels = list(csv.DictReader(open(EYE / "levels" / "levels_cofw.csv")))
    fields = list(levels[0])
    levels += [{"folder": folder, "file": f, "level": lv, "source": "genki_review"} for f, folder, lv in genki]
    with open(EYE / "levels" / "levels_genki.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(levels)

    base_bs = {(r["folder"], r["file"]): r for r in csv.DictReader(open(EYE / "levels" / "blendshapes_all.csv"))
               if r.get("eyeBlinkLeft")}
    genki_bs = {r["file"]: r for r in csv.DictReader(open(HERE / "blendshapes.csv"))}
    rows, missing = [], 0
    for r in levels:
        key = (r["folder"], r["file"])
        if r["source"] == "genki_review":
            score = teacher(genki_bs[r["file"]])
        elif key in base_bs:
            score = teacher(base_bs[key])
        else:
            score, missing = "", missing + 1   # COFW 선글라스, FaceLandmarker 미검출
        rows.append({"folder": r["folder"], "file": r["file"], "level": r["level"], "teacher_score": score})
    with open(EYE / "levels" / "teacher_baseline.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["folder", "file", "level", "teacher_score"])
        w.writeheader()
        w.writerows(rows)

    by = {}
    for _, folder, lv in genki:
        by[(folder, lv)] = by.get((folder, lv), 0) + 1
    print(f"GENKI 추가 {len(genki)}장 (검수 {len(review)}, 제외 {len(review) - len(genki)}): {sorted(by.items())}")
    print(f"zip 이미지 {n_img}장, levels {len(levels)}줄, 교사 점수 빈칸 {missing}")
    print(EYE / "preprocessed_genki.zip", EYE / "levels" / "levels_genki.csv", EYE / "levels" / "teacher_baseline.csv")


if __name__ == "__main__":
    main()
