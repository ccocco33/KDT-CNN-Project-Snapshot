"""eye 학습 노트북 생성기 (셀 내용 -> .ipynb)
- 합성 안경류 함수(선글라스, 투명 안경)는 training/data/eye/augment_review/eyewear_fn.py 에서 가져옴 (SUNGLASSES_SRC)
- 실행: python training/notebooks/build_eye_nb.py training/notebooks/eye_train.ipynb training/data/eye/augment_review/eyewear_fn.py
"""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
SUNGLASSES_SRC = Path(sys.argv[2]).read_text().rstrip()

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md('''
# eye 분류 모델 학습 (눈 뜸 / 감음)

- 입력: YuNet 으로 자른 얼굴 정사각형의 **위쪽 `EYE_TOP` 비율**을 `IMG_SIZE` x `IMG_SIZE` 로 늘린 이미지
  - 전처리 데이터 zip `DRIVE_FACE/eye/preprocessed.zip` (64x64 PNG, opened / closed)
- 출력: 눈 뜸 점수 (0~1, **1 = 뜸**)
- 모델: ImageNet 사전학습 백본 + 전역 평균 풀링 + 이진 분류. 백본까지 한 번에 학습
- 눈 뜬 정도 4단계 목표: 뜸 > 웃는 눈 > 감기는 눈 > 감은 눈 순으로 점수가 낮아지도록 단계 점수(LEVEL_TARGETS)를 목표로 학습
  - 단계 라벨: `DRIVE_FACE/eye/levels.csv` (blendshape 로 뽑은 후보를 사람이 검수. 나머지는 폴더 라벨)
  - 뜸/감음 (평가, 임계값): 단계 1, 2 는 뜸, 3, 4 는 감음
- 지식 증류: 기하 교사 점수를 눈 뜸 확률로 바꿔 단계 점수와 섞음
  - 기하 교사: FaceLandmarker 랜드마크, 얼굴 회전으로 잰 눈 모양(위 눈꺼풀 휨), 눈 열림, 입꼬리, 아래봄(pitch)을 백분위로 바꿔 가중 평균
  - 만든 곳: training/data/eye/levels/ (geometry_metrics.py, geometry_teacher.py)
- 합성 선글라스 증강: 눈 좌표 위에 선글라스를 그리고 목표를 "뜸" 으로 (요구사항: 선글라스류로 눈이 보이지 않으면 눈 조건 충족)
- 합성 투명 안경 증강: 테, 옅은 렌즈 색, 반사광을 그리고 목표는 그대로 (안경테, 반사를 "뜸" 의 단서로 쓰지 않도록)
- 임계값: 감은 눈 recall 을 먼저 맞춤 (감은 눈을 뜸으로 판정하면 눈 감은 사진이 결과로 나감)
- 결과: tflite (fp32, dynamic range, int8), 추론 설정 json, 지표 (`OUTPUT_ROOT/RUN_NAME`)
- 실행: Colab GPU 런타임에서 위에서 아래로. 설정은 1~6번 셀에서만 바꿈
''')

code('''
# 1. 경로 설정
# - DRIVE_FACE: 데이터셋 루트 (Google Drive)
# - DATA_ZIP: 전처리한 eye 데이터 zip. 안에 CLASS_DIRS 의 폴더 (한 겹 상위 폴더가 있어도 됨)
# - LOCAL_DATA_DIR: zip 을 풀 로컬 경로 (Drive 에서 직접 읽으면 느림). 이미 풀려 있으면 다시 풀지 않음
# - TEACHER_CSV: 지식 증류용 교사 점수 (folder, file, level, teacher_score). 베이스라인 교사: 1 - blendshape 눈 감음 최대
#   (training/data/eye/genki_smile/build_dataset.py. 이전 기하 교사는 eye/teacher_geometry.csv). DISTILL_ALPHA 가 1.0 이면 안 씀
# - LANDMARKS_CSV: 합성 선글라스용 두 눈 좌표 (class, file, rx, ry, lx, ly. 64px 기준). AUG_SUNGLASSES_PROB 가 0 이면 안 씀
# - LEVELS_CSV: 눈 뜬 정도 4단계 라벨 (folder, file, level, source). level 1 뜸 / 2 웃는 눈 / 3 감기는 눈 / 4 감은 눈
#   - 만든 곳: training/data/eye/levels/ (review.py 로 검수, make_levels.py 로 생성)
# - OUTPUT_ROOT: 학습 결과 저장 위치. 실행마다 RUN_NAME 하위 폴더 생성
# - RUN_NAME: None 이면 실행 시각 (예: 20260926-1530)
DRIVE_FACE = "/content/drive/MyDrive/Team-11/dataset"
DATA_ZIP = f"{DRIVE_FACE}/eye/preprocessed.zip"
LOCAL_DATA_DIR = "/content/data/eye"
TEACHER_CSV = f"{DRIVE_FACE}/eye/teacher_baseline.csv"
LANDMARKS_CSV = f"{DRIVE_FACE}/eye/eye_landmarks.csv"
LEVELS_CSV = f"{DRIVE_FACE}/eye/levels.csv"
OUTPUT_ROOT = "/content/drive/MyDrive/Team-11/models/eye"
RUN_NAME = None
''')

code('''
# 2. 데이터 설정
# - CLASS_DIRS: 읽을 폴더 (값은 폴더 라벨. 1 = 뜸, 0 = 감음). 학습, 평가 라벨은 LEVELS_CSV 의 단계를 씀
# - VAL_FRACTION, TEST_FRACTION: 검증, 시험 비율 (eye 데이터는 시험 폴더가 따로 없어 전체에서 나눔)
# - DHASH_DISTANCE: 비슷한 이미지로 묶는 dHash 거리 (이하면 같은 묶음. 묶음은 분할을 넘지 않음)
# - SEED: 난수 시드
CLASS_DIRS = {"opened": 1, "closed": 0}
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
DHASH_DISTANCE = 4
SEED = 42
''')

code('''
# 3. 모델 설정
# - BACKBONE: mobilenet_v3_small | mobilenet_v3_large | mobilenet_v2 | efficientnet_b0
# - ALPHA: MobileNet 폭 배율 (efficientnet_b0 는 무시)
# - EYE_TOP: 얼굴 정사각형에서 쓸 위쪽 비율 (눈 포함)
# - IMG_SIZE: 모델 입력 크기. 위쪽 EYE_TOP 영역을 IMG_SIZE x IMG_SIZE 로 늘림 (가로세로 비율 유지 안 함)
# - DROPOUT: 분류 직전 드롭아웃 비율
# - TRAINABLE_BACKBONE_LAYERS: None 이면 백본 전체 학습, 정수 N 이면 백본 마지막 N 개 층만 학습
BACKBONE = "mobilenet_v3_small"
ALPHA = 1.0
EYE_TOP = 0.6
IMG_SIZE = 160
DROPOUT = 0.3
TRAINABLE_BACKBONE_LAYERS = None
''')

code('''
# 4. 학습 설정
# - LEARNING_RATE: 최대 학습률. WARMUP_EPOCHS 동안 1/10 에서 올라간 뒤 코사인으로 감소
# - WEIGHT_DECAY: AdamW 가중치 감쇠
# - LABEL_SMOOTHING: 라벨 스무딩 (0 이면 끔)
# - EARLY_STOP_PATIENCE: 검증 AUC 가 이 epoch 수 동안 오르지 않으면 중단 (가장 좋은 가중치로 복원)
# - USE_CLASS_WEIGHT: 뜸/감음 수 차이 보정 (샘플 가중치)
# - USE_EMA: 가중치 이동 평균(EMA)을 최종 모델로. 검증, 체크포인트도 EMA 가중치로 (과적합 완화)
# - EMA_MOMENTUM: EMA 비율. 클수록 과거 가중치를 오래 반영 (0.99: 약 100 step)
# - LEVEL_TARGETS: 단계 -> 학습 목표 점수. 뜸 > 웃는 눈 > 감기는 눈 > 감은 눈 순서를 배우도록
#   - 앱 기준값은 웃는 눈(0.7)과 감기는 눈(0.3) 사이가 되도록 (웃는 눈은 통과, 감기는 눈은 감음)
# - DISTILL_ALPHA: 지식 증류에서 단계 점수 비중 a. 학습 목표 = a x 단계 점수 + (1 - a) x 교사 확률. 1.0 이면 증류 끔
#   - 검수하지 않은 이미지(단계가 폴더 라벨로 정해짐)에도 눈을 뜬 정도가 들어가도록 교사와 섞음
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WARMUP_EPOCHS = 2
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 8
USE_CLASS_WEIGHT = True
USE_EMA = True
EMA_MOMENTUM = 0.99
LEVEL_TARGETS = {1: 1.0, 2: 0.7, 3: 0.3, 4: 0.0}
DISTILL_ALPHA = 0.5
''')

code('''
# 5. 증강 설정 (학습 데이터에만 적용. 확률 0 또는 범위 0 이면 끔)
# - 실제 카메라에서 생기는 변화만 흔듦. 흑백은 제외 (실제 카메라는 컬러)
# - 합성 선글라스: 확률 AUG_SUNGLASSES_PROB 로 두 눈 위에 선글라스를 그리고 목표를 "뜸"(1) 으로
#   - 렌즈 종류는 eyewear_fn.SUNGLASS_STYLES 확률로 (dark, reflect, mirror, light)
#   - light (옅은 렌즈, 눈이 비침) 는 목표를 바꾸지 않음 (투명 안경과 같음)
#   - 렌즈 불투명도 하한 (dark): 뜬 눈 SUNGLASSES_MIN_ALPHA_OPEN, 감은 눈 SUNGLASSES_MIN_ALPHA_CLOSED (감은 눈이 비치면 "뜸" 과 모순)
AUG_SUNGLASSES_PROB = 0.10
SUNGLASSES_MIN_ALPHA_OPEN = 0.70
SUNGLASSES_MIN_ALPHA_CLOSED = 0.90
# - 합성 투명 안경: 선글라스를 씌우지 않은 이미지에 확률 AUG_CLEAR_GLASSES_PROB 로 투명 안경을 그림. 목표는 그대로
#   - 뜬 눈, 감은 눈에 같은 확률 (한쪽에만 그리면 "안경 = 그 클래스" 를 배움)
#   - 투명 안경에 반사가 비친 감은 눈을 뜸으로 판정한 사례가 있어 추가
AUG_CLEAR_GLASSES_PROB = 0.30
# - 기하: 좌우 뒤집기, 회전(도), 확대/축소(비율 ±), 이동(비율 ±). 빈 곳은 반사로 채움
#   - 확대/축소는 smile 보다 넓게 (opened, closed 의 구도 차이를 무작위화로 덮음)
AUG_FLIP = True
AUG_ROTATION_DEG = 20
AUG_ZOOM = 0.15
AUG_TRANSLATE = 0.08
# - 해상도: 확률 AUG_DOWNSCALE_PROB 로 [MIN, MAX] px 로 줄였다 원래 크기로 키움 (멀리 있는 작은 얼굴)
AUG_DOWNSCALE_PROB = 0.5
AUG_DOWNSCALE_MIN = 24
AUG_DOWNSCALE_MAX = 56
# - 흐림: 확률 AUG_BLUR_PROB 로 가우시안 흐림, 시그마 최대 AUG_BLUR_SIGMA_MAX (px, 64px 원본 기준)
AUG_BLUR_PROB = 0.2
AUG_BLUR_SIGMA_MAX = 1.2
# - 색: 밝기(± 0~1 척도), 대비(배율 ±), 채도(배율 ±). 밝기는 smile 보다 넓게 (closed 가 더 밝은 치우침을 덮음)
AUG_BRIGHTNESS = 0.20
AUG_CONTRAST = 0.2
AUG_SATURATION = 0.3
# - 잡음: 확률 AUG_NOISE_PROB 로 가우시안 잡음, 표준편차 최대 AUG_NOISE_STD_MAX (0~1 척도). 어두운 실내 촬영이면 켬
AUG_NOISE_PROB = 0.0
AUG_NOISE_STD_MAX = 0.03
''')

code('''
# 6. 평가, 내보내기 설정
# - TARGET_CLOSED_RECALL: 감은 눈 recall 목표. 검증 데이터에서 이 값을 만족하는 임계값 중 눈 뜸 recall 최대
#   - 감은 눈을 뜸으로 판정하면 눈 감은 사진이 결과로 나감. 뜬 눈을 감음으로 판정하는 것은 다른 프레임이 메움
#   - 감은 눈 쪽만 보고 정하므로 데이터의 감은 눈 비율에 덜 흔들림 (뜸 precision 기준은 감은 눈이 드문 실제 사진에서 느슨해짐)
#   - 앱의 최종 기준값은 단체 사진 평가(WIDER dev)로 따로 정함. 이 값은 추천값
# - REAL_SUNGLASSES: opened 폴더의 실제 선글라스 이미지 (val, test 에 있으면 뜸 판정 비율을 따로 확인)
#   - 이름이 REAL_SUNGLASSES_PREFIX 로 시작하는 이미지도 실제 선글라스 (COFW 에서 검수해 추가한 선글라스 얼굴)
# - tflite 변환 (fp32 는 항상)
#   - EXPORT_DYNAMIC: dynamic range 양자화. 가중치 int8, 입출력 float32. 대표 데이터 불필요
#   - EXPORT_INT8: 전체 int8 양자화. 가중치, 활성값, 입출력 int8
# - TTA_FLIP: 원본과 좌우 뒤집은 입력의 점수 평균을 모델 안에 넣어 내보냄 (앱 코드 변경 없음, 추론 비용 2배)
# - REPRESENTATIVE_SAMPLES: int8 양자화에서 활성값 범위를 잴 학습 이미지 수 (뜸, 감음 반씩)
TARGET_CLOSED_RECALL = 0.95
REAL_SUNGLASSES = ["01450", "00184", "00873", "01114", "01484", "00720", "00265", "00786", "00951",
                   "01142", "00824", "00836", "00257"]
REAL_SUNGLASSES_PREFIX = "cofw_"
TTA_FLIP = True
EXPORT_DYNAMIC = True
EXPORT_INT8 = True
REPRESENTATIVE_SAMPLES = 300
''')

code('''
# 7. 환경 준비
# - Colab 이면 Google Drive 연결, tflite 검증용 패키지 설치
# - 버전, GPU 확인, 난수 시드 고정
# - 결과 폴더 생성
import json
import math
import subprocess
import sys
import time
from pathlib import Path

try:
    from google.colab import drive
    drive.mount("/content/drive")
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "ai-edge-litert", "scikit-learn"], check=True)
except ImportError:
    pass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

keras.utils.set_random_seed(SEED)
print("TensorFlow", tf.__version__, "/ Keras", keras.__version__)
print("GPU:", tf.config.list_physical_devices("GPU") or "없음 (CPU 로 실행)")

RUN_NAME = RUN_NAME or time.strftime("%Y%m%d-%H%M")
OUT = Path(OUTPUT_ROOT) / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
print("결과 폴더:", OUT)
''')

code('''
# 8. 데이터 읽기
# - DATA_ZIP 을 LOCAL_DATA_DIR 에 풂 (맥 zip 의 __MACOSX, ._* 는 제외)
# - CLASS_DIRS 의 폴더가 모두 있는 위치를 찾음 (zip 안 상위 폴더 유무와 무관)
# - CLASS_DIRS 의 폴더에서 이미지를 모두 읽음 (RGB, 0~255). 이미지마다 dHash 계산 (비슷한 이미지 묶기용)
# - 모든 이미지는 같은 크기여야 함 (전처리 결과 64x64)
# - 단계 라벨(LEVELS_CSV), 교사 점수(TEACHER_CSV), 두 눈 좌표(LANDMARKS_CSV)를 폴더, 파일 이름으로 연결
#   - 단계가 없는 이미지가 있으면 중단. 교사 점수, 눈 좌표는 없으면 빈 값
#   - label (뜸/감음) 은 단계로 정함 (폴더 라벨이 아님)
import zipfile

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def is_junk(name):
    return "__MACOSX" in Path(name).parts or Path(name).name.startswith("._")


def find_data_root(base):
    for d in [base, *sorted(p for p in base.rglob("*") if p.is_dir())]:
        if all((d / folder).is_dir() for folder in CLASS_DIRS):
            return d
    raise FileNotFoundError(f"{base} 아래에 {list(CLASS_DIRS)} 폴더가 모두 있는 위치가 없음")


local = Path(LOCAL_DATA_DIR)
done_mark = local / ".unzipped"
if not done_mark.exists():
    local.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATA_ZIP) as zf:
        members = [m for m in zf.namelist() if not is_junk(m)]
        zf.extractall(local, members=members)
    done_mark.write_text(DATA_ZIP)
DATA_DIR = find_data_root(local)
print("데이터 위치:", DATA_DIR)


def dhash(img):
    gray = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (gray[:, 1:] > gray[:, :-1]).reshape(-1)
    return int("".join("1" if b else "0" for b in bits), 2)


rows, images = [], []
for folder, label in CLASS_DIRS.items():
    paths = sorted(p for p in (DATA_DIR / folder).iterdir()
                   if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._"))
    for p in paths:
        with Image.open(p) as src:
            img = src.convert("RGB")
        rows.append({"path": str(p), "folder": folder, "file": p.name, "label": label, "dhash": dhash(img)})
        images.append(np.asarray(img))

df = pd.DataFrame(rows)
levels = pd.read_csv(LEVELS_CSV)[["folder", "file", "level"]]
df = df.merge(levels, on=["folder", "file"], how="left")
assert df.level.notna().all(), f"단계 라벨이 없는 이미지 {df.level.isna().sum()}장"
df["level"] = df.level.astype(int)
df["label"] = (df.level <= 2).astype(int)   # 뜸/감음: 1, 2 는 뜸, 3, 4 는 감음
print("단계 x 폴더:")
display(pd.crosstab(df.level, df.folder))
if DISTILL_ALPHA < 1:
    teacher = pd.read_csv(TEACHER_CSV)[["folder", "file", "teacher_score"]]
    df = df.merge(teacher, on=["folder", "file"], how="left")
    print(f"교사 점수 연결: {df.teacher_score.notna().sum()} / {len(df)}")
else:
    df["teacher_score"] = np.nan
if AUG_SUNGLASSES_PROB > 0:
    lm = pd.read_csv(LANDMARKS_CSV).rename(columns={"class": "folder"})[["folder", "file", "rx", "ry", "lx", "ly"]]
    df = df.merge(lm, on=["folder", "file"], how="left")
    print(f"눈 좌표 연결: {df.rx.notna().sum()} / {len(df)}")
else:
    df[["rx", "ry", "lx", "ly"]] = np.nan
sizes = {im.shape for im in images}
assert len(sizes) == 1, f"이미지 크기가 섞여 있음: {sizes}"
X = np.stack(images)
SRC_SIZE = X.shape[1]
print("이미지:", X.shape)
display(df.groupby("folder").size().rename("장수"))
''')

code('''
# 9. 중복 정리와 분할
# - dHash 거리 DHASH_DISTANCE 이하인 이미지를 한 묶음으로 (정확한 중복 포함)
# - 묶음 단위로 test, val, train 을 나눔 (단계별 TEST_FRACTION, VAL_FRACTION 에 가깝게. 드문 단계도 val, test 에 들어가도록)
# - 결과: df["split"] = train | val | test
hashes = df["dhash"].tolist()
parent = list(range(len(df)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for i in range(len(hashes)):
    for j in range(i):
        if (hashes[i] ^ hashes[j]).bit_count() <= DHASH_DISTANCE:
            parent[find(i)] = find(j)
df["group"] = [find(i) for i in range(len(df))]

mixed = df.groupby("group")["level"].nunique()
print("단계가 섞인 묶음 (라벨 확인 필요):", int((mixed > 1).sum()))

rng = np.random.default_rng(SEED)
df["split"] = "train"
group_level = df.groupby("group")["level"].agg(lambda s: int(s.mode().iloc[0]))
group_size = df.groupby("group").size()
for level in sorted(LEVEL_TARGETS):
    groups = group_level.index[group_level == level].to_numpy().copy()
    rng.shuffle(groups)
    n_label = (df.level == level).sum()
    taken = {"test": 0, "val": 0}
    for g in groups:
        if taken["test"] < TEST_FRACTION * n_label:
            split = "test"
        elif taken["val"] < VAL_FRACTION * n_label:
            split = "val"
        else:
            break
        df.loc[df.group == g, "split"] = split
        taken[split] += group_size[g]

for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    assert not set(df.loc[df.split == a, "group"]) & set(df.loc[df.split == b, "group"]), f"{a}, {b} 묶음 겹침"
df.to_csv(OUT / "split_manifest.csv", index=False)
display(df.groupby(["split", "level"]).size().unstack().rename(columns={1: "뜸", 2: "웃는 눈", 3: "감기는 눈", 4: "감은 눈"}))
idx = {s: df.index[df.split == s].to_numpy() for s in ("train", "val", "test")}
Y = df["label"].to_numpy(np.float32)
LEVEL = df["level"].to_numpy(int)
LM = df[["rx", "ry", "lx", "ly"]].to_numpy(np.float32)
''')

code('''
# 10. 학습 목표 (단계 점수, 선택: 교사 확률)
# - 기본 목표: 단계 점수 LEVEL_TARGETS[level]
# - 교사 점수 s (기하 지표 백분위의 가중 평균, 0~1) 는 확률이 아니므로 눈 뜸 확률 p 로 보정: p = sigmoid(w x s + b)
#   - 정답은 단계로 정한 뜸(1, 2) / 감음(3, 4)
#   - w, b 는 train 분할만으로 맞춤 (로지스틱 회귀). val, test 는 보정에 쓰지 않음
# - 증류를 켜면 (DISTILL_ALPHA < 1) T = DISTILL_ALPHA x 단계 점수 + (1 - DISTILL_ALPHA) x p. 교사 미검출 이미지는 단계 점수 그대로
# - 합성 선글라스를 씌운 이미지는 학습 중에 목표를 1 로 바꿈 (12번 셀)
# - 검증, 시험은 뜸/감음(Y) 와 단계(LEVEL) 로 평가
T = np.array([LEVEL_TARGETS[lv] for lv in LEVEL], np.float32)
print("학습 목표 (train):", pd.Series(T[idx["train"]]).value_counts().sort_index().to_dict())
if DISTILL_ALPHA < 1:
    from sklearn.linear_model import LogisticRegression

    s_all = df["teacher_score"].to_numpy(float)
    fit_idx = idx["train"][~np.isnan(s_all[idx["train"]])]
    calib = LogisticRegression(C=1e6).fit(s_all[fit_idx, None], Y[fit_idx])
    w, b = float(calib.coef_[0, 0]), float(calib.intercept_[0])
    has = ~np.isnan(s_all)
    p_teacher = np.full(len(df), np.nan)
    p_teacher[has] = calib.predict_proba(s_all[has, None])[:, 1]
    train_has = idx["train"][has[idx["train"]]]
    T[train_has] = DISTILL_ALPHA * T[train_has] + (1 - DISTILL_ALPHA) * p_teacher[train_has]
    print(f"보정: p = sigmoid({w:.2f} x s + {b:.2f}), p = 0.5 인 s = {-b / w:.3f}")
    print("교사 점수 s -> 눈 뜸 확률 p:", ", ".join(f"{v:.1f}->{1 / (1 + np.exp(-(w * v + b))):.2f}" for v in (0.2, 0.4, 0.6, 0.8, 1.0)))
    val_has = idx["val"][has[idx["val"]]]
    print(f"교사 AUC (val, 참고): {roc_auc_score(Y[val_has], s_all[val_has]):.4f}")
    print(f"학습 목표: 교사 반영 {len(train_has)}장, 정답만 {len(idx['train']) - len(train_has)}장")
    fig, ax = plt.subplots(figsize=(5, 3))
    for c, name in ((1, "open"), (0, "closed")):
        ax.hist(T[idx["train"]][Y[idx["train"]] == c], bins=40, alpha=0.6, label=name)
    ax.set_title("train target distribution")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("증류 끔 (DISTILL_ALPHA = 1.0): 정답 라벨로만 학습")
''')

code('''
# 11. 합성 안경류 (증강)
# - 선글라스 (add_sunglasses): 두 눈 위에 렌즈와 테. 종류 dark, reflect (밝은 반사), mirror (색 미러), light (옅은 렌즈). 목표는 "뜸", light 만 그대로
# - 투명 안경 (add_clear_glasses): 테, 옅은 렌즈 색, 반사광. 눈이 비쳐 보여 목표는 그대로
# - 렌즈 모양(타원, 둥근 사각형, 보잉, 캣아이), 크기, 색, 불투명도, 반사광은 무작위. 두 눈 각도만큼 기울임
# - 검토 스크립트: training/data/eye/augment_review/sunglasses_preview.py, clear_glasses_preview.py (같은 함수)
''' + SUNGLASSES_SRC + '''


def sunglasses_np(image, lm, label, force=False, style=None):
    """학습 증강용 (numpy). 반환: (이미지, 적용 결과 0 안 씌움 / 1 씌움, 목표 "뜸" / 2 light, 목표 그대로)
    - AUG_SUNGLASSES_PROB 확률 (force 면 항상). 눈 좌표가 없으면 적용 안 함
    - style: 렌즈 종류 고정 (None 이면 SUNGLASS_STYLES 확률로)
    """
    rng = np.random.default_rng()
    if np.isnan(lm).any() or (not force and rng.random() >= AUG_SUNGLASSES_PROB):
        return image, np.float32(0)
    style = style or pick_sunglass_style(rng)
    min_alpha = SUNGLASSES_MIN_ALPHA_OPEN if label >= 0.5 else SUNGLASSES_MIN_ALPHA_CLOSED
    out = add_sunglasses(np.ascontiguousarray(image), (lm[0], lm[1]), (lm[2], lm[3]), rng, min_alpha=min_alpha, style=style)
    return out.astype(np.uint8), np.float32(2 if style == "light" else 1)


def clear_glasses_np(image, lm, force=False):
    """학습 증강용 (numpy). 반환: 이미지
    - AUG_CLEAR_GLASSES_PROB 확률 (force 면 항상). 눈 좌표가 없으면 적용 안 함
    """
    rng = np.random.default_rng()
    if np.isnan(lm).any() or (not force and rng.random() >= AUG_CLEAR_GLASSES_PROB):
        return image
    return add_clear_glasses(np.ascontiguousarray(image), (lm[0], lm[1]), (lm[2], lm[3]), rng).astype(np.uint8)
''')

code('''
# 12. 입력 파이프라인
# - 공통 (학습, 검증, 시험, 앱): 0~1 로 바꿔 처리 -> 위쪽 EYE_TOP 영역 -> IMG_SIZE x IMG_SIZE 로 bilinear 늘림 -> 0~255
#   - 모델 입력은 RGB 0~255. 정규화는 모델 안에서
# - 학습만 (이미지마다 무작위)
#   - 순서: 합성 안경류 (64px 눈 좌표 기준이라 가장 먼저) -> 기하 -> 화질 -> 색 -> 위쪽 영역
#     - 선글라스를 씌우지 않은 이미지에만 투명 안경
#   - 목표: 10번 셀의 T. 합성 선글라스를 씌우면 1, 투명 안경은 그대로
# - 검증, 시험: 증강 없음, 목표는 정답 Y
UPPER_H = int(round(SRC_SIZE * EYE_TOP))


def affine(img):
    """회전, 확대/축소, 이동을 한 번에. 빈 곳은 반사로 채움"""
    s = tf.cast(SRC_SIZE, tf.float32)
    theta = tf.random.uniform([], -AUG_ROTATION_DEG, AUG_ROTATION_DEG) * np.pi / 180
    zoom = tf.random.uniform([], 1 - AUG_ZOOM, 1 + AUG_ZOOM)
    tx = tf.random.uniform([], -AUG_TRANSLATE, AUG_TRANSLATE) * s
    ty = tf.random.uniform([], -AUG_TRANSLATE, AUG_TRANSLATE) * s
    c = (s - 1) / 2
    cos, sin = tf.cos(theta) / zoom, tf.sin(theta) / zoom
    # 출력 좌표 -> 입력 좌표 변환 [a0, a1, a2, b0, b1, b2, 0, 0]
    t = tf.stack([cos, -sin, c - cos * c + sin * c - tx, sin, cos, c - sin * c - cos * c - ty, 0.0, 0.0])
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None], transforms=t[None], output_shape=tf.shape(img)[:2],
        fill_value=0.0, interpolation="BILINEAR", fill_mode="REFLECT")
    return out[0]


def downscale(img):
    """줄였다 키우기"""
    size = tf.random.uniform([], AUG_DOWNSCALE_MIN, AUG_DOWNSCALE_MAX + 1, dtype=tf.int32)
    small = tf.image.resize(img, [size, size], method="area")
    return tf.image.resize(small, [SRC_SIZE, SRC_SIZE], method="bilinear")


def blur(img):
    """가우시안 흐림 (커널 5x5)"""
    sigma = tf.random.uniform([], 0.3, AUG_BLUR_SIGMA_MAX)
    x = tf.range(-2.0, 3.0)
    k1 = tf.exp(-(x ** 2) / (2 * sigma ** 2))
    k1 = k1 / tf.reduce_sum(k1)
    k2 = tf.tensordot(k1, k1, axes=0)[:, :, None, None]
    kernel = tf.tile(k2, [1, 1, 3, 1])
    padded = tf.pad(img[None], [[0, 0], [2, 2], [2, 2], [0, 0]], mode="REFLECT")
    return tf.nn.depthwise_conv2d(padded, kernel, strides=[1, 1, 1, 1], padding="VALID")[0]


def maybe(prob, fn, img):
    if prob <= 0:
        return img
    return tf.cond(tf.random.uniform([]) < prob, lambda: fn(img), lambda: img)


def augment(img):
    """합성 선글라스를 뺀 나머지 증강 (0~1 이미지, 64px 얼굴 전체)"""
    if AUG_FLIP:
        img = tf.image.random_flip_left_right(img)
    if AUG_ROTATION_DEG or AUG_ZOOM or AUG_TRANSLATE:
        img = affine(img)
    img = maybe(AUG_DOWNSCALE_PROB, downscale, img)
    img = maybe(AUG_BLUR_PROB, blur, img)
    if AUG_BRIGHTNESS:
        img = img + tf.random.uniform([], -AUG_BRIGHTNESS, AUG_BRIGHTNESS)
    if AUG_CONTRAST:
        img = tf.image.adjust_contrast(img, tf.random.uniform([], 1 - AUG_CONTRAST, 1 + AUG_CONTRAST))
    if AUG_SATURATION:
        img = tf.image.adjust_saturation(tf.clip_by_value(img, 0, 1),
                                         tf.random.uniform([], 1 - AUG_SATURATION, 1 + AUG_SATURATION))
    img = maybe(AUG_NOISE_PROB, lambda im: im + tf.random.normal(tf.shape(im)) * tf.random.uniform([], 0, AUG_NOISE_STD_MAX), img)
    return tf.clip_by_value(img, 0.0, 1.0)


def to_model_input(img):
    """0~1 얼굴 이미지 (정사각형) -> 모델 입력 (위쪽 EYE_TOP, IMG_SIZE x IMG_SIZE, bilinear, 0~255)"""
    upper = img[:UPPER_H]
    return tf.image.resize(upper, [IMG_SIZE, IMG_SIZE], method="bilinear") * 255.0


def add_sunglasses_tf(image, lm, label):
    out, flag = tf.numpy_function(sunglasses_np, [image, lm, label], [tf.uint8, tf.float32])
    out.set_shape(image.shape)
    flag.set_shape([])
    return out, flag


def add_clear_glasses_tf(image, lm):
    out = tf.numpy_function(clear_glasses_np, [image, lm], tf.uint8)
    out.set_shape(image.shape)
    return out


# - 학습: 샘플 가중치로 뜸/감음 수 차이 보정 (USE_CLASS_WEIGHT). 목표가 0.7 같은 소수라 class_weight 대신 사용
counts = np.bincount(Y[idx["train"]].astype(int), minlength=2)
CLASS_W = np.array([len(idx["train"]) / (2 * counts[c]) for c in (0, 1)], np.float32) if USE_CLASS_WEIGHT else np.ones(2, np.float32)
print("뜸/감음 샘플 가중치:", CLASS_W.round(3).tolist())


def make_dataset(indices, training):
    targets = T if training else Y
    weights = CLASS_W[Y[indices].astype(int)]
    ds = tf.data.Dataset.from_tensor_slices((X[indices], targets[indices].astype(np.float32), Y[indices], LM[indices], weights))
    if training:
        ds = ds.shuffle(len(indices), seed=SEED, reshuffle_each_iteration=True)

    def prep(image, target, label, lm, weight):
        if training and AUG_SUNGLASSES_PROB > 0:
            image, flag = add_sunglasses_tf(image, lm, label)
            target = tf.where(tf.equal(flag, 1.0), 1.0, target)   # light(2) 는 목표 그대로
        else:
            flag = tf.constant(0.0)
        if training and AUG_CLEAR_GLASSES_PROB > 0:
            image = tf.cond(flag > 0, lambda: image, lambda: add_clear_glasses_tf(image, lm))
        img = tf.cast(image, tf.float32) / 255.0
        if training:
            img = augment(img)
        if training:
            return to_model_input(img), target[None], weight
        return to_model_input(img), target[None]

    return ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(idx["train"], training=True)
val_ds = make_dataset(idx["val"], training=False)
test_ds = make_dataset(idx["test"], training=False)
''')

code('''
# 13. 증강 미리보기
# - 위: 원본 얼굴 / 아래: 모델 입력 (증강 후 위쪽 영역)
# - 뜬 눈 4장, 감은 눈 4장. 각각 1번째는 합성 선글라스, 2번째는 합성 투명 안경을 강제로 적용
# - 증강이 너무 세거나 약하면 5번 셀 조정
sample = np.concatenate([idx["train"][Y[idx["train"]] == 1][:4], idx["train"][Y[idx["train"]] == 0][:4]])
fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
for col, i in enumerate(sample):
    image, extra = X[i], ""
    if col in (0, 4):
        image, _ = sunglasses_np(X[i], LM[i], Y[i], force=True, style="reflect")
        extra = " + sunglasses"
    elif col in (1, 5):
        image = clear_glasses_np(X[i], LM[i], force=True)
        extra = " + glasses"
    model_in = to_model_input(augment(tf.constant(image, tf.float32) / 255.0)).numpy() / 255.0
    axes[0, col].imshow(image)
    axes[1, col].imshow(np.clip(model_in, 0, 1))
    axes[0, col].set_title(("open" if Y[i] == 1 else "closed") + extra)
    for row in (0, 1):
        axes[row, col].axis("off")
plt.tight_layout()
plt.savefig(OUT / "augmentation_preview.png", dpi=100)
plt.show()
''')

code('''
# 14. 모델 만들기
# - 입력: upper_face_rgb_0_255 (IMG_SIZE x IMG_SIZE x 3, RGB 0~255, 얼굴 위쪽 EYE_TOP 을 늘린 이미지)
# - 백본 정규화는 모델 안에서 (앱은 0~255 그대로 넣음)
# - 백본의 BatchNorm 은 추론 모드로 고정 (작은 데이터에서 통계가 흔들리지 않게)
# - 출력: eye_open_score (sigmoid, 1 = 뜸)
def build_model():
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="upper_face_rgb_0_255")
    shape = (IMG_SIZE, IMG_SIZE, 3)
    if BACKBONE == "mobilenet_v3_small":
        base = keras.applications.MobileNetV3Small(input_shape=shape, alpha=ALPHA, include_top=False,
                                                    weights="imagenet", include_preprocessing=True)
        x = inputs
    elif BACKBONE == "mobilenet_v3_large":
        base = keras.applications.MobileNetV3Large(input_shape=shape, alpha=ALPHA, include_top=False,
                                                    weights="imagenet", include_preprocessing=True)
        x = inputs
    elif BACKBONE == "mobilenet_v2":
        base = keras.applications.MobileNetV2(input_shape=shape, alpha=ALPHA, include_top=False, weights="imagenet")
        x = keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)
    elif BACKBONE == "efficientnet_b0":
        base = keras.applications.EfficientNetB0(input_shape=shape, include_top=False, weights="imagenet")
        x = inputs
    else:
        raise ValueError(f"알 수 없는 BACKBONE: {BACKBONE}")

    base.trainable = True
    if TRAINABLE_BACKBONE_LAYERS is not None:
        for layer in base.layers[:-TRAINABLE_BACKBONE_LAYERS]:
            layer.trainable = False
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="eye_open_score")(x)
    return keras.Model(inputs, outputs, name=f"eye_{BACKBONE}")


model = build_model()
trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
print(f"{model.name}: 전체 파라미터 {model.count_params():,}, 학습 파라미터 {trainable:,}")
''')

code('''
# 15. 학습
# - AdamW + 워밍업 후 코사인 감소
# - USE_EMA 이면 epoch 끝마다 EMA 가중치로 바꿔 검증, 저장 (SwapEMAWeights 는 ModelCheckpoint 앞에)
# - 검증 AUC 기준으로 가장 좋은 모델 저장 (best.keras), 조기 종료
#   - 증류, 합성 선글라스 때문에 train auc, accuracy 는 소프트 목표로 계산되어 부정확. val_auc(정답 라벨)만 참고
# - 추론 모델 infer_model: TTA_FLIP 이면 원본과 좌우 뒤집은 입력의 점수 평균
steps = int(np.ceil(len(idx["train"]) / BATCH_SIZE))
schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LEARNING_RATE * 0.1,
    decay_steps=steps * max(1, EPOCHS - WARMUP_EPOCHS),
    alpha=0.01,
    warmup_target=LEARNING_RATE,
    warmup_steps=steps * WARMUP_EPOCHS,
)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=schedule, weight_decay=WEIGHT_DECAY,
                                     use_ema=USE_EMA, ema_momentum=EMA_MOMENTUM),
    loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="accuracy")],
)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=([keras.callbacks.SwapEMAWeights(swap_on_epoch=True)] if USE_EMA else []) + [
        keras.callbacks.ModelCheckpoint(str(OUT / "best.keras"), monitor="val_auc", mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=EARLY_STOP_PATIENCE,
                                      restore_best_weights=True),
        keras.callbacks.CSVLogger(str(OUT / "history.csv")),
    ],
)
model = keras.models.load_model(OUT / "best.keras")


def with_tta(net):
    """원본과 좌우 뒤집은 입력의 점수 평균 (입력 이름, 크기는 그대로)"""
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="upper_face_rgb_0_255")
    outputs = (net(inputs) + net(keras.ops.flip(inputs, axis=2))) / 2
    return keras.Model(inputs, outputs, name=f"{net.name}_tta")


infer_model = with_tta(model) if TTA_FLIP else model
''')

code('''
# 16. 학습 곡선
# - 손실, 검증 AUC. 검증 손실이 계속 오르면 과적합 (증강 강화, 드롭아웃, 학습률 조정)
hist = pd.read_csv(OUT / "history.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(hist["loss"], label="train")
axes[0].plot(hist["val_loss"], label="val")
axes[0].set_title("loss")
axes[1].plot(hist["auc"], label="train")
axes[1].plot(hist["val_auc"], label="val")
axes[1].set_title("AUC")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.legend()
plt.tight_layout()
plt.savefig(OUT / "learning_curve.png", dpi=100)
plt.show()
print(f"최고 val AUC {hist['val_auc'].max():.4f} (epoch {int(hist['val_auc'].idxmax()) + 1})")
''')

code('''
# 17. 임계값 결정 (검증 데이터만 사용)
# - 감은 눈 recall >= TARGET_CLOSED_RECALL 인 임계값 중 눈 뜸 recall 최대
#   - 점수 >= 임계값이면 뜸. 감은 눈 recall = 감은 눈 중 점수 < 임계값 비율 = 1 - 감은 눈의 뜸 판정 비율(fpr)
# - 반환: (임계값, 감은 눈 recall, 눈 뜸 recall)
def pick_threshold(y, p):
    fpr, tpr, thresholds = roc_curve(y, p)
    ok = np.flatnonzero(1 - fpr >= TARGET_CLOSED_RECALL)
    i = ok[np.argmax(tpr[ok])]   # fpr = 0 인 점이 항상 있어 비지 않음
    return float(min(thresholds[i], 1.0)), float(1 - fpr[i]), float(tpr[i])


def scores(ds):
    return infer_model.predict(ds, verbose=0).ravel()


p_val = scores(val_ds)
THRESHOLD, val_closed_rec, val_open_rec = pick_threshold(Y[idx["val"]], p_val)
print(f"val AUC {roc_auc_score(Y[idx['val']], p_val):.4f}")
print(f"임계값 {THRESHOLD:.4f}: 감은 눈 recall {val_closed_rec:.3f}, 눈 뜸 recall {val_open_rec:.3f}")
''')

code('''
# 18. 시험 평가 (결과를 보고 설정을 다시 고르면 test 가 검증 데이터가 됨)
# - AUC, 임계값에서의 눈 뜸 precision / recall, 감은 눈 recall, 혼동 행렬 (뜸/감음 = 단계 1, 2 / 3, 4)
# - 단계 확인 (val + test): 단계별 점수 중앙값, 임계값 미만(감음 판정) 비율, 이웃 단계 순서 AUC (1>2, 2>3, 3>4. 1 이면 순서 완벽)
# - 선글라스 확인
#   - val, test 에 합성 선글라스를 씌웠을 때 "뜸" 판정 비율 (뜬 눈, 감은 눈 따로. 둘 다 높아야 함)
#   - 실제 선글라스 (REAL_SUNGLASSES 중 val, test 에 있는 것) 의 "뜸" 판정 비율
# - 투명 안경 확인: val, test 에 합성 투명 안경을 씌웠을 때 감은 눈 recall, 눈 뜸 recall (안경이 없을 때와 비슷해야 함)
# - 확신하며 틀린 이미지 12장 (라벨 오류, 약점 확인용)
def report(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)), "threshold": t,
            "open_precision": tp / max(tp + fp, 1), "open_recall": tp / max(tp + fn, 1),
            "closed_recall": tn / max(tn + fp, 1),
            "accuracy": (tp + tn) / len(y), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


p_test = scores(test_ds)
metrics = {"val": report(Y[idx["val"]], p_val, THRESHOLD), "test": report(Y[idx["test"]], p_test, THRESHOLD)}
display(pd.DataFrame(metrics).T)

ev = np.concatenate([idx["val"], idx["test"]])
p_ev = np.concatenate([p_val, p_test])
level_score = {}
for lv in sorted(LEVEL_TARGETS):
    s_lv = p_ev[LEVEL[ev] == lv]
    level_score[str(lv)] = {"n": int(len(s_lv)), "median": float(np.median(s_lv)) if len(s_lv) else None,
                            "below_threshold": float(np.mean(s_lv < THRESHOLD)) if len(s_lv) else None}
order_auc = {}
for lv in sorted(LEVEL_TARGETS)[:-1]:
    hi, lo = p_ev[LEVEL[ev] == lv], p_ev[LEVEL[ev] == lv + 1]
    if len(hi) and len(lo):
        order_auc[f"{lv}>{lv + 1}"] = float(roc_auc_score(np.r_[np.ones(len(hi)), np.zeros(len(lo))], np.r_[hi, lo]))
display(pd.DataFrame(level_score).T.rename(index={"1": "1 뜸", "2": "2 웃는 눈", "3": "3 감기는 눈", "4": "4 감은 눈"}))
print("이웃 단계 순서 AUC:", {k: round(v, 3) for k, v in order_auc.items()})
metrics["levels"] = {"score": level_score, "order_auc": order_auc}

eval_idx = np.concatenate([idx["val"], idx["test"]])
eval_idx = eval_idx[~np.isnan(LM[eval_idx]).any(1)]
np.random.seed(SEED)
sunglasses = {}
for style in ("dark", "reflect", "mirror"):   # light 는 눈이 비쳐 목표가 그대로라 제외
    worn = np.stack([sunglasses_np(X[i], LM[i], Y[i], force=True, style=style)[0] for i in eval_idx])
    worn_in = np.stack([to_model_input(tf.constant(im, tf.float32) / 255.0).numpy() for im in worn])
    p_worn = infer_model.predict(worn_in, verbose=0).ravel()
    sunglasses[f"synthetic_{style}_open_as_open"] = float(np.mean(p_worn[Y[eval_idx] == 1] >= THRESHOLD))
    sunglasses[f"synthetic_{style}_closed_as_open"] = float(np.mean(p_worn[Y[eval_idx] == 0] >= THRESHOLD))
# 실제 선글라스: 눈 좌표 유무와 상관없이 val, test 전체에서
real = [i for i in np.concatenate([idx["val"], idx["test"]]) if df.folder[i] == "opened"
        and (df.file[i].rsplit(".", 1)[0] in REAL_SUNGLASSES or df.file[i].startswith(REAL_SUNGLASSES_PREFIX))]
if real:
    real_in = np.stack([to_model_input(tf.constant(X[i], tf.float32) / 255.0).numpy() for i in real])
    p_real = infer_model.predict(real_in, verbose=0).ravel()
    is_cofw = np.array([df.file[i].startswith(REAL_SUNGLASSES_PREFIX) for i in real])
    sunglasses["real_as_open"] = float(np.mean(p_real >= THRESHOLD))
    sunglasses["real_count"] = len(real)
    if is_cofw.any():
        sunglasses["real_cofw_as_open"] = float(np.mean(p_real[is_cofw] >= THRESHOLD))
        sunglasses["real_cofw_count"] = int(is_cofw.sum())
print("선글라스 뜸 판정 비율:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sunglasses.items()})
metrics["sunglasses"] = sunglasses

glasses = np.stack([clear_glasses_np(X[i], LM[i], force=True) for i in eval_idx])
glasses_in = np.stack([to_model_input(tf.constant(im, tf.float32) / 255.0).numpy() for im in glasses])
p_glasses = infer_model.predict(glasses_in, verbose=0).ravel()
clear = {
    "closed_recall": float(np.mean(p_glasses[Y[eval_idx] == 0] < THRESHOLD)),
    "open_recall": float(np.mean(p_glasses[Y[eval_idx] == 1] >= THRESHOLD)),
}
print("투명 안경을 씌웠을 때:", {k: round(v, 3) for k, v in clear.items()})
metrics["clear_glasses"] = clear

wrong = np.flatnonzero((p_test >= THRESHOLD).astype(int) != Y[idx["test"]])
wrong = wrong[np.argsort(-np.abs(p_test[wrong] - THRESHOLD))][:12]
if len(wrong):
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, k in zip(axes.flat, wrong):
        ax.imshow(X[idx["test"][k]])
        ax.set_title(f"label {'open' if Y[idx['test'][k]] else 'closed'} / {p_test[k]:.2f}")
    plt.tight_layout()
    plt.savefig(OUT / "test_errors.png", dpi=100)
    plt.show()
''')

code('''
# 19. tflite 변환과 검증
# - 추론 모델(infer_model, TTA 포함 여부는 TTA_FLIP)을 SavedModel 로 내보낸 뒤 변환: fp32 (항상), dynamic, int8 (설정)
# - 변환 모델로 val 을 다시 판정해 Keras 결과와 비교 (점수 차이, AUC)
# - int8 은 입력/출력도 int8. dynamic, int8 은 실패하면 건너뜀
#   - MobileNetV3 int8 은 XNNPACK(CPU 가속기)에서 준비 실패할 수 있음 -> XNNPACK 을 끄고 다시 검증, 결과에 기록
#   - 앱에서도 같은 옵션(BUILTIN_WITHOUT_DEFAULT_DELEGATES)으로 열어야 함
from ai_edge_litert.interpreter import Interpreter, OpResolverType

saved_dir = OUT / "saved_model"
infer_model.export(str(saved_dir))


def convert(kind):
    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
    if kind == "dynamic":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if kind == "int8":
        rng = np.random.default_rng(SEED)
        rep = np.concatenate([
            rng.choice(pool, min(REPRESENTATIVE_SAMPLES // 2, len(pool)), replace=False)
            for pool in (idx["train"][Y[idx["train"]] == c] for c in (0, 1))])
        rng.shuffle(rep)

        def representative():
            for i in rep:
                yield [to_model_input(tf.constant(X[i], tf.float32) / 255.0)[None].numpy()]

        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


def tflite_scores(path, indices, xnnpack=True):
    resolver = OpResolverType.AUTO if xnnpack else OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    it = Interpreter(model_path=str(path), experimental_op_resolver_type=resolver)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    result = []
    for i in indices:
        x = to_model_input(tf.constant(X[i], tf.float32) / 255.0)[None].numpy()
        if np.issubdtype(inp["dtype"], np.integer):
            scale, zero = inp["quantization"]
            x = np.clip(np.rint(x / scale + zero), -128, 127)
        it.set_tensor(inp["index"], x.astype(inp["dtype"]))
        it.invoke()
        y = it.get_tensor(out["index"]).astype(np.float32)
        if np.issubdtype(out["dtype"], np.integer):
            scale, zero = out["quantization"]
            y = (y - zero) * scale
        result.append(float(y.ravel()[0]))
    return np.array(result)


kinds = ["fp32"] + (["dynamic"] if EXPORT_DYNAMIC else []) + (["int8"] if EXPORT_INT8 else [])
exported = {}
for kind in kinds:
    path = OUT / f"eye_{kind}.tflite"
    try:
        path.write_bytes(convert(kind))
        xnnpack = True
        try:
            p = tflite_scores(path, idx["val"])
        except RuntimeError as exc:
            print(f"{kind}: XNNPACK 준비 실패, XNNPACK 없이 다시 검증 ({exc})")
            xnnpack = False
            p = tflite_scores(path, idx["val"], xnnpack=False)
    except Exception as exc:
        if kind == "fp32":
            raise
        print(f"{kind} 변환 또는 검증 실패, 건너뜀:", repr(exc))
        path.unlink(missing_ok=True)
        continue
    exported[kind] = {"file": path.name, "size_kib": round(path.stat().st_size / 1024, 1), "xnnpack": xnnpack,
                      "val_auc": float(roc_auc_score(Y[idx["val"]], p)),
                      "max_score_diff": float(np.max(np.abs(p - p_val)))}
display(pd.DataFrame(exported).T)
''')

code('''
# 20. 결과 저장
# - inference_config.json: 앱이 따라야 할 전처리와 출력 의미, 임계값
# - metrics.json: val / test 지표, 선글라스 확인, tflite 검증 결과
# - 그 밖에 best.keras, *.tflite, history.csv, split_manifest.csv, 그림
config = {
    "model_files": {k: v["file"] for k, v in exported.items()},
    "xnnpack": {k: v["xnnpack"] for k, v in exported.items()},
    "input": {"name": "upper_face_rgb_0_255", "size": IMG_SIZE, "color": "RGB", "range": [0, 255],
              "resize": "bilinear",
              "crop": f"YuNet 박스 중심 정사각형(한 변 = 박스 높이)의 위쪽 {EYE_TOP} 을 {IMG_SIZE}x{IMG_SIZE} 로 늘림"},
    "output": {"meaning": "눈 뜸 점수, 1 = 뜸 (선글라스류는 뜸)", "tta_flip": TTA_FLIP},
    "distill_alpha": DISTILL_ALPHA,
    "level_targets": {str(k): v for k, v in LEVEL_TARGETS.items()},
    "threshold": THRESHOLD,
    "threshold_rule": f"val 에서 감은 눈 recall >= {TARGET_CLOSED_RECALL} 인 값 중 눈 뜸 recall 최대",
    "train_source_size": SRC_SIZE,
}
settings = {k: v for k, v in globals().items()
            if k.isupper() and isinstance(v, (int, float, str, bool, type(None), dict, list))
            and k not in {"THRESHOLD", "SRC_SIZE", "UPPER_H", "LENS_SHAPES", "CLEAR_FRAME_COLORS"}}
(OUT / "inference_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
(OUT / "metrics.json").write_text(json.dumps({"metrics": metrics, "tflite": exported, "settings": settings},
                                             ensure_ascii=False, indent=2, default=str))
print("저장:", OUT)
for p in sorted(OUT.iterdir()):
    print(" -", p.name)
''')


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": [],
}
for i, (kind, src) in enumerate(CELLS):
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": source}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    nb["cells"].append(cell)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("written", OUT, len(CELLS), "cells")
