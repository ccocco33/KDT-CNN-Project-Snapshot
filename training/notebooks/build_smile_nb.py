"""smile 학습 노트북 생성기 (셀 내용 -> .ipynb)
- 합성 선글라스 함수는 training/data/eye/augment_review/eyewear_fn.py 에서 가져옴 (SUNGLASSES_SRC. eye 노트북과 같은 함수)
- 실행: python training/notebooks/build_smile_nb.py training/notebooks/smile_train.ipynb training/data/eye/augment_review/eyewear_fn.py
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
# smile 분류 모델 학습

- 입력: YuNet 으로 자른 얼굴 (전처리 데이터 zip `DRIVE_FACE/smile/preprocessed.zip`, 64x64 PNG)
  - 모델에는 얼굴 정사각형의 **아래쪽 `SMILE_BOTTOM` 비율**을 `IMG_SIZE` x `IMG_SIZE` 로 늘려 넣음 (1.0 이면 얼굴 전체)
- 출력: 웃음 점수 (0~1, **1 = 웃음**)
- 모델: ImageNet 사전학습 백본 + 전역 평균 풀링 + 이진 분류. 백본까지 한 번에 학습
- 지식 증류: 교사(FaceLandmarker blendshape 웃음 점수)를 웃음 확률로 바꿔 정답 라벨과 섞은 목표로 학습 (`DISTILL_ALPHA`)
- 합성 선글라스 증강: 선글라스를 쓴 무표정을 웃음으로 보는 문제 대응. 선글라스는 웃음과 무관 -> 목표 그대로 (`AUG_SUNGLASSES_PROB`)
- 결과: tflite (fp32, dynamic range, int8), 추론 설정 json, 지표 (`OUTPUT_ROOT/RUN_NAME`)
- 실행: Colab GPU 런타임에서 위에서 아래로. 설정은 1~6번 셀에서만 바꿈
''')

code('''
# 1. 경로 설정
# - DRIVE_FACE: 데이터셋 루트 (Google Drive)
# - DATA_ZIP: 전처리한 smile 데이터 zip. 안에 CLASS_DIRS 의 폴더 (한 겹 상위 폴더가 있어도 됨)
# - LOCAL_DATA_DIR: zip 을 풀 로컬 경로 (Drive 에서 직접 읽으면 느림). 이미 풀려 있으면 다시 풀지 않음
# - TEACHER_CSV: 지식 증류용 교사 점수 (folder, file, label, teacher_score). DISTILL_ALPHA 가 1.0 이면 안 씀
# - LANDMARKS_CSV: 합성 선글라스용 두 눈 좌표 (folder, file, rx, ry, lx, ly. 64px 기준). AUG_SUNGLASSES_PROB 가 0 이면 안 씀
# - OUTPUT_ROOT: 학습 결과 저장 위치. 실행마다 RUN_NAME 하위 폴더 생성
# - RUN_NAME: None 이면 실행 시각 (예: 20260926-1530)
DRIVE_FACE = "/content/drive/MyDrive/Team-11/dataset"
DATA_ZIP = f"{DRIVE_FACE}/smile/preprocessed.zip"
LOCAL_DATA_DIR = "/content/data/smile"
TEACHER_CSV = f"{DRIVE_FACE}/smile/teacher_scores.csv"
LANDMARKS_CSV = f"{DRIVE_FACE}/smile/eye_landmarks.csv"
OUTPUT_ROOT = "/content/drive/MyDrive/Team-11/models/smile"
RUN_NAME = None
''')

code('''
# 2. 데이터 설정
# - CLASS_DIRS: 폴더 이름 -> (원래 분할, 라벨). 라벨 1 = 웃음, 0 = 안 웃음
# - VAL_FRACTION: train 에서 떼어 낼 검증 비율
# - DHASH_DISTANCE: 비슷한 이미지로 묶는 dHash 거리 (이하면 같은 묶음. 묶음은 분할을 넘지 않음)
# - SEED: 난수 시드
CLASS_DIRS = {
    "smile_train": ("train", 1),
    "no_smile_train": ("train", 0),
    "smile_test": ("test", 1),
    "no_smile_test": ("test", 0),
}
VAL_FRACTION = 0.2
DHASH_DISTANCE = 4
SEED = 42
''')

code('''
# 3. 모델 설정
# - BACKBONE: mobilenet_v3_small | mobilenet_v3_large | mobilenet_v2 | efficientnet_b0
# - ALPHA: MobileNet 폭 배율 (efficientnet_b0 는 무시)
# - IMG_SIZE: 모델 입력 크기 (정사각형)
# - SMILE_BOTTOM: 얼굴 정사각형에서 쓸 아래쪽 비율 (코끝, 입, 볼). 1.0 이면 얼굴 전체
#   - 눈 중심 높이 약 0.37 (99% 가 0.42 이내), 선글라스 렌즈 아래 끝 약 0.52 -> 0.45 면 렌즈 제외
#   - 아래쪽 영역을 IMG_SIZE x IMG_SIZE 로 늘림 (가로세로 비율 유지 안 함)
# - DROPOUT: 분류 직전 드롭아웃 비율
# - TRAINABLE_BACKBONE_LAYERS: None 이면 백본 전체 학습, 정수 N 이면 백본 마지막 N 개 층만 학습
BACKBONE = "mobilenet_v3_small"
ALPHA = 1.0
IMG_SIZE = 128
SMILE_BOTTOM = 0.45
DROPOUT = 0.3
TRAINABLE_BACKBONE_LAYERS = None
''')

code('''
# 4. 학습 설정
# - LEARNING_RATE: 최대 학습률. WARMUP_EPOCHS 동안 1/10 에서 올라간 뒤 코사인으로 감소
# - WEIGHT_DECAY: AdamW 가중치 감쇠
# - LABEL_SMOOTHING: 라벨 스무딩 (0 이면 끔)
# - EARLY_STOP_PATIENCE: 검증 AUC 가 이 epoch 수 동안 오르지 않으면 중단 (가장 좋은 가중치로 복원)
# - USE_CLASS_WEIGHT: 클래스 수 차이 보정
# - USE_EMA: 가중치 이동 평균(EMA)을 최종 모델로. 검증, 체크포인트도 EMA 가중치로 (과적합 완화)
# - EMA_MOMENTUM: EMA 비율. 클수록 과거 가중치를 오래 반영 (0.99: 약 100 step)
# - DISTILL_ALPHA: 지식 증류에서 정답 라벨 비중 a. 학습 목표 = a x 정답 + (1 - a) x 교사 확률. 1.0 이면 증류 끔
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WARMUP_EPOCHS = 2
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 8
USE_CLASS_WEIGHT = True
USE_EMA = True
EMA_MOMENTUM = 0.99
DISTILL_ALPHA = 0.5
''')

code('''
# 5. 증강 설정 (학습 데이터에만 적용. 확률 0 또는 범위 0 이면 끔)
# - 실제 카메라에서 생기는 변화만 흔듦. 흑백은 제외 (실제 카메라는 컬러)
# - 합성 선글라스: 확률 AUG_SUNGLASSES_PROB 로 두 눈 위에 선글라스. 목표는 그대로 (웃음과 무관)
#   - 렌즈 종류는 eyewear_fn.SUNGLASS_STYLES 확률로 (dark, reflect, mirror, light). SUNGLASSES_MIN_ALPHA: dark 렌즈 불투명도 하한
AUG_SUNGLASSES_PROB = 0.10
SUNGLASSES_MIN_ALPHA = 0.70
# - 기하: 좌우 뒤집기, 회전(도), 확대/축소(비율 ±), 이동(비율 ±). 빈 곳은 반사로 채움
AUG_FLIP = True
AUG_ROTATION_DEG = 20
AUG_ZOOM = 0.10
AUG_TRANSLATE = 0.08
# - 해상도: 확률 AUG_DOWNSCALE_PROB 로 [MIN, MAX] px 로 줄였다 원래 크기로 키움 (멀리 있는 작은 얼굴)
AUG_DOWNSCALE_PROB = 0.5
AUG_DOWNSCALE_MIN = 24
AUG_DOWNSCALE_MAX = 56
# - 흐림: 확률 AUG_BLUR_PROB 로 가우시안 흐림, 시그마 최대 AUG_BLUR_SIGMA_MAX (px, 64px 원본 기준)
AUG_BLUR_PROB = 0.2
AUG_BLUR_SIGMA_MAX = 1.2
# - 색: 밝기(± 0~1 척도), 대비(배율 ±), 채도(배율 ±)
AUG_BRIGHTNESS = 0.15
AUG_CONTRAST = 0.2
AUG_SATURATION = 0.3
# - 잡음: 확률 AUG_NOISE_PROB 로 가우시안 잡음, 표준편차 최대 AUG_NOISE_STD_MAX (0~1 척도). 어두운 실내 촬영이면 켬
AUG_NOISE_PROB = 0.0
AUG_NOISE_STD_MAX = 0.03
''')

code('''
# 6. 평가, 내보내기 설정
# - TARGET_PRECISION: 웃음 판정 precision 목표. 검증 데이터에서 이 값을 만족하는 임계값 중 recall 최대
#   - 앱은 조건 충족 사진의 precision 을 우선하므로 "웃음" 판정을 보수적으로
# - tflite 변환 (fp32 는 항상)
#   - EXPORT_DYNAMIC: dynamic range 양자화. 가중치 int8, 입출력 float32. 대표 데이터 불필요
#   - EXPORT_INT8: 전체 int8 양자화. 가중치, 활성값, 입출력 int8
# - TTA_FLIP: 원본과 좌우 뒤집은 얼굴의 점수 평균을 모델 안에 넣어 내보냄
#   - 앱 코드 변경 없음, 추론 비용 2배. 임계값과 평가도 이 모델로
# - REPRESENTATIVE_SAMPLES: int8 양자화에서 활성값 범위를 잴 학습 이미지 수 (웃음, 안 웃음 반씩)
#   - 값 범위 측정용이라 데이터셋 크기와 무관하게 수백 장이면 충분
TARGET_PRECISION = 0.95
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score

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
# - CLASS_DIRS 의 폴더에서 이미지를 모두 읽음 (RGB, 0~255)
# - 이미지마다 dHash 계산 (비슷한 이미지 묶기용)
# - 모든 이미지는 같은 크기여야 함 (전처리 결과 64x64)
# - DISTILL_ALPHA < 1 이면 교사 점수(TEACHER_CSV)를 폴더, 파일 이름으로 연결 (없는 이미지는 빈 값)
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
for folder, (split, label) in CLASS_DIRS.items():
    paths = sorted(p for p in (DATA_DIR / folder).iterdir()
                   if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._"))
    for p in paths:
        with Image.open(p) as src:
            img = src.convert("RGB")
        rows.append({"path": str(p), "folder": folder, "file": p.name, "orig_split": split, "label": label,
                     "dhash": dhash(img)})
        images.append(np.asarray(img))

df = pd.DataFrame(rows)
if DISTILL_ALPHA < 1:
    teacher = pd.read_csv(TEACHER_CSV)[["folder", "file", "teacher_score"]]
    df = df.merge(teacher, on=["folder", "file"], how="left")
    print(f"교사 점수 연결: {df.teacher_score.notna().sum()} / {len(df)}")
else:
    df["teacher_score"] = np.nan
if AUG_SUNGLASSES_PROB > 0:
    lm = pd.read_csv(LANDMARKS_CSV)[["folder", "file", "rx", "ry", "lx", "ly"]]
    df = df.merge(lm, on=["folder", "file"], how="left")
else:
    df[["rx", "ry", "lx", "ly"]] = np.nan
LM = df[["rx", "ry", "lx", "ly"]].to_numpy(np.float32)   # 두 눈 좌표 (없으면 NaN)
print(f"눈 좌표 없음: {int(np.isnan(LM).any(1).sum())} / {len(df)}")
sizes = {im.shape for im in images}
assert len(sizes) == 1, f"이미지 크기가 섞여 있음: {sizes}"
X = np.stack(images)
SRC_SIZE = X.shape[1]
print("이미지:", X.shape)
display(df.groupby(["orig_split", "label"]).size().rename("장수"))
''')

code('''
# 9. 중복 정리와 분할
# - dHash 거리 DHASH_DISTANCE 이하인 이미지를 한 묶음으로 (정확한 중복 포함)
# - test 와 같은 묶음인 train 이미지는 제외 (시험 누수 방지)
# - train 을 묶음 단위로 train / val 로 나눔 (라벨별 VAL_FRACTION 에 가깝게)
# - 결과: df["split"] = train | val | test | excluded
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

mixed = df.groupby("group")["label"].nunique()
print("라벨이 섞인 묶음 (라벨 확인 필요):", int((mixed > 1).sum()))

test_groups = set(df.loc[df.orig_split == "test", "group"])
df["split"] = np.where(df.orig_split == "test", "test", "train")
df.loc[(df.orig_split == "train") & df.group.isin(test_groups), "split"] = "excluded"

rng = np.random.default_rng(SEED)
train_pool = df[df.split == "train"]
group_label = train_pool.groupby("group")["label"].agg(lambda s: int(round(s.mean())))
group_size = train_pool.groupby("group").size()
for label in (0, 1):
    groups = group_label.index[group_label == label].to_numpy().copy()
    rng.shuffle(groups)
    target = VAL_FRACTION * (train_pool.label == label).sum()
    taken = 0
    for g in groups:
        if taken >= target:
            break
        df.loc[df.group == g, "split"] = "val"
        taken += group_size[g]

for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    assert not set(df.loc[df.split == a, "group"]) & set(df.loc[df.split == b, "group"]), f"{a}, {b} 묶음 겹침"
df.to_csv(OUT / "split_manifest.csv", index=False)
display(df.groupby(["split", "label"]).size().unstack().rename(columns={0: "안 웃음", 1: "웃음"}))
idx = {s: df.index[df.split == s].to_numpy() for s in ("train", "val", "test")}
Y = df["label"].to_numpy(np.float32)
''')

code('''
# 10. 교사 확률 (지식 증류)
# - 교사 점수 s (입꼬리 점수 0~1) 는 확률이 아니므로 웃음 확률 p 로 보정: p = sigmoid(w x s + b)
#   - w, b 는 train 분할만으로 맞춤 (로지스틱 회귀). val, test 는 보정에 쓰지 않음
# - 학습 목표 T = DISTILL_ALPHA x 정답 + (1 - DISTILL_ALPHA) x p. 교사 미검출 이미지는 정답 그대로
# - 검증, 시험은 정답 라벨(Y)로만 평가
T = Y.copy()
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
    T[train_has] = DISTILL_ALPHA * Y[train_has] + (1 - DISTILL_ALPHA) * p_teacher[train_has]
    print(f"보정: p = sigmoid({w:.2f} x s + {b:.2f}), p = 0.5 인 s = {-b / w:.3f}")
    print("교사 점수 s -> 웃음 확률 p:", ", ".join(f"{v:.1f}->{1 / (1 + np.exp(-(w * v + b))):.2f}" for v in (0, 0.1, 0.2, 0.3, 0.5)))
    val_has = idx["val"][has[idx["val"]]]
    print(f"교사 AUC (val, 참고): {roc_auc_score(Y[val_has], s_all[val_has]):.4f}")
    print(f"학습 목표: 교사 반영 {len(train_has)}장, 정답만 {len(idx['train']) - len(train_has)}장")
    fig, ax = plt.subplots(figsize=(5, 3))
    for c, name in ((1, "smile"), (0, "no smile")):
        ax.hist(T[idx["train"]][Y[idx["train"]] == c], bins=40, alpha=0.6, label=name)
    ax.set_title("train target distribution")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("증류 끔 (DISTILL_ALPHA = 1.0): 정답 라벨로만 학습")
''')

code('''
# 10-1. 합성 선글라스 (증강)
# - add_sunglasses: 두 눈 위에 렌즈와 테. 종류 dark, reflect (밝은 반사), mirror (색 미러), light (옅은 렌즈)
# - 목표는 바꾸지 않음 (선글라스는 웃음과 무관)
# - eye 노트북과 같은 함수 (training/data/eye/augment_review/eyewear_fn.py)
import math

import cv2

''' + SUNGLASSES_SRC + '''


def sunglasses_np(image, lm, force=False, style=None):
    """학습 증강용 (numpy). AUG_SUNGLASSES_PROB 확률 (force 면 항상). 눈 좌표가 없으면 적용 안 함
    - style: 렌즈 종류 고정 (None 이면 SUNGLASS_STYLES 확률로)
    """
    rng = np.random.default_rng()
    if np.isnan(lm).any() or (not force and rng.random() >= AUG_SUNGLASSES_PROB):
        return image
    style = style or pick_sunglass_style(rng)
    out = add_sunglasses(np.ascontiguousarray(image), (lm[0], lm[1]), (lm[2], lm[3]), rng,
                         min_alpha=SUNGLASSES_MIN_ALPHA, style=style)
    return out.astype(np.uint8)


def add_sunglasses_tf(image, lm):
    out = tf.numpy_function(sunglasses_np, [image, lm], tf.uint8)
    out.set_shape(image.shape)
    return out
''')

code('''
# 11. 입력 파이프라인
# - 공통 (학습, 검증, 시험, 앱): 0~1 로 바꿔 처리 -> 아래쪽 SMILE_BOTTOM 영역 -> IMG_SIZE x IMG_SIZE 로 bilinear 늘림 -> 0~255
#   - 모델 입력은 RGB 0~255. 정규화는 모델 안에서
# - 학습만: 합성 선글라스 (10-1번 셀), 5번 셀의 증강 (이미지마다 무작위), 목표는 10번 셀의 T (증류 반영). 검증, 시험은 정답 Y
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


LOWER_H = int(round(SRC_SIZE * SMILE_BOTTOM))


def to_model_input(img):
    """0~1 얼굴 이미지 (정사각형) -> 모델 입력 (아래쪽 SMILE_BOTTOM, IMG_SIZE x IMG_SIZE, bilinear, 0~255)"""
    lower = img[SRC_SIZE - LOWER_H:]
    return tf.image.resize(lower, [IMG_SIZE, IMG_SIZE], method="bilinear") * 255.0


def make_dataset(indices, training):
    targets = T if training else Y
    ds = tf.data.Dataset.from_tensor_slices((X[indices], targets[indices].astype(np.float32), LM[indices]))
    if training:
        ds = ds.shuffle(len(indices), seed=SEED, reshuffle_each_iteration=True)

    def prep(image, label, lm):
        if training and AUG_SUNGLASSES_PROB > 0:
            image = add_sunglasses_tf(image, lm)
        img = tf.cast(image, tf.float32) / 255.0
        if training:
            img = augment(img)
        return to_model_input(img), label[None]

    return ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(idx["train"], training=True)
val_ds = make_dataset(idx["val"], training=False)
test_ds = make_dataset(idx["test"], training=False)
''')

code('''
# 12. 증강 미리보기
# - 위: 원본 / 아래: 증강 결과 (같은 이미지)
# - 증강이 너무 세거나 약하면 5번 셀 조정
sample = idx["train"][:8]
fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
for col, i in enumerate(sample):
    original = X[i].astype(np.float32) / 255.0
    augmented = augment(tf.constant(original)).numpy()
    for row, im in enumerate((original, augmented)):
        axes[row, col].imshow(im)
        axes[row, col].axis("off")
    axes[0, col].set_title("smile" if Y[i] == 1 else "no smile")
plt.tight_layout()
plt.savefig(OUT / "augmentation_preview.png", dpi=100)
plt.show()
''')

code('''
# 13. 모델 만들기
# - 입력: face_rgb_0_255 (IMG_SIZE x IMG_SIZE x 3, RGB 0~255)
# - 백본 정규화는 모델 안에서 (앱은 0~255 그대로 넣음)
# - 백본의 BatchNorm 은 추론 모드로 고정 (작은 데이터에서 통계가 흔들리지 않게)
# - 출력: smile_score (sigmoid, 1 = 웃음)
def build_model():
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="face_rgb_0_255")
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
    outputs = keras.layers.Dense(1, activation="sigmoid", name="smile_score")(x)
    return keras.Model(inputs, outputs, name=f"smile_{BACKBONE}")


model = build_model()
trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
print(f"{model.name}: 전체 파라미터 {model.count_params():,}, 학습 파라미터 {trainable:,}")
''')

code('''
# 14. 학습
# - AdamW + 워밍업 후 코사인 감소
# - USE_EMA 이면 epoch 끝마다 EMA 가중치로 바꿔 검증, 저장 (SwapEMAWeights 는 ModelCheckpoint 앞에)
# - 검증 AUC 기준으로 가장 좋은 모델 저장 (best.keras), 조기 종료
#   - 증류 중에는 train auc 가 소프트 목표로 계산되어 부정확. val_auc(정답 라벨)만 참고
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
class_weight = None
if USE_CLASS_WEIGHT:
    counts = np.bincount(Y[idx["train"]].astype(int), minlength=2)
    class_weight = {c: len(idx["train"]) / (2 * counts[c]) for c in (0, 1)}
    print("클래스 가중치:", class_weight)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
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
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="face_rgb_0_255")
    outputs = (net(inputs) + net(keras.ops.flip(inputs, axis=2))) / 2
    return keras.Model(inputs, outputs, name=f"{net.name}_tta")


infer_model = with_tta(model) if TTA_FLIP else model
''')

code('''
# 15. 학습 곡선
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
# 16. 임계값 결정 (검증 데이터만 사용)
# - 웃음 precision >= TARGET_PRECISION 인 임계값 중 recall 최대
# - 만족하는 값이 없으면 precision 최대인 임계값
def pick_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    ok = np.flatnonzero(precision >= TARGET_PRECISION)
    i = ok[np.argmax(recall[ok])] if len(ok) else int(np.argmax(precision))
    return float(thresholds[i]), float(precision[i]), float(recall[i])


def scores(ds):
    return infer_model.predict(ds, verbose=0).ravel()


p_val = scores(val_ds)
THRESHOLD, val_prec, val_rec = pick_threshold(Y[idx["val"]], p_val)
print(f"val AUC {roc_auc_score(Y[idx['val']], p_val):.4f}")
print(f"임계값 {THRESHOLD:.4f}: 웃음 precision {val_prec:.3f}, recall {val_rec:.3f}")
''')

code('''
# 17. 시험 평가 (결과를 보고 설정을 다시 고르면 test 가 검증 데이터가 됨)
# - AUC, 임계값에서의 웃음 precision / recall, 혼동 행렬
# - 확신하며 틀린 이미지 12장 (라벨 오류, 약점 확인용)
def report(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)), "threshold": t,
            "smile_precision": tp / max(tp + fp, 1), "smile_recall": tp / max(tp + fn, 1),
            "accuracy": (tp + tn) / len(y), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


p_test = scores(test_ds)
metrics = {"val": report(Y[idx["val"]], p_val, THRESHOLD), "test": report(Y[idx["test"]], p_test, THRESHOLD)}
display(pd.DataFrame(metrics).T)

# 선글라스 확인: test 얼굴에 합성 선글라스를 씌워도 같은 판정인지 (렌즈 종류별)
# - no_smile_as_smile: 안 웃은 얼굴을 웃음으로 본 비율 (씌우기 전 값과 비교). smile_recall: 웃은 얼굴을 웃음으로 본 비율
has_lm = idx["test"][~np.isnan(LM[idx["test"]]).any(1)]
y_lm = Y[has_lm]
p_plain = infer_model.predict(np.stack([to_model_input(tf.constant(X[i], tf.float32) / 255.0).numpy() for i in has_lm]),
                              verbose=0).ravel()
sunglasses = {"plain": {"no_smile_as_smile": float(np.mean(p_plain[y_lm == 0] >= THRESHOLD)),
                        "smile_recall": float(np.mean(p_plain[y_lm == 1] >= THRESHOLD))}}
for style in ("dark", "reflect", "mirror", "light"):
    worn = np.stack([to_model_input(tf.constant(sunglasses_np(X[i], LM[i], force=True, style=style), tf.float32) / 255.0).numpy()
                     for i in has_lm])
    p_worn = infer_model.predict(worn, verbose=0).ravel()
    sunglasses[style] = {"no_smile_as_smile": float(np.mean(p_worn[y_lm == 0] >= THRESHOLD)),
                         "smile_recall": float(np.mean(p_worn[y_lm == 1] >= THRESHOLD))}
print("합성 선글라스를 씌운 test (렌즈 종류별)")
display(pd.DataFrame(sunglasses).T)
metrics["sunglasses"] = sunglasses

wrong = np.flatnonzero((p_test >= THRESHOLD).astype(int) != Y[idx["test"]])
wrong = wrong[np.argsort(-np.abs(p_test[wrong] - THRESHOLD))][:12]
if len(wrong):
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, k in zip(axes.flat, wrong):
        ax.imshow(X[idx["test"][k]])
        ax.set_title(f"label {'smile' if Y[idx['test'][k]] else 'no'} / {p_test[k]:.2f}")
    plt.tight_layout()
    plt.savefig(OUT / "test_errors.png", dpi=100)
    plt.show()
''')

code('''
# 18. tflite 변환과 검증
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
    path = OUT / f"smile_{kind}.tflite"
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
# 19. 결과 저장
# - inference_config.json: 앱이 따라야 할 전처리와 출력 의미, 임계값
# - metrics.json: val / test 지표, tflite 검증 결과
# - 그 밖에 best.keras, *.tflite, history.csv, split_manifest.csv, 그림
config = {
    "model_files": {k: v["file"] for k, v in exported.items()},
    "xnnpack": {k: v["xnnpack"] for k, v in exported.items()},
    "input": {"name": "face_rgb_0_255", "size": IMG_SIZE, "color": "RGB", "range": [0, 255],
              "resize": "bilinear", "bottom": SMILE_BOTTOM,
              "crop": f"YuNet 박스 중심 정사각형(한 변 = 박스 높이)의 아래쪽 {SMILE_BOTTOM} 을 {IMG_SIZE}x{IMG_SIZE} 로 늘림"},
    "output": {"meaning": "웃음 점수, 1 = 웃음", "tta_flip": TTA_FLIP},
    "distill_alpha": DISTILL_ALPHA,
    "threshold": THRESHOLD,
    "threshold_rule": f"val 에서 웃음 precision >= {TARGET_PRECISION} 인 값 중 recall 최대",
    "train_source_size": SRC_SIZE,
}
settings = {k: v for k, v in globals().items()
            if k.isupper() and isinstance(v, (int, float, str, bool, type(None), dict))
            and k not in {"THRESHOLD", "SRC_SIZE"}}
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
