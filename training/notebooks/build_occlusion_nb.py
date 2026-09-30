"""occlusion(가림) 학습 노트북 생성기 (셀 내용 -> .ipynb)
- 합성 선글라스 함수는 training/data/eye/augment_review/eyewear_fn.py 에서 가져옴 (SUNGLASSES_SRC. eye 노트북과 같은 함수)

실행: python training/notebooks/build_occlusion_nb.py training/notebooks/occlusion_train.ipynb training/data/eye/augment_review/eyewear_fn.py
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
# occlusion 분류 모델 학습 (얼굴 가림)

- 가림 기준: 입이 안 보이거나 두 눈이 모두 안 보이면 가림. 한쪽 눈만 가림, 선글라스류는 보임
- 입력: YuNet 으로 자른 얼굴 (전처리 데이터 zip `DRIVE_FACE/occlusion/preprocessed.zip`, 64x64 PNG)
  - zip 안 `preprocessed/manifest.csv`: folder, file, label (1 가림, 0 보임), split (train / val / test), source (cofw / synth), kind, rx, ry, lx, ly (두 눈)
  - 분할은 manifest 에 고정 (합성 원본 얼굴, 가리개 출처가 분할을 넘지 않게 만든 것). 노트북에서 다시 나누지 않음
  - 가리개 합성(앞사람, COFW 옮겨 붙이기, 피부색 맞춤)은 zip 에 미리 / 합성 선글라스는 여기서 학습 중 무작위로 (라벨 그대로)
- 출력: 가림 점수 (0~1, **1 = 가림**)
- 모델: ImageNet 사전학습 백본 + 전역 평균 풀링 + 이진 분류. 백본까지 한 번에 학습
- 실제 가림 얼굴(COFW)은 적고 합성이 많음 -> 실제 얼굴에 가중치 (`REAL_WEIGHT`), 평가는 실제와 합성을 나눠 봄
  - 최종 판단은 노트북 밖 WIDER 단체 사진 평가로 (합성 점수가 높아도 실제에서 낮을 수 있음)
- 결과: tflite (fp32, dynamic range, int8), 추론 설정 json, 지표 (`OUTPUT_ROOT/RUN_NAME`)
- 실행: Colab GPU 런타임에서 위에서 아래로. 설정은 1~6번 셀에서만 바꿈
''')

code('''
# 1. 경로 설정
# - DRIVE_FACE: 데이터셋 루트 (Google Drive)
# - DATA_ZIP: 전처리한 가림 데이터 zip. 안에 manifest.csv 와 occluded / visible 폴더
# - LOCAL_DATA_DIR: zip 을 풀 로컬 경로 (Drive 에서 직접 읽으면 느림). 이미 풀려 있으면 다시 풀지 않음
# - OUTPUT_ROOT: 학습 결과 저장 위치. 실행마다 RUN_NAME 하위 폴더 생성
# - RUN_NAME: None 이면 실행 시각 (예: 20260927-1530)
DRIVE_FACE = "/content/drive/MyDrive/Team-11/dataset"
DATA_ZIP = f"{DRIVE_FACE}/occlusion/preprocessed.zip"
LOCAL_DATA_DIR = "/content/data/occlusion"
OUTPUT_ROOT = "/content/drive/MyDrive/Team-11/models/occlusion"
RUN_NAME = None
''')

code('''
# 2. 데이터 설정
# - SYNTH_KINDS: 쓸 합성 종류. 빼면 그 종류는 모든 분할에서 제외
#   - clean: 그대로 (보임)
#   - person: 앞사람 겹침 / transfer: COFW 입 가림 옮겨 붙이기 / transfer_skin: 같은 가리개를 대상 피부색으로 (본인 손)
#   - eyes_person, eyes_transfer, eyes_transfer_skin: 두 눈 가림 (앞사람 머리가 위에서, 손, 물건이 가로 띠로)
#   - decoy_person, decoy_transfer, decoy_skin: 가리개를 이마, 볼, 옆에 붙임 (대부분 보임. "붙인 흔적 = 가림" 지름길 방지)
# - USE_COFW: COFW 실제 얼굴 사용 여부
# - SEED: 난수 시드
SYNTH_KINDS = ["clean", "person", "transfer", "transfer_skin", "eyes_person", "eyes_transfer", "eyes_transfer_skin",
               "decoy_person", "decoy_transfer", "decoy_skin"]
USE_COFW = True
SEED = 42
''')

code('''
# 3. 모델 설정
# - BACKBONE: mobilenet_v3_small | mobilenet_v3_large | mobilenet_v2 | efficientnet_b0
# - ALPHA: MobileNet 폭 배율 (efficientnet_b0 는 무시)
# - IMG_SIZE: 모델 입력 크기 (정사각형. 얼굴 전체)
# - DROPOUT: 분류 직전 드롭아웃 비율
# - TRAINABLE_BACKBONE_LAYERS: None 이면 백본 전체 학습, 정수 N 이면 백본 마지막 N 개 층만 학습
BACKBONE = "mobilenet_v3_small"
ALPHA = 1.0
IMG_SIZE = 128
DROPOUT = 0.3
TRAINABLE_BACKBONE_LAYERS = None
''')

code('''
# 4. 학습 설정
# - LEARNING_RATE: 최대 학습률. WARMUP_EPOCHS 동안 1/10 에서 올라간 뒤 코사인으로 감소
# - WEIGHT_DECAY: AdamW 가중치 감쇠
# - LABEL_SMOOTHING: 라벨 스무딩 (0 이면 끔)
# - EARLY_STOP_PATIENCE: 검증 AUC 가 이 epoch 수 동안 오르지 않으면 중단 (가장 좋은 가중치로 복원)
# - USE_CLASS_WEIGHT: 클래스 수 차이 보정 (가림, 보임 전체 수 기준)
# - REAL_WEIGHT: COFW 실제 얼굴의 샘플 가중치 배율 (합성에 묻히지 않도록. 1.0 이면 합성과 같음)
# - USE_EMA: 가중치 이동 평균(EMA)을 최종 모델로. 검증, 체크포인트도 EMA 가중치로 (과적합 완화)
# - EMA_MOMENTUM: EMA 비율. 클수록 과거 가중치를 오래 반영 (0.99: 약 100 step)
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WARMUP_EPOCHS = 2
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 8
USE_CLASS_WEIGHT = True
REAL_WEIGHT = 3.0
USE_EMA = True
EMA_MOMENTUM = 0.99
''')

code('''
# 5. 증강 설정 (학습 데이터에만 적용. 확률 0 또는 범위 0 이면 끔)
# - 합성 선글라스: 확률 AUG_SUNGLASSES_PROB 로 두 눈 위에 선글라스. 라벨은 그대로 (선글라스류는 보이는 얼굴. 입 가림이면 가림 그대로)
#   - SUNGLASSES_MIN_ALPHA: 렌즈 불투명도 하한
#   - 눈 좌표(manifest rx, ry, lx, ly)가 없는 이미지는 적용 안 함
AUG_SUNGLASSES_PROB = 0.10
SUNGLASSES_MIN_ALPHA = 0.7
# - 실제 카메라에서 생기는 변화만 흔듦
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
# - TARGET_VISIBLE_FPR: 보이는 얼굴을 가림으로 보는 비율 목표. 검증 데이터에서 이 값 이하인 임계값 중 가림 recall 최대
#   - 보이는 얼굴을 가림으로 보면 미리보기 안내가 떠서 촬영이 시작되지 않음 -> 이쪽 오판을 작게
# - tflite 변환 (fp32 는 항상)
#   - EXPORT_DYNAMIC: dynamic range 양자화. 가중치 int8, 입출력 float32. 대표 데이터 불필요
#   - EXPORT_INT8: 전체 int8 양자화. 가중치, 활성값, 입출력 int8
# - TTA_FLIP: 원본과 좌우 뒤집은 얼굴의 점수 평균을 모델 안에 넣어 내보냄
#   - 앱 코드 변경 없음, 추론 비용 2배. 임계값과 평가도 이 모델로
# - REPRESENTATIVE_SAMPLES: int8 양자화에서 활성값 범위를 잴 학습 이미지 수 (가림, 보임 반씩)
TARGET_VISIBLE_FPR = 0.02
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
# - manifest.csv 가 있는 위치를 찾음 (zip 안 상위 폴더 유무와 무관)
# - manifest 의 이미지를 모두 읽음 (RGB, 0~255). 모든 이미지는 같은 크기 (전처리 결과 64x64)
# - SYNTH_KINDS, USE_COFW 로 거름
import zipfile


def is_junk(name):
    return "__MACOSX" in Path(name).parts or Path(name).name.startswith("._")


local = Path(LOCAL_DATA_DIR)
done_mark = local / ".unzipped"
if not done_mark.exists():
    local.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATA_ZIP) as zf:
        members = [m for m in zf.namelist() if not is_junk(m)]
        zf.extractall(local, members=members)
    done_mark.write_text(DATA_ZIP)
manifests = sorted(p for p in local.rglob("manifest.csv") if not is_junk(str(p)))
assert manifests, f"{local} 아래에 manifest.csv 없음"
DATA_DIR = manifests[0].parent
print("데이터 위치:", DATA_DIR)

df = pd.read_csv(DATA_DIR / "manifest.csv")
keep = (df.source == "synth") & df.kind.isin(SYNTH_KINDS)
if USE_COFW:
    keep |= df.source == "cofw"
df = df[keep].reset_index(drop=True)
images = []
for r in df.itertuples():
    with Image.open(DATA_DIR / r.folder / r.file) as src:
        images.append(np.asarray(src.convert("RGB")))
sizes = {im.shape for im in images}
assert len(sizes) == 1, f"이미지 크기가 섞여 있음: {sizes}"
X = np.stack(images)
SRC_SIZE = X.shape[1]
Y = df["label"].to_numpy(np.float32)
LM = df[["rx", "ry", "lx", "ly"]].to_numpy(np.float32)   # 두 눈 좌표 (없으면 NaN)
print("이미지:", X.shape, f"/ 눈 좌표 없음 {int(np.isnan(LM).any(1).sum())}장")
''')

code('''
# 9. 분할 확인
# - manifest 의 split 그대로 (train / val / test)
# - 분할, 출처, 종류별 가림 / 보임 수
idx = {s: df.index[df.split == s].to_numpy() for s in ("train", "val", "test")}
table = df.groupby(["split", "source", "kind", "label"]).size().unstack(fill_value=0)
table.columns = [{0: "보임", 1: "가림"}[c] for c in table.columns]
display(table)
display(df.groupby(["split", "label"]).size().unstack().rename(columns={0: "보임", 1: "가림"}))
df.to_csv(OUT / "split_manifest.csv", index=False)
''')

code('''
# 10. 샘플 가중치
# - 클래스 가중치 (USE_CLASS_WEIGHT): train 의 가림, 보임 수 차이 보정
# - 실제 얼굴 가중치 (REAL_WEIGHT): COFW 얼굴에 곱함
W = np.ones(len(df), np.float32)
if USE_CLASS_WEIGHT:
    counts = np.bincount(Y[idx["train"]].astype(int), minlength=2)
    class_weight = {c: len(idx["train"]) / (2 * counts[c]) for c in (0, 1)}
    W *= np.where(Y == 1, class_weight[1], class_weight[0]).astype(np.float32)
    print("클래스 가중치:", class_weight)
W[(df.source == "cofw").to_numpy()] *= REAL_WEIGHT
print("train 가중치 합: 가림 {:.0f}, 보임 {:.0f} (COFW 가림 {:.0f})".format(
    W[idx["train"]][Y[idx["train"]] == 1].sum(), W[idx["train"]][Y[idx["train"]] == 0].sum(),
    W[idx["train"]][(Y[idx["train"]] == 1) & (df.source.to_numpy()[idx["train"]] == "cofw")].sum()))
''')

code('''
# 10-1. 합성 선글라스 (증강)
# - add_sunglasses: 두 눈 위에 어두운 렌즈, 테, 코 받침. 렌즈 모양, 크기, 색, 불투명도, 반사광은 무작위
# - 라벨은 바꾸지 않음 (선글라스류는 보이는 얼굴)
# - eye 노트북과 같은 함수 (training/data/eye/augment_review/eyewear_fn.py)
import math

import cv2

''' + SUNGLASSES_SRC + '''


def sunglasses_np(image, lm, force=False):
    """학습 증강용 (numpy). AUG_SUNGLASSES_PROB 확률 (force 면 항상). 눈 좌표가 없으면 적용 안 함"""
    rng = np.random.default_rng()
    if np.isnan(lm).any() or (not force and rng.random() >= AUG_SUNGLASSES_PROB):
        return image
    out = add_sunglasses(np.ascontiguousarray(image), (lm[0], lm[1]), (lm[2], lm[3]), rng, min_alpha=SUNGLASSES_MIN_ALPHA)
    return out.astype(np.uint8)


def add_sunglasses_tf(image, lm):
    out = tf.numpy_function(sunglasses_np, [image, lm], tf.uint8)
    out.set_shape(image.shape)
    return out
''')

code('''
# 11. 입력 파이프라인
# - 공통 (학습, 검증, 시험, 앱): 0~1 로 바꿔 처리 -> IMG_SIZE 로 bilinear 크기 변환 -> 0~255
#   - 모델 입력은 RGB 0~255. 정규화는 모델 안에서
# - 학습만: 합성 선글라스 (10-1번 셀), 5번 셀의 증강 (이미지마다 무작위), 10번 셀의 샘플 가중치
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


def to_model_input(img):
    """0~1 이미지 -> 모델 입력 (IMG_SIZE, bilinear, 0~255)"""
    return tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear") * 255.0


def make_dataset(indices, training):
    if training:
        ds = tf.data.Dataset.from_tensor_slices((X[indices], Y[indices], W[indices], LM[indices]))
        ds = ds.shuffle(len(indices), seed=SEED, reshuffle_each_iteration=True)

        def prep(image, label, weight, lm):
            if AUG_SUNGLASSES_PROB > 0:
                image = add_sunglasses_tf(image, lm)
            img = augment(tf.cast(image, tf.float32) / 255.0)
            return to_model_input(img), label[None], weight
    else:
        ds = tf.data.Dataset.from_tensor_slices((X[indices], Y[indices]))

        def prep(image, label):
            return to_model_input(tf.cast(image, tf.float32) / 255.0), label[None]

    return ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(idx["train"], training=True)
val_ds = make_dataset(idx["val"], training=False)
test_ds = make_dataset(idx["test"], training=False)
''')

code('''
# 12. 증강 미리보기
# - 위: 원본 / 아래: 증강 결과 (같은 이미지). 가림, 보임 반씩
# - 증강이 너무 세거나 약하면 5번 셀 조정
rng = np.random.default_rng(SEED)
sample = np.concatenate([rng.choice(idx["train"][Y[idx["train"]] == c], 4, replace=False) for c in (1, 0)])
fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
for col, i in enumerate(sample):
    original = X[i].astype(np.float32) / 255.0
    augmented = augment(tf.constant(original)).numpy()
    for row, im in enumerate((original, augmented)):
        axes[row, col].imshow(im)
        axes[row, col].axis("off")
    axes[0, col].set_title(f"{'occluded' if Y[i] == 1 else 'visible'} / {df.kind[i]}", fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "augmentation_preview.png", dpi=100)
plt.show()
''')

code('''
# 13. 모델 만들기
# - 입력: face_rgb_0_255 (IMG_SIZE x IMG_SIZE x 3, RGB 0~255)
# - 백본 정규화는 모델 안에서 (앱은 0~255 그대로 넣음)
# - 백본의 BatchNorm 은 추론 모드로 고정 (작은 데이터에서 통계가 흔들리지 않게)
# - 출력: occlusion_score (sigmoid, 1 = 가림)
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
    outputs = keras.layers.Dense(1, activation="sigmoid", name="occlusion_score")(x)
    return keras.Model(inputs, outputs, name=f"occlusion_{BACKBONE}")


model = build_model()
trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
print(f"{model.name}: 전체 파라미터 {model.count_params():,}, 학습 파라미터 {trainable:,}")
''')

code('''
# 14. 학습
# - AdamW + 워밍업 후 코사인 감소
# - USE_EMA 이면 epoch 끝마다 EMA 가중치로 바꿔 검증, 저장 (SwapEMAWeights 는 ModelCheckpoint 앞에)
# - 검증 AUC 기준으로 가장 좋은 모델 저장 (best.keras), 조기 종료
# - 샘플 가중치는 train_ds 에 포함 (10번 셀)
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
# - 보이는 얼굴을 가림으로 보는 비율(FPR) <= TARGET_VISIBLE_FPR 인 임계값 중 가림 recall 최대
def pick_threshold(y, p):
    fpr, tpr, thresholds = roc_curve(y, p)
    ok = np.flatnonzero(fpr <= TARGET_VISIBLE_FPR)
    i = ok[np.argmax(tpr[ok])]
    return float(min(thresholds[i], 1.0)), float(fpr[i]), float(tpr[i])


def scores(ds):
    return infer_model.predict(ds, verbose=0).ravel()


p_val = scores(val_ds)
THRESHOLD, val_fpr, val_tpr = pick_threshold(Y[idx["val"]], p_val)
print(f"val AUC {roc_auc_score(Y[idx['val']], p_val):.4f}")
print(f"임계값 {THRESHOLD:.4f}: 가림 recall {val_tpr:.3f}, 보이는 얼굴을 가림으로 {val_fpr:.3f}")
''')

code('''
# 17. 시험 평가 (결과를 보고 설정을 다시 고르면 test 가 검증 데이터가 됨)
# - 전체, 출처별(COFW 실제 / 합성), 종류별 지표
#   - 가림 recall: 가린 얼굴을 가림으로 / 보임 오판: 보이는 얼굴을 가림으로
# - 확신하며 틀린 이미지 12장 (라벨 오류, 약점 확인용)
def report(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")
    return {"n": len(y), "auc": auc, "threshold": t,
            "occluded_recall": tp / max(tp + fn, 1), "visible_fpr": fp / max(fp + tn, 1),
            "occluded_precision": tp / max(tp + fp, 1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


p_test = scores(test_ds)
metrics = {"val": report(Y[idx["val"]], p_val, THRESHOLD), "test": report(Y[idx["test"]], p_test, THRESHOLD)}
test_df = df.loc[idx["test"]].assign(score=p_test)
for source, g in test_df.groupby("source"):
    metrics[f"test_{source}"] = report(g.label.to_numpy(), g.score.to_numpy(), THRESHOLD)
display(pd.DataFrame(metrics).T)

by_kind = test_df.assign(pred=(test_df.score >= THRESHOLD).astype(int)).groupby(["kind", "label"]).agg(
    n=("pred", "size"), predicted_occluded=("pred", "mean"), median_score=("score", "median"))
print("시험 종류별: predicted_occluded = 가림으로 판정한 비율 (label 1 이면 recall, 0 이면 오판)")
display(by_kind)
metrics["test_by_kind"] = {f"{k}_{l}": v for (k, l), v in by_kind.to_dict("index").items()}

# 선글라스 확인: test 얼굴에 합성 선글라스를 씌워도 라벨대로 판정하는지 (보임은 보임, 가림은 가림)
has_lm = idx["test"][~np.isnan(LM[idx["test"]]).any(1)]
worn = np.stack([sunglasses_np(X[i], LM[i], force=True) for i in has_lm])
worn_ds = tf.data.Dataset.from_tensor_slices(worn).map(lambda im: to_model_input(tf.cast(im, tf.float32) / 255.0)).batch(BATCH_SIZE)
p_worn = infer_model.predict(worn_ds, verbose=0).ravel()
y_worn = Y[has_lm]
metrics["test_sunglasses"] = report(y_worn, p_worn, THRESHOLD)
print(f"합성 선글라스를 씌운 test: 보임을 가림으로 {np.mean(p_worn[y_worn == 0] >= THRESHOLD):.3f} "
      f"(씌우기 전 {metrics['test']['visible_fpr']:.3f}), 가림 recall {np.mean(p_worn[y_worn == 1] >= THRESHOLD):.3f}")

wrong = np.flatnonzero((p_test >= THRESHOLD).astype(int) != Y[idx["test"]])
wrong = wrong[np.argsort(-np.abs(p_test[wrong] - THRESHOLD))][:12]
if len(wrong):
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, k in zip(axes.flat, wrong):
        i = idx["test"][k]
        ax.imshow(X[i])
        ax.set_title(f"{'occ' if Y[i] else 'vis'} {df.kind[i]} / {p_test[k]:.2f}", fontsize=8)
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
    path = OUT / f"occlusion_{kind}.tflite"
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
# - metrics.json: val / test 지표 (출처별, 종류별 포함), tflite 검증 결과
# - 그 밖에 best.keras, *.tflite, history.csv, split_manifest.csv, 그림
config = {
    "model_files": {k: v["file"] for k, v in exported.items()},
    "xnnpack": {k: v["xnnpack"] for k, v in exported.items()},
    "input": {"name": "face_rgb_0_255", "size": IMG_SIZE, "color": "RGB", "range": [0, 255],
              "resize": "bilinear", "crop": "YuNet 박스 중심 정사각형, 한 변 = 박스 높이"},
    "output": {"meaning": "가림 점수, 1 = 가림 (입이 안 보이거나 두 눈이 모두 안 보임)", "tta_flip": TTA_FLIP},
    "threshold": THRESHOLD,
    "threshold_rule": f"val 에서 보이는 얼굴을 가림으로 보는 비율 <= {TARGET_VISIBLE_FPR} 인 값 중 가림 recall 최대",
    "train_source_size": SRC_SIZE,
}
settings = {k: v for k, v in globals().items()
            if k.isupper() and isinstance(v, (int, float, str, bool, type(None), dict, list))
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
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": source}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    nb["cells"].append(cell)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("written", OUT, len(CELLS), "cells")
