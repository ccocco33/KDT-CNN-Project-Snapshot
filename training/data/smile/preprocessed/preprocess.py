"""smile 학습 데이터 전처리 (GENKI-4K, 64x64)
- 입력: ../raw/{smile_train,no_smile_train,smile_test,no_smile_test}/ (얼굴을 잘라 둔 이미지)
- 출력: ./{같은 폴더}/*.png, ./manifest.csv (이미지별 기록), ./summary.txt
- eye 전처리(../../eye/preprocessed/preprocess.py)와 같은 처리. 다른 점은 흑백 제외를 기본으로 하지 않음
- 처리 (학습과 앱의 입력 모양을 맞추기 위한 고정 처리)
  - YuNet 으로 얼굴을 다시 찾아 자름: 정사각형, 한 변 = min(박스 높이, 원본 너비, 원본 높이)
    - 이미 잘린 이미지라 작으면 못 찾으므로 확대 + 둘레 여백을 두고 검출, 좌표는 원본 기준으로 되돌림
    - 얼굴이 여러 개면 가장 큰 얼굴
    - 박스 중심에 맞추되 원본 밖으로 나가면 안쪽으로 밀어 넣음 (채우는 영역 없음)
      - 원본 크롭이 YuNet 박스보다 좁은 이미지에 채움 흔적(줄무늬, 검은 띠)이 생기는 것을 막음
  - 해상도 통일: 자른 이미지를 OUT_SIZE x OUT_SIZE 로 저장 (줄일 때 INTER_AREA, 키울 때 INTER_LINEAR)
    - eye 데이터와 같은 해상도. 원본이 64px 이라 자른 영역을 다시 64px 로 키움
    - 모델 입력 크기로 바꾸기는 학습 코드와 앱에서 같은 방식으로
  - 제외: YuNet 미검출, 자른 크기 MIN_CROP px 미만 (잘못 검출, 얼굴이 아주 작은 원본)
  - PNG 저장 (재압축 없음)
- 선택 처리 (기본 꺼짐)
  - --drop-gray: 흑백 제외 (채도 평균 < GRAY_SAT). smile 데이터는 흑백 비율이 두 클래스에 비슷해(3.9%, 4.6%) 기본 꺼짐
  - --align: 두 눈이 수평이 되게 회전한 뒤 자름
  - --clahe: 밝기(LAB 의 L)에 CLAHE 적용

실행: app/.venv/bin/python training/data/smile/preprocessed/preprocess.py [--drop-gray] [--align] [--clahe]
"""
import argparse
import csv
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[2]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "smile" / "preprocessed"   # 데이터 위치 (training/paths.py)
RAW = HERE.parent / "raw"
YUNET = REPO / "app/models/face_detection_yunet_2023mar.onnx"

CLASSES = ("smile_train", "no_smile_train", "smile_test", "no_smile_test")
DETECT_SIZE = 256      # 검출용 확대 크기 (긴 변)
PAD = 64               # 검출용 둘레 여백
SCORE_TH = 0.5         # YuNet 신뢰도 기준 (얼굴 1개가 있다고 아는 이미지라 낮게)
GRAY_SAT = 25          # 채도 평균이 이 값 미만이면 흑백으로 봄
OUT_SIZE = 64          # 저장 해상도 (한 변 px)
MIN_CROP = 40          # 자른 크기가 이 값 미만이면 제외
FIELDS = ("class", "src", "dst", "status", "src_w", "src_h", "score",
          "box_x", "box_y", "box_w", "box_h", "roll", "crop", "shift", "upscaled", "saturation")


def detect(yunet, img):
    """가장 큰 얼굴 (x, y, w, h, 오른눈, 왼눈, score). 좌표는 원본 기준. 못 찾으면 None"""
    h, w = img.shape[:2]
    s = DETECT_SIZE / max(h, w)
    big = cv2.resize(img, (round(w * s), round(h * s)))
    big = cv2.copyMakeBorder(big, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT)
    yunet.setInputSize((big.shape[1], big.shape[0]))
    _, det = yunet.detect(big)
    if det is None:
        return None
    d = max(det, key=lambda r: r[2] * r[3])
    back = lambda v, off: (v - off) / s   # 검출 좌표 -> 원본 좌표
    x, y = back(d[0], PAD), back(d[1], PAD)
    bw, bh = d[2] / s, d[3] / s
    right_eye = (back(d[4], PAD), back(d[5], PAD))
    left_eye = (back(d[6], PAD), back(d[7], PAD))
    return x, y, bw, bh, right_eye, left_eye, float(d[14])


def square_crop_inside(img, cx, cy, side):
    """(cx, cy) 중심 정사각형을 원본 안으로 밀어 넣어 자름
    - side 는 원본 너비, 높이 이하
    - 반환: (이미지, 중심에서 밀린 거리 / side)
    """
    h, w = img.shape[:2]
    x1 = min(max(int(round(cx - side / 2)), 0), w - side)
    y1 = min(max(int(round(cy - side / 2)), 0), h - side)
    shift = math.hypot(x1 + side / 2 - cx, y1 + side / 2 - cy) / side
    return img[y1:y1 + side, x1:x1 + side], shift


def rotate(img, cx, cy, angle):
    """(cx, cy) 기준 회전. 빈 곳은 가장자리 복제"""
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def process(yunet, path, cls, args):
    img = cv2.imread(str(path))
    row = {"class": cls, "src": str(path.relative_to(RAW.parent)), "dst": "", "status": "",
           "src_w": img.shape[1], "src_h": img.shape[0]}
    sat = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean())
    row["saturation"] = round(sat, 1)
    if args.drop_gray and sat < GRAY_SAT:
        row["status"] = "excluded_gray"
        return row, None
    found = detect(yunet, img)
    if found is None:
        row["status"] = "excluded_no_face"
        return row, None
    x, y, bw, bh, (rx, ry), (lx, ly), score = found
    roll = math.degrees(math.atan2(ly - ry, lx - rx))
    cx, cy = x + bw / 2, y + bh / 2
    if args.align:
        img = rotate(img, cx, cy, roll)
    side = max(1, min(int(round(bh)), img.shape[0], img.shape[1]))
    if side < MIN_CROP:
        row.update({"status": "excluded_small", "score": round(score, 3), "crop": side})
        return row, None
    face, shift = square_crop_inside(img, cx, cy, side)
    upscaled = side < OUT_SIZE
    face = cv2.resize(face, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR if upscaled else cv2.INTER_AREA)
    if args.clahe:
        face = clahe(face)
    row.update({"status": "ok", "score": round(score, 3), "box_x": round(x, 1), "box_y": round(y, 1),
                "box_w": round(bw, 1), "box_h": round(bh, 1), "roll": round(roll, 1),
                "crop": side, "shift": round(shift, 3), "upscaled": int(upscaled)})
    return row, face


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-gray", action="store_true", help="흑백 제외")
    ap.add_argument("--align", action="store_true", help="두 눈이 수평이 되게 회전")
    ap.add_argument("--clahe", action="store_true", help="밝기 CLAHE")
    args = ap.parse_args()
    yunet = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), score_threshold=SCORE_TH)
    rows = []
    for cls in CLASSES:
        out_dir = HERE / cls
        if out_dir.exists():
            shutil.rmtree(out_dir)   # 이전 결과만 지움 (이 스크립트와 raw 는 건드리지 않음)
        out_dir.mkdir()
        for path in sorted((RAW / cls).iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            row, face = process(yunet, path, cls, args)
            if face is not None:
                dst = out_dir / (path.stem + ".png")
                cv2.imwrite(str(dst), face)
                row["dst"] = str(dst.relative_to(HERE))
            rows.append(row)
    with open(HERE / "manifest.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    summary(rows, args)


def summary(rows, args):
    lines = [f"smile 전처리 (drop_gray={args.drop_gray}, align={args.align}, clahe={args.clahe}, 저장 {OUT_SIZE}x{OUT_SIZE})", "",
             f"{'':<16} {'원본':>6} {'ok':>6} {'흑백 제외':>9} {'미검출 제외':>11} {'작은 크롭 제외':>13}"]
    for cls in CLASSES:
        rs = [r for r in rows if r["class"] == cls]
        n = lambda s: sum(r["status"] == s for r in rs)
        lines.append(f"{cls:<16} {len(rs):>6} {n('ok'):>6} {n('excluded_gray'):>9} {n('excluded_no_face'):>11} {n('excluded_small'):>13}")
    lines.append("")
    for cls in CLASSES:
        ok = [r for r in rows if r["class"] == cls and r["status"] == "ok"]
        crop = np.array([r["crop"] for r in ok])
        shift = np.array([r["shift"] for r in ok])
        up = sum(r["upscaled"] for r in ok)
        lines.append(f"{cls}: 자른 크기 중앙값 {int(np.median(crop))}px ({int(crop.min())}~{int(crop.max())}), "
                     f"{OUT_SIZE}px 미만이라 키운 이미지 {up}장, "
                     f"중심에서 밀림(변 길이 대비) 중앙값 {np.median(shift):.1%}, 5% 이상 {np.mean(shift >= 0.05):.1%}")
    text = "\n".join(lines)
    print(text)
    (HERE / "summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
