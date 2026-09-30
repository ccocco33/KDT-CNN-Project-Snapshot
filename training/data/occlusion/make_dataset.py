"""가림 분류 학습 데이터 생성 + 전처리 (64x64) -> preprocessed.zip
- 보임 기준 (2026-09-27 thinkcat): 입이 안 보이거나 두 눈이 모두 안 보이면 가림(1). 한쪽 눈만 가림, 선글라스류는 보임(0)
- 전처리: smile, eye 전처리본과 같음 (YuNet 박스 중심 정사각형, 한 변 = 박스 높이, 64x64. ../smile/preprocessed/preprocess.py)
- 출처
  - COFW 실제 얼굴 (cofw/faces, cofw_extract.py)
    - 가림: 검수 1 (review.csv) / 보임: 검수 2 (수염, 선글라스, 한쪽 눈 가림) + 가림 거의 없음 (group V, LFPW 포함)
    - 애매(group X)는 제외. YuNet 미검출도 제외 (앱에서도 판정 대상이 아님)
  - 합성 (smile, eye 전처리본 얼굴에 가리개를 붙임). 원본 얼굴마다 한 가지
    - clean: 그대로 (보임)
    - person: 다른 얼굴을 머리 모양으로 잘라 입 쪽을 겹침 (앞사람)
    - transfer: COFW 검수 확정 입 가림 얼굴을 눈, 입 3점으로 정렬해 입 부위만 옮겨 붙임 (손, 마스크, 물건)
    - transfer_skin: transfer 와 같지만 가리개 색을 대상 얼굴 피부색에 맞춤 (본인 손처럼 색 차이가 없는 가림)
    - eyes_person: 앞사람 머리가 위나 옆에서 두 눈을 덮음 (앞사람 뒤통수, 모자챙)
    - eyes_transfer, eyes_transfer_skin: COFW 가리개를 두 눈 사이로 옮겨 가로로 긴 띠 영역만 섞음 (손으로 눈 가림). _skin 은 피부색 맞춤
      - 어두운 가리개(검은 띠 등)도 그대로 둠 (2026-09-27 thinkcat: 선글라스(보임)와 구분하려면 넣는 게 나음)
    - decoy_person, decoy_transfer, decoy_skin: 같은 가리개를 이마, 볼, 옆에 붙임 (가리개가 있지만 보임. "붙인 흔적 = 가림" 지름길 방지)
      - 한쪽 눈만 덮이면 보임 (새 기준). 선글라스(두 눈 위 어두운 렌즈, 보임)는 노트북에서
  - 합성 선글라스는 여기서 만들지 않음 (라벨이 항상 보임이고 눈 좌표만 필요 -> 노트북에서 학습 중 무작위로)
    - 합성 기준: 다른 이미지가 재료로 필요하거나 덮인 정도로 라벨을 정하는 합성은 여기서 미리, 라벨이 고정이고 좌표만 필요한 합성은 노트북에서
- 피부색 맞춤: LAB 색공간에서 가리개 영역의 평균, 표준편차를 대상 얼굴 피부(코 주변 타원)의 값으로 바꿈
    - 가리개 경계는 불규칙한 다각형 + 흐림 (타원 흔적 방지)
- 라벨 (합성): 덮인 비율 = 영역 중 가리개가 절반 이상 섞인 픽셀 비율
  - 가림: 입 >= POS_COVER 또는 두 눈 모두 >= POS_COVER / 보임: 입 < NEG_COVER 이고 두 눈 중 하나 이상 < NEG_COVER
  - 그 사이는 애매 -> 버림
- 분할 (manifest 에 고정. 노트북은 그대로 사용)
  - COFW: cofw_test -> test, cofw_train, lfpw -> train / val (VAL_FRACTION)
  - 합성: smile *_test -> test, 나머지 원본 얼굴 -> train / val / test (VAL_FRACTION, TEST_FRACTION)
  - 가리개 출처도 분할을 넘지 않음: test 합성은 COFW test 얼굴, train / val 합성은 COFW train 얼굴에서 가져옴
    - person 가리개 얼굴은 같은 분할의 원본 얼굴에서
- 출력: preprocessed/{occluded,visible}/*.png, preprocessed/manifest.csv, preprocessed/summary.txt, preprocessed_preview.png, preprocessed.zip
  - manifest: folder, file, label, split, source (cofw | synth), kind, origin (원본 파일), rx, ry, lx, ly (두 눈, 64px 좌표. 노트북 선글라스용. 못 찾으면 빈칸)
    - 합성: 원본 얼굴의 YuNet 랜드마크 / COFW: 64px 로 자른 결과에서 YuNet 랜드마크

실행: app/.venv/bin/python training/data/occlusion/make_dataset.py
"""
import csv
import importlib.util
import math
import shutil
import zipfile
from collections import Counter
import sys
from pathlib import Path

import cv2
import numpy as np

from preview import SIZE, Landmarks, region_masks

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parents[1]))   # training/
from paths import DATASET, REPO  # noqa: E402

HERE = DATASET / "occlusion"   # 데이터 위치 (training/paths.py)
DATA = HERE.parent
COFW = HERE / "cofw"
OUT = HERE / "preprocessed"
ZIP = HERE / "preprocessed.zip"

EYES = ((8, 10, 12, 13, 16), (9, 11, 14, 15, 17))
MOUTH = (22, 23, 24, 25, 26, 27)
POS_COVER, NEG_COVER = 0.6, 0.2
VAL_FRACTION, TEST_FRACTION = 0.15, 0.15
KIND_PROB = {"clean": 0.22, "person": 0.14, "transfer": 0.1, "transfer_skin": 0.1,     # 입 가림
             "eyes_person": 0.08, "eyes_transfer": 0.06, "eyes_transfer_skin": 0.06,  # 두 눈 가림 (입 가림의 약 절반)
             "decoy_person": 0.08, "decoy_transfer": 0.06, "decoy_skin": 0.1}
SEED = 0

_spec = importlib.util.spec_from_file_location("smile_preprocess", CODE.parent / "smile" / "preprocessed" / "preprocess.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def crop64(yunet, img):
    """smile, eye 전처리와 같은 자르기 -> 64x64. 못 찾거나 작으면 None"""
    found = prep.detect(yunet, img)
    if found is None:
        return None
    x, y, bw, bh, _, _, _ = found
    side = max(1, min(int(round(bh)), img.shape[0], img.shape[1]))
    if side < prep.MIN_CROP:
        return None
    face, _ = prep.square_crop_inside(img, x + bw / 2, y + bh / 2, side)
    interp = cv2.INTER_LINEAR if side < SIZE else cv2.INTER_AREA
    return cv2.resize(face, (SIZE, SIZE), interpolation=interp)


# ---- COFW ----

def cofw_rows():
    review = {r["file"]: int(r["label"]) for r in csv.DictReader(open(HERE / "review.csv"))}
    return list(csv.DictReader(open(COFW / "faces.csv"))), review


def cofw_label(r, review):
    if r["file"] in review:
        return {1: 1, 2: 0}.get(review[r["file"]])
    return 0 if r["group"] == "V" else None


def cofw_split(r, rng):
    if r["part"] == "cofw_test":
        return "test"
    return "val" if rng.random() < VAL_FRACTION else "train"


def transfer_sources(rows, review, split):
    """입 가림 옮겨 붙이기 출처 (COFW 검수 확정 입 가림). test 는 cofw_test, train / val 은 cofw_train"""
    parts = ("cofw_test",) if split == "test" else ("cofw_train", "lfpw")
    out = []
    for r in rows:
        if r["part"] in parts and r["group"] == "M" and review.get(r["file"]) == 1:
            pt = lambda ids: (np.mean([float(r[f"x{i}"]) for i in ids]), np.mean([float(r[f"y{i}"]) for i in ids]))  # noqa: E731
            out.append((r["file"], pt(EYES[0]), pt(EYES[1]), pt(MOUTH)))
    return out


# ---- 합성 ----

def irregular_mask(cx, cy, rx, ry, rng):
    """(cx, cy) 중심, 반지름 약 (rx, ry) 의 불규칙 다각형 마스크"""
    n = int(rng.integers(7, 13))
    t = np.sort(rng.uniform(0, 2 * np.pi, n))
    k = rng.uniform(0.75, 1.25, n)
    pts = np.stack([cx + rx * k * np.cos(t), cy + ry * k * np.sin(t)], 1).round().astype(np.int32)
    m = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m


def blend(img, layer, mask, rng):
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), rng.uniform(0.6, 1.6))[..., None]
    return np.clip(img * (1 - a) + layer * a, 0, 255).astype(np.uint8), a[..., 0] > 0.5


def mouth_center(lm):
    (_, _), (_, _), (mrx, mry), (mlx, mly) = lm
    return (mrx + mlx) / 2, (mry + mly) / 2


def decoy_point(lm, rng):
    """입, 눈에서 떨어진 곳: 이마, 볼 바깥, 얼굴 옆"""
    (rx, ry), (lx, ly), _, _ = lm
    d = math.hypot(lx - rx, ly - ry)
    options = [((rx + lx) / 2, ry - d * rng.uniform(0.6, 0.9)),
               (rx - d * rng.uniform(0.5, 0.8), ry + d * rng.uniform(0.3, 0.6)),
               (lx + d * rng.uniform(0.5, 0.8), ly + d * rng.uniform(0.3, 0.6))]
    return options[int(rng.integers(len(options)))]


def eyes_center(lm):
    (rx, ry), (lx, ly), _, _ = lm
    return (rx + lx) / 2, (ry + ly) / 2


def occ_person(img, lm, donor, target, rng, side="below"):
    """앞사람: 다른 얼굴을 머리 모양으로 잘라 target 쪽으로 겹침
    - side: below (아래에서, 입 가림) / above (위나 옆에서, 눈 가림) / any (아무 방향, decoy)
    """
    tx, ty = target
    s = rng.uniform(0.9, 1.4)
    d = cv2.resize(donor, None, fx=s, fy=s)
    if rng.random() < 0.5:
        d = d[:, ::-1]
    h, w = d.shape[:2]
    ang = {"below": lambda: rng.uniform(np.pi * 0.2, np.pi * 0.8),
           "above": lambda: rng.uniform(-np.pi * 0.95, -np.pi * 0.05),
           "any": lambda: rng.uniform(0, 2 * np.pi)}[side]()
    off = rng.uniform(0.15, 0.3) * w if side == "above" else rng.uniform(0.3, 0.45) * w   # 위에서는 더 가까이 (두 눈까지 덮도록)
    cx, cy = tx + off * math.cos(ang), ty + off * math.sin(ang)
    layer = np.zeros_like(img)
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    xs, ys, xe, ye = max(0, x0), max(0, y0), min(SIZE, x0 + w), min(SIZE, y0 + h)
    if xe <= xs or ye <= ys:
        return img, np.zeros((SIZE, SIZE), bool)
    layer[ys:ye, xs:xe] = d[ys - y0:ye - y0, xs - x0:xe - x0]
    mask = irregular_mask(cx, cy, w * 0.42, h * 0.5, rng)
    return blend(img, layer, mask, rng)


def skin_stats(img, lm):
    """대상 얼굴 피부 (코 주변 타원) 의 LAB 평균, 표준편차"""
    (rx, ry), (lx, ly), _, _ = lm
    mx, my = mouth_center(lm)
    d = math.hypot(lx - rx, ly - ry)
    m = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(m, (int((rx + lx) / 2), int(((ry + ly) / 2 + my) / 2)), (max(2, int(d * 0.45)), max(2, int(d * 0.3))), 0, 0, 360, 1, -1)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[m > 0]
    return lab.mean(0), lab.std(0) + 1e-3


def match_skin(layer, mask, stats):
    """가리개 영역(mask) 의 LAB 평균, 표준편차를 stats 로 바꿈"""
    lab = cv2.cvtColor(layer, cv2.COLOR_BGR2LAB).astype(np.float32)
    sel = mask > 0
    if sel.sum() < 4:
        return layer
    mean, std = lab[sel].mean(0), lab[sel].std(0) + 1e-3
    tmean, tstd = stats
    out = (lab - mean) / std * tstd + tmean
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def occ_transfer(img, lm, src, target, rng, skin=False, eyes=False):
    """COFW 입 가림 부위를 정렬해 target 중심 불규칙 영역만 섞음. skin 이면 가리개 색을 대상 피부색에 맞춤
    - eyes: 두 눈 사이로 옮겨 가로로 긴 띠 영역 (두 눈을 함께 덮음). target 은 무시
    """
    if eyes:
        target = eyes_center(lm)
    f, ce0, ce1, cm = src
    face = cv2.imread(str(COFW / "faces" / f))
    (rx, ry), (lx, ly), _, _ = lm
    dst = np.float32([(rx, ry), (lx, ly), mouth_center(lm)])
    M = cv2.getAffineTransform(np.float32([ce0, ce1, cm]), dst)
    warped = cv2.warpAffine(face, M, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    if target is not None:   # decoy: 입 부위 가리개를 다른 곳으로 옮김
        mx, my = mouth_center(lm)
        shift = np.float32([[1, 0, target[0] - mx], [0, 1, target[1] - my]])
        warped = cv2.warpAffine(warped, shift, (SIZE, SIZE), borderMode=cv2.BORDER_REPLICATE)
        c = target
    else:
        c = mouth_center(lm)
    d = math.hypot(lx - rx, ly - ry)
    if eyes:
        mask = irregular_mask(c[0], c[1], d * rng.uniform(0.85, 1.15), d * rng.uniform(0.3, 0.45), rng)
    else:
        mask = irregular_mask(c[0], c[1], d * rng.uniform(0.7, 1.0), d * rng.uniform(0.45, 0.65), rng)
    if skin:
        warped = match_skin(warped, mask, skin_stats(img, lm))
    return blend(img, warped, mask, rng)


def synth_label(covered, lm):
    reg = region_masks(lm)
    cov = {k: float((covered & (m > 0)).sum()) / max(1, (m > 0).sum()) for k, m in reg.items()}
    both_eyes = min(cov["right_eye"], cov["left_eye"])
    if cov["mouth"] >= POS_COVER or both_eyes >= POS_COVER:
        return 1
    if cov["mouth"] < NEG_COVER and both_eyes < NEG_COVER:
        return 0
    return None


def base_faces(rng):
    """[(경로, 분할)] smile 전처리본 + eye 전처리본"""
    out = []
    for folder in ("smile_train", "no_smile_train", "smile_test", "no_smile_test"):
        for p in sorted((DATA / "smile" / "preprocessed" / folder).glob("*.png")):
            out.append((p, "test" if folder.endswith("_test") else None))
    for folder in ("opened", "closed"):
        out += [(p, None) for p in sorted((DATA / "eye" / "preprocessed" / folder).glob("*.png"))]
    result = []
    for p, sp in out:
        if sp is None:
            u = rng.random()
            sp = "test" if u < TEST_FRACTION else ("val" if u < TEST_FRACTION + VAL_FRACTION else "train")
        result.append((p, sp))
    return result


def main():
    rng = np.random.default_rng(SEED)
    yunet = cv2.FaceDetectorYN.create(str(prep.YUNET), "", (320, 320), score_threshold=prep.SCORE_TH)
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ("occluded", "visible"):
        (OUT / d).mkdir(parents=True)
    rows = []

    lmk = Landmarks()

    def save(img, label, split, source, kind, origin, name, lm=None):
        folder = "occluded" if label == 1 else "visible"
        cv2.imwrite(str(OUT / folder / name), img)
        if lm is None:
            found_lm, ok = lmk(img)
            lm = found_lm if ok else None
        eyes = {"rx": "", "ry": "", "lx": "", "ly": ""} if lm is None else \
            {"rx": round(lm[0][0], 2), "ry": round(lm[0][1], 2), "lx": round(lm[1][0], 2), "ly": round(lm[1][1], 2)}
        rows.append({"folder": folder, "file": name, "label": label, "split": split, "source": source,
                     "kind": kind, "origin": origin, **eyes})

    # COFW
    crows, review = cofw_rows()
    dropped = Counter()
    for r in crows:
        label = cofw_label(r, review)
        if label is None:
            dropped["cofw_ambiguous"] += 1
            continue
        face = crop64(yunet, cv2.imread(str(COFW / "faces" / r["file"])))
        if face is None:
            dropped[f"cofw_no_face_label{label}"] += 1
            continue
        kind = "cofw_" + ("reviewed" if r["file"] in review else "plain")
        save(face, label, cofw_split(r, rng), "cofw", kind, r["file"], "cofw_" + r["file"])

    # 합성
    bases = base_faces(rng)
    by_split = {s: [p for p, sp in bases if sp == s] for s in ("train", "val", "test")}
    sources = {s: transfer_sources(crows, review, s) for s in ("train", "val", "test")}
    kinds, probs = list(KIND_PROB), np.array(list(KIND_PROB.values()))
    for i, (path, split) in enumerate(bases):
        img = cv2.imread(str(path))
        lm, found = lmk(img)
        if not found:
            dropped["synth_no_landmark"] += 1
            continue
        kind = kinds[int(rng.choice(len(kinds), p=probs / probs.sum()))]
        if kind == "clean":
            out, label = img, 0
        else:
            if kind in ("person", "eyes_person", "decoy_person"):
                donor = cv2.imread(str(by_split[split][int(rng.integers(len(by_split[split])))]))
                target, side = {"person": (mouth_center(lm), "below"), "eyes_person": (eyes_center(lm), "above"),
                                "decoy_person": (decoy_point(lm, rng), "any")}[kind]
                out, covered = occ_person(img, lm, donor, target, rng, side=side)
            else:
                src = sources[split][int(rng.integers(len(sources[split])))]
                target = decoy_point(lm, rng) if kind.startswith("decoy") else None
                out, covered = occ_transfer(img, lm, src, target, rng, skin=kind.endswith("skin"), eyes=kind.startswith("eyes"))
            label = synth_label(covered, lm)
            if label is None:
                dropped[f"synth_ambiguous_{kind}"] += 1
                continue
        origin = f"{path.parent.parent.parent.name}/{path.parent.name}/{path.name}"
        save(out, label, split, "synth", kind, origin, f"synth_{i:05d}.png", lm)

    with open(OUT / "manifest.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary(rows, dropped)
    preview(rows)
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as zf:   # PNG 는 이미 압축됨
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(HERE))
    print(f"zip: {ZIP} ({ZIP.stat().st_size / 1e6:.1f} MB)")


def summary(rows, dropped):
    lines = ["가림 분류 데이터 (64x64). label 1 = 가림, 0 = 보임", ""]
    c = Counter((r["split"], r["source"], r["kind"], r["label"]) for r in rows)
    lines.append(f"{'split':<6} {'source':<6} {'kind':<16} {'가림':>6} {'보임':>6}")
    for split in ("train", "val", "test"):
        for source, kind in sorted({(s, k) for sp, s, k, _ in c if sp == split}):
            lines.append(f"{split:<6} {source:<6} {kind:<16} {c[(split, source, kind, 1)]:>6} {c[(split, source, kind, 0)]:>6}")
        tot = Counter(r["label"] for r in rows if r["split"] == split)
        lines.append(f"{split:<6} {'합계':<23} {tot[1]:>6} {tot[0]:>6}")
        lines.append("")
    lines.append("제외: " + ", ".join(f"{k} {v}" for k, v in sorted(dropped.items())))
    text = "\n".join(lines)
    print(text)
    (OUT / "summary.txt").write_text(text + "\n")


def preview(rows, per=10):
    """종류별 예시 (칸 아래: 라벨)"""
    rng = np.random.default_rng(1)
    groups = {}
    for r in rows:
        groups.setdefault((r["kind"], r["label"]), []).append(r)
    lines = []
    for (kind, label), rs in sorted(groups.items()):
        pick = rng.choice(len(rs), min(per, len(rs)), replace=False)
        tiles = []
        for i in pick:
            t = cv2.resize(cv2.imread(str(OUT / rs[i]["folder"] / rs[i]["file"])), (96, 96), interpolation=cv2.INTER_NEAREST)
            tiles.append(t)
        while len(tiles) < per:
            tiles.append(np.full((96, 96, 3), 40, np.uint8))
        bar = np.full((96, 170, 3), 40, np.uint8)
        cv2.putText(bar, kind, (4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, "OCC" if label else "vis", (4, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (80, 80, 255) if label else (120, 230, 120), 1, cv2.LINE_AA)
        lines.append(np.hstack([bar] + tiles))
    cv2.imwrite(str(HERE / "preprocessed_preview.png"), np.vstack(lines))


if __name__ == "__main__":
    main()
