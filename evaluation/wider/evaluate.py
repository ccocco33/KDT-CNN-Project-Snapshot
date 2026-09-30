"""단체 사진 평가: 앱 모델(config.MODEL)을 라벨과 비교
- 대상: 모든 얼굴 라벨이 끝났고 부적절(unusable)이 아닌 사진
  - 환경변수 EVAL_SUBSET=dev|holdout 이면 split.csv 의 해당 사진만 (없으면 전부). 이 파일의 load() 를 쓰는 스크립트 모두에 적용
- 모델: MODEL (기본 config.MODEL). SCORE 를 주면 yunet_landmark 를 그 YuNet 신뢰도 기준으로 생성
- 입력: 가로 WIDTH px 로 줄인 사진 (카메라 해상도 흉내). 0 이면 원본
- 매칭: 예측 얼굴과 정답 얼굴을 IoU 큰 순으로 1:1 짝지음 (IoU >= IOU_TH)
- 판단 불가(-1): 얼굴 단위 웃음, 눈 지표에서 제외. 사진 단위는 두 가지로 계산 (-1 얼굴이 있는 사진을 미충족으로 / 제외)
- 출력: results_{MODEL}_{WIDTH}/faces.csv (얼굴별), results/photos.csv (사진별), 터미널 요약

실행: [EVAL_SUBSET=dev] app/.venv/bin/python evaluation/wider/evaluate.py [WIDTH] [MODEL] [SCORE]
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, RESULTS, WIDER  # noqa: E402

HERE = RESULTS   # 평가 결과 위치 (evaluation/paths.py)
sys.path.insert(0, str(ROOT / "app"))

from photo_app import config  # noqa: E402
from photo_app.models import create_model  # noqa: E402

WIDTH = int(sys.argv[1]) if len(sys.argv) > 1 else 640
MODEL = sys.argv[2] if len(sys.argv) > 2 else config.MODEL
SCORE = float(sys.argv[3]) if len(sys.argv) > 3 else None   # yunet_landmark 의 YuNet 신뢰도 기준. None 이면 설정값
IOU_TH = 0.3
SUBSET = os.environ.get("EVAL_SUBSET", "")                   # dev | holdout | 빈 값 (전부)
OUT = HERE / (f"results_{MODEL}_{WIDTH or 'orig'}" + (f"_score{SCORE}" if SCORE is not None else "")
              + (f"_{SUBSET}" if SUBSET else ""))


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def num(v):
    """예측 값 -> CSV 값. None(미검출, 판정 불가)은 빈 칸"""
    return "" if v is None else int(v)


def done(r):
    """얼굴 라벨 완료 여부"""
    if r["visible"] == "0":
        return r["hidden_reason"] != ""
    return r["visible"] == "1" and r["smile"] != "" and r["eyes_open"] != ""


def label_ok(r):
    """정답 기준 조건 충족 얼굴: 보임, 웃음, 눈 뜸(안경류 포함)"""
    return r["visible"] == "1" and r["smile"] == "1" and r["eyes_open"] in ("1", "2")


def has_unknown(faces):
    """보이는 얼굴 중 웃음 또는 눈이 판단 불가(-1)인 얼굴이 있는지. 이런 사진은 정답(조건 충족 여부)을 모름"""
    return any(r["visible"] == "1" and "-1" in (r["smile"], r["eyes_open"]) for r in faces)


def load():
    unusable = {r["image"] for r in csv.DictReader(open(LABELS / "images.csv")) if r["unusable"] == "1"}
    by_image = defaultdict(list)
    for r in csv.DictReader(open(LABELS / "labels.csv")):
        by_image[r["image"]].append(r)
    keep = None
    if SUBSET:
        if SUBSET not in ("dev", "holdout"):
            raise ValueError(f"EVAL_SUBSET 은 dev 또는 holdout: {SUBSET}")
        keep = {r["image"] for r in csv.DictReader(open(LABELS / "split.csv")) if r["subset"] == SUBSET}
    return {img: faces for img, faces in by_image.items()
            if img not in unusable and all(done(f) for f in faces) and (keep is None or img in keep)}


def match(gts, preds):
    """IoU 큰 순 1:1 매칭 -> {정답 번호: (예측 번호, IoU)}"""
    pairs = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gts) for pi, p in enumerate(preds)), reverse=True)
    used_g, used_p, out = set(), set(), {}
    for v, gi, pi in pairs:
        if v < IOU_TH:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out[gi] = (pi, v)
    return out


def main():
    images = load()
    if SCORE is not None:
        from photo_app.models.yunet_landmark import YunetLandmarkModel
        model = YunetLandmarkModel(score_th=SCORE)
    else:
        model = create_model(MODEL)
    OUT.mkdir(exist_ok=True)
    face_rows, photo_rows = [], []
    for n, (path, labels) in enumerate(images.items(), 1):
        img = cv2.imread(str(WIDER / path))
        scale = WIDTH / img.shape[1] if WIDTH else 1.0
        if WIDTH:
            img = cv2.resize(img, (WIDTH, round(img.shape[0] * scale)))
        pred = model.predict(img, smile=True, eye=True, hand=False, visibility=True)
        gts = [tuple(int(r[k]) * scale for k in ("x", "y", "w", "h")) for r in labels]
        m = match(gts, [f.bbox for f in pred.faces])
        for gi, r in enumerate(labels):
            pi, v = m.get(gi, (None, 0.0))
            f = pred.faces[pi] if pi is not None else None
            face_rows.append({
                "image": path, "face_id": r["face_id"], "h_px": round(gts[gi][3]),
                "occlusion": r["occlusion"], "visible": r["visible"], "hidden_reason": r["hidden_reason"],
                "smile": r["smile"], "eyes_open": r["eyes_open"],
                "detected": int(f is not None), "iou": round(v, 3),
                "pred_smile": num(f and f.smile), "pred_eyes_open": num(f and f.eyes_open),
            })
        photo_rows.append({
            "image": path, "faces": len(labels), "pred_faces": len(pred.faces),
            "unmatched_pred": len(pred.faces) - len(m),
            "label_ok": int(all(label_ok(r) for r in labels)), "pred_ok": int(pred.meets_condition()),
            "unknown": int(has_unknown(labels)),
            "elapsed_ms": round(pred.elapsed_ms, 1),
        })
        print(f"\r{n}/{len(images)}", end="", flush=True)
    print()
    for name, rows in (("faces.csv", face_rows), ("photos.csv", photo_rows)):
        with open(OUT / name, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    summary(face_rows, photo_rows)


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else f"{a}/0"


def binary(rows, label_key, pred_key, pos):
    """라벨 vs 예측 혼동 행렬. pos: 라벨에서 양성으로 볼 값들"""
    tp = sum(1 for r in rows if r[label_key] in pos and r[pred_key] == 1)
    fn = sum(1 for r in rows if r[label_key] in pos and r[pred_key] == 0)
    fp = sum(1 for r in rows if r[label_key] not in pos and r[pred_key] == 1)
    tn = sum(1 for r in rows if r[label_key] not in pos and r[pred_key] == 0)
    return tp, fn, fp, tn


def show(title, tp, fn, fp, tn, pos_name, neg_name):
    print(f"\n[{title}] 정확도 {pct(tp + tn, tp + fn + fp + tn)}")
    print(f"  라벨 {pos_name}: 예측 {pos_name} {tp}, 예측 {neg_name} {fn}")
    print(f"  라벨 {neg_name}: 예측 {pos_name} {fp}, 예측 {neg_name} {tn}")
    print(f"  {pos_name} precision {pct(tp, tp + fp)}, recall {pct(tp, tp + fn)}")
    print(f"  {neg_name} precision {pct(tn, tn + fn)}, recall {pct(tn, tn + fp)}")


def summary(faces, photos):
    print(f"\n입력 가로 {WIDTH or '원본'} px, IoU >= {IOU_TH}, 모델 {MODEL}" + (f", YuNet score >= {SCORE}" if SCORE is not None else ""))
    print(f"사진 {len(photos)}장, 정답 얼굴 {len(faces)}개")

    print("\n[얼굴 검출률]")
    print(f"  전체 {pct(sum(r['detected'] for r in faces), len(faces))}")
    for name, cond in (("보임", lambda r: r["visible"] == "1"), ("보이지 않음", lambda r: r["visible"] == "0")):
        rs = [r for r in faces if cond(r)]
        print(f"  {name} {pct(sum(r['detected'] for r in rs), len(rs))}")
    for lo, hi in ((0, 40), (40, 60), (60, 100), (100, 10**9)):
        rs = [r for r in faces if r["visible"] == "1" and lo <= r["h_px"] < hi]
        print(f"  보임, 얼굴 높이 {lo}~{hi if hi < 10**9 else ''}px {pct(sum(r['detected'] for r in rs), len(rs))}")
    print(f"  짝 없는 예측 얼굴 (정답 밖) {sum(p['unmatched_pred'] for p in photos)}개")

    seen = [r for r in faces if r["detected"] and r["visible"] == "1"]
    show("웃음 (검출된 보임 얼굴, 판단 불가 제외)",
         *binary([r for r in seen if r["smile"] in ("0", "1")], "smile", "pred_smile", ("1",)), "웃음", "안 웃음")
    show("눈 뜸 (검출된 보임 얼굴, 판단 불가 제외, 안경류는 뜸)",
         *binary([r for r in seen if r["eyes_open"] in ("0", "1", "2")], "eyes_open", "pred_eyes_open", ("1", "2")),
         "뜸", "감음")
    glasses = [r for r in seen if r["eyes_open"] == "2"]
    print(f"  안경류 얼굴 중 예측 뜸 {pct(sum(r['pred_eyes_open'] == 1 for r in glasses), len(glasses))}")

    print("\n[판정 불가 (검출됐지만 웃음, 눈 결과 None)]")
    for name, v in (("보임", "1"), ("보이지 않음", "0")):
        rs = [r for r in faces if r["detected"] and r["visible"] == v]
        print(f"  {name} {pct(sum(r['pred_smile'] == '' for r in rs), len(rs))}")

    hidden = [r for r in faces if r["detected"] and r["visible"] == "0"]
    ok = sum(1 for r in hidden if r["pred_smile"] == 1 and r["pred_eyes_open"] == 1)
    print(f"\n[보이지 않는 얼굴이 검출된 경우] 웃음+눈 뜸으로 판정 {pct(ok, len(hidden))}")

    for name, ps in (("판단 불가(-1) 얼굴이 있는 사진은 미충족", photos),
                     ("판단 불가(-1) 얼굴이 있는 사진 제외", [p for p in photos if not p["unknown"]])):
        tp = sum(1 for p in ps if p["label_ok"] and p["pred_ok"])
        fn = sum(1 for p in ps if p["label_ok"] and not p["pred_ok"])
        fp = sum(1 for p in ps if not p["label_ok"] and p["pred_ok"])
        tn = sum(1 for p in ps if not p["label_ok"] and not p["pred_ok"])
        show(f"사진 조건 충족 (meets_condition), {name}, 사진 {len(ps)}장", tp, fn, fp, tn, "충족", "미충족")

    ms = sorted(p["elapsed_ms"] for p in photos)
    print(f"\n[추론 시간] 중앙값 {ms[len(ms) // 2]} ms, 최대 {ms[-1]} ms (이 PC 기준)")


if __name__ == "__main__":
    main()
