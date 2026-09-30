"""WIDER FACE 에서 단체 사진 후보를 골라 라벨링용 CSV 생성
- 실행 (저장소 루트에서): python3 evaluation/wider/make_label_sheet.py
- 입력: evaluation/work/wider_face/wider_face_split/wider_face_{val,train}_bbx_gt.txt (WIDER 원본 배포의 라벨)
- 분할별 사용 (SPLITS)
  - val: 후보 전부
  - train: 행사 비율대로 TRAIN_SAMPLE 장 무작위 추출
    - 행사마다 고정 시드(SEED)로 섞은 순서의 앞에서부터. 장수를 늘려도 이미 뽑힌 사진은 유지
- 출력: evaluation/wider/labels/labels.csv (얼굴 1개 = 1행. visible, hidden_reason, smile, eyes_open 은 빈 칸)
- 후보 기준
  - 행사: EVENTS
  - 인원 MIN_PEOPLE ~ MAX_PEOPLE (WIDER 박스 수 기준, invalid 포함)
  - 유효(invalid=0) 얼굴의 높이가 모두 MIN_FACE_H px 이상
- 추가 모드: labels.csv 가 있으면 기존 행(라벨 포함)은 그대로 두고, 아직 없는 사진의 얼굴만 뒤에 추가
  - 쓰기 전 기존 파일을 labels.backup.csv 로 복사
"""
import csv
import random
import shutil
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, WIDER  # noqa: E402

SPLITS = {   # 분할 -> (라벨 파일, 사진 폴더(WIDER 폴더 기준), 추출 장수. None 이면 전부)
    "val": (WIDER / "wider_face_split" / "wider_face_val_bbx_gt.txt", "WIDER_val/images", None),
    "train": (WIDER / "wider_face_split" / "wider_face_train_bbx_gt.txt", "WIDER_train/images", None),
}
SEED = 0
OUT = LABELS / "labels.csv"

EVENTS = (
    "12--Group", "20--Family_Group", "19--Couple",                   # 1차 (114장)
    "49--Greeting", "50--Celebration_Or_Party", "29--Students_Schoolkids",
    "28--Sports_Fan", "16--Award_Ceremony", "1--Handshaking", "22--Picnic",   # 2차 추가
)
MIN_PEOPLE, MAX_PEOPLE = 2, 10
MIN_FACE_H = 40

# WIDER 속성 순서: x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
ATTRS = ("x", "y", "w", "h", "blur", "expression", "illumination", "invalid", "occlusion", "pose")
FIELDS = ("image", "face_id") + ATTRS + ("target", "visible", "hidden_reason", "smile", "eyes_open")


def parse(path):
    """[(사진 경로, [얼굴 속성 dict, ...]), ...]
    - 얼굴 0개인 사진도 속성 줄이 1줄(0 ...) 있음
    """
    lines = path.read_text().split("\n")
    i, out = 0, []
    while i < len(lines) and lines[i].strip():
        name, n = lines[i].strip(), int(lines[i + 1])
        rows = lines[i + 2:i + 2 + max(n, 1)]
        faces = [dict(zip(ATTRS, map(int, r.split()))) for r in rows] if n else []
        out.append((name, faces))
        i += 2 + max(n, 1)
    return out


def is_candidate(name, faces):
    if not name.startswith(tuple(e + "/" for e in EVENTS)):
        return False
    if not MIN_PEOPLE <= len(faces) <= MAX_PEOPLE:
        return False
    return all(f["h"] >= MIN_FACE_H for f in faces if f["invalid"] == 0)


def sample_by_event(cands, n):
    """행사 비율대로 n 장 추출
    - 행사마다 고정 시드로 섞은 순서의 앞에서부터 (n 을 늘리면 이어서 뽑힘)
    - 행사별 장수: round(n * 행사 후보 수 / 전체 후보 수)
    """
    by_event = {}
    for name, faces in cands:
        by_event.setdefault(name.split("/")[0], []).append((name, faces))
    total = len(cands)
    out = []
    for ev in EVENTS:
        items = sorted(by_event.get(ev, []), key=lambda x: x[0])
        random.Random(f"{SEED}:{ev}").shuffle(items)
        out += items[:round(n * len(items) / total)]
    return out


def main():
    existing = list(csv.DictReader(open(OUT, encoding="utf-8"))) if OUT.exists() else []
    have = {r["image"] for r in existing}
    picked = []   # [(사진 경로, 얼굴 목록), ...]
    for split, (gt, image_dir, n) in SPLITS.items():
        if not gt.exists():
            continue
        cands = [(name, faces) for name, faces in parse(gt) if is_candidate(name, faces)]
        chosen = cands if n is None else sample_by_event(cands, n)
        picked += [(f"{image_dir}/{name}", faces) for name, faces in chosen if f"{image_dir}/{name}" not in have]
    missing = [p for p, _ in picked if not (WIDER / p).exists()]
    if missing:
        raise SystemExit(f"사진 파일 없음 {len(missing)}장 (예: {missing[0]}). 압축을 먼저 풀어야 함")
    if OUT.exists():
        shutil.copy(OUT, OUT.with_name("labels.backup.csv"))
    new_rows = []
    for path, faces in picked:
        for i, face in enumerate(faces):
            # target: 라벨링 대상 (유효 얼굴). invalid 얼굴은 0 으로 두고 라벨 생략 가능
            new_rows.append({"image": path, "face_id": i, **face,
                             "target": int(face["invalid"] == 0), "visible": "", "hidden_reason": "", "smile": "", "eyes_open": ""})
    with open(OUT, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in existing)
        w.writerows(new_rows)
    by_split_event = {}
    for path, _ in picked:
        key = (path.split("/")[3], path.split("/")[5])   # (WIDER_val|WIDER_train, 행사)
        by_split_event[key] = by_split_event.get(key, 0) + 1
    print(f"기존 사진 {len(have)}장 유지, 추가 사진 {len(picked)}장, 추가 얼굴 {len(new_rows)}개 -> {OUT}")
    for k, c in sorted(by_split_event.items()):
        print(" ", k, c)


if __name__ == "__main__":
    main()
