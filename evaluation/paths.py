"""evaluation 스크립트가 쓰는 경로
- REPO: 저장소 루트
- LABELS: WIDER 평가 라벨 (evaluation/wider/labels. git 에 있음). image 열은 WIDER 폴더 기준 경로
- WORK: 로컬 작업 폴더 (git 에 올리지 않음). 환경변수 YLYL_EVAL_WORK 로 바꿀 수 있음. 기본 evaluation/work
  - WIDER = WORK/wider_face: WIDER FACE 사진 (팀 드라이브의 평가 사진 zip 을 풂). 환경변수 WIDER_DIR 로 바꿀 수 있음
  - RESULTS = WORK/results: 평가 결과
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LABELS = REPO / "evaluation" / "wider" / "labels"
WORK = Path(os.environ.get("YLYL_EVAL_WORK", REPO / "evaluation" / "work"))
WIDER = Path(os.environ.get("WIDER_DIR", WORK / "wider_face"))
RESULTS = WORK / "results"
