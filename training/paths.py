"""training 스크립트가 쓰는 경로
- REPO: 저장소 루트
- WORK: 데이터와 결과를 두는 로컬 폴더 (git 에 올리지 않음). 환경변수 YLYL_WORK 로 바꿀 수 있음. 기본 training/work
  - DATASET = WORK/dataset: 학습 데이터 (smile, eye, occlusion). 폴더 구조는 training/data 와 같음. 팀 드라이브 Team-11/dataset 에서 받음
- MODELS: 앱 모델 폴더 (app/models)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("YLYL_WORK", REPO / "training" / "work"))
DATASET = WORK / "dataset"
MODELS = REPO / "app" / "models"
