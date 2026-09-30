# training

CNN 분류 모델(smile, eye, occlusion)의 학습 노트북과 학습 데이터 준비 스크립트.
평가 (WIDER 단체 사진, 라즈베리파이 속도) 는 `evaluation/`, 모델 버전별 결과는 `docs/model_versions.md`, 데이터 출처는 `data/README.md`.

## 폴더 구조

```
training/
  paths.py        스크립트가 쓰는 경로 (저장소 루트, 로컬 작업 폴더)
  notebooks/      학습 노트북 (Colab) 과 생성 스크립트
  data/           학습 데이터 준비 스크립트 (smile, eye, occlusion)
  work/           로컬 작업 폴더 (git 에 올리지 않음. 학습 데이터)
```

- 스크립트는 데이터를 `work/` 에서 읽고 결과도 `work/` 에 씀 (`paths.py`)
  - `work/dataset/`: 학습 데이터. 폴더 구조는 `data/` 와 같음 (예: `data/eye/levels/make_levels.py` 는 `work/dataset/eye/levels/` 를 씀)
  - 다른 곳을 쓰려면 환경변수 `YLYL_WORK` 로 지정
- 실행은 앱 가상환경으로: `app/.venv/bin/python training/...`
  - `data/occlusion/cofw_extract.py`, `cofw_check.py` 는 h5py 가 더 필요

## 학습 노트북

| 모델 | 노트북 | 생성 스크립트 |
|---|---|---|
| smile | `notebooks/smile_train.ipynb` | `notebooks/build_smile_nb.py` |
| eye | `notebooks/eye_train.ipynb` | `notebooks/build_eye_nb.py` |
| occlusion | `notebooks/occlusion_train.ipynb` | `notebooks/build_occlusion_nb.py` |

- 노트북은 생성 스크립트로 만든 파일. 고칠 때는 생성 스크립트를 고치고 다시 만듦
  ```
  python training/notebooks/build_eye_nb.py training/notebooks/eye_train.ipynb training/data/eye/augment_review/eyewear_fn.py
  ```
  - 두 번째 인자: 합성 안경류 함수 (`data/eye/augment_review/eyewear_fn.py`). 노트북 셀에 그대로 들어감
- Colab 에서 실행. 데이터는 팀 드라이브 `Team-11/dataset/` 에서 읽고 결과는 `Team-11/models/{smile,eye,occlusion}/{실행 시각}/` 에 저장
  - 결과: tflite (fp32, dynamic, int8), `inference_config.json` (입력, 출력, 기준값), `metrics.json`, 학습 곡선
  - 앱에 넣는 방법은 `docs/model_versions.md` 의 "새 버전을 추가할 때"

## 학습 데이터

노트북이 읽는 팀 드라이브 파일과 그 파일을 만드는 스크립트.

| 드라이브 `Team-11/dataset/` | 로컬 `work/dataset/` | 만드는 스크립트 (`data/`) |
|---|---|---|
| `smile/preprocessed.zip` | `smile/preprocessed/` (zip 은 폴더를 묶은 것) | `smile/preprocessed/preprocess.py` (원본 `smile/raw/`, GENKI-4K) |
| `smile/teacher_scores.csv` | `smile/teacher/teacher_scores.csv` | `smile/teacher/make_teacher_scores.py` |
| `smile/eye_landmarks.csv` | `smile/landmarks/eye_landmarks.csv` | `smile/landmarks/make_eye_landmarks.py` |
| `eye/preprocessed.zip` | `eye/preprocessed_genki.zip` (v0.0.7 부터. v0.0.6 은 `preprocessed_cofw.zip`) | `eye/preprocessed/preprocess.py` (원본 `eye/raw/`) + `eye/cofw_sunglasses/preprocess_cofw.py` + `eye/genki_smile/build_dataset.py` |
| `eye/levels.csv` | `eye/levels/levels_genki.csv` (v0.0.7 부터. v0.0.6 은 `levels_cofw.csv`) | `eye/levels/review.py` (검수), `make_levels.py` + COFW 선글라스 85줄 + GENKI 웃는 눈 293줄 (`eye/genki_smile/review.py` 검수) |
| `eye/teacher_baseline.csv` (v0.0.7 부터) | `eye/levels/teacher_baseline.csv` | `eye/genki_smile/build_dataset.py` (1 - blendshape 눈 감음 최대) |
| `eye/teacher_geometry.csv` (v0.0.6 까지) | `eye/levels/teacher_geometry.csv` | `eye/levels/geometry_metrics.py`, `geometry_teacher.py` |
| `eye/eye_landmarks.csv` | `eye/landmarks/eye_landmarks.csv` | - |
| `occlusion/preprocessed.zip` | `occlusion/preprocessed.zip` | `occlusion/make_dataset.py` (COFW 추출 `cofw_extract.py`, 검수 `review.py`) |

- 원본 데이터 (GENKI-4K, eye opened/closed, COFW) 와 중간 결과는 팀 드라이브에만 둠 (크기, 라이선스)
- COFW: Burgos-Artizzu, Perona, Dollár, ICCV 2013, CC BY 4.0
- eye COFW 선글라스: `eye/cofw_sunglasses/review.py` 로 검수 (선글라스, 눈 보임 뜸, 눈 보임 감음, 제외) -> `preprocess_cofw.py` 로 eye 전처리와 같은 방식으로 64px
- 합성 안경류 미리보기: `eye/augment_review/sunglasses_v2_preview.py`, `clear_glasses_preview.py`
