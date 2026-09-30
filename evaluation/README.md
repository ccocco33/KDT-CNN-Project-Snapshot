# evaluation

앱 모델 평가: WIDER FACE 단체 사진 정확도와 라즈베리파이 속도, CPU/메모리, 성능.
학습 노트북, 학습 데이터는 `training/`, 모델 버전별 결과는 `docs/model_versions.md`, 데이터 출처는 `data/README.md`.

## 폴더 구조

```
evaluation/
  paths.py          스크립트가 쓰는 경로 (라벨, WIDER 사진, 결과)
  wider/            WIDER 단체 사진 정확도 평가
    labels/         평가 라벨 (git)
  speed/            라즈베리파이 속도, CPU/메모리 (measure_speed.py), 결과 보기 (show_results.py)
    results/        기기별 측정 결과 (git)
  work/             로컬 작업 폴더 (git 에 올리지 않음)
    wider_face/     WIDER 평가 사진 (드라이브 zip 을 풂)
    results/        WIDER 평가 결과
```

- 실행은 저장소 루트에서 앱 가상환경으로: `app/.venv/bin/python evaluation/...`
- 다른 위치를 쓰려면 환경변수: `YLYL_EVAL_WORK` (work 폴더), `WIDER_DIR` (WIDER 사진 폴더)

## 준비: WIDER 평가 사진

1. 팀 드라이브 `Team-11/dataset/wider/wider_eval_photos.zip` (평가 사진 1,040장, 115MB) 을 받음
2. `evaluation/work/wider_face/` 에 풂 (zip 안이 `WIDER_val/images/...`, `WIDER_train/images/...`)
   ```
   mkdir -p evaluation/work/wider_face
   unzip wider_eval_photos.zip -d evaluation/work/wider_face
   ```
- zip 은 `wider/make_photos_zip.py` 로 만듦 (WIDER 원본 폴더에서 라벨에 있는 사진만 묶음)
- WIDER FACE 는 CC BY-NC-ND: 팀 내부 평가에만 쓰고 사진, 라벨, 결과 그림을 외부에 공개하지 않음

## 라즈베리파이 측정: 속도, CPU/메모리, 성능 (`speed/`)

### 측정할 것 (목표는 `docs/dev_spec.md` 9절)

| 항목 | 목표 | 측정 |
|---|---|---|
| 판정 지연 (predict 1회, 단계별) | 200 ms 이하 | 1. 속도 |
| 판정 FPS | 5 이상 | 1. 속도 (1000 / 평균), 3. 앱 로그 |
| 인원 수별 판정 지연, `WARN_PEOPLE` (200 ms 를 넘는 가장 적은 인원) | - | 1. 속도 |
| 베이스라인 (yunet_landmark) 대비 속도, 세부 시간 (YuNet, 손, FaceLandmarker, CNN) | - | 1. 속도 |
| CNN 형식 (fp32, dynamic, int8) 별 속도 | - | 1. 속도 |
| CPU 사용률, 메모리 사용량 | 측정 후 결정 | 1. 속도 |
| 발열에 따른 느려짐 (CPU 온도, 스로틀) | 스로틀 없음 | 1. 속도 |
| 성능 (정확도): Mac 과 같은 판정인지, CNN 형식별 AUC | Mac 결과와 같음 | 2. 성능 |
| 카메라 FPS | 20 이상 | 3. 앱 로그 |
| 화면 FPS | 15 이상 | 3. 앱 로그 |
| 촬영 장수 (CAPTURE 한 번) | 40장 이상 | 3. 앱 로그 |
| CHECKING 대기 (촬영 끝 ~ 결과 표시) | 1초 이하 | 3. 앱 로그 |

- 1 은 모델만 따로 잼 (같은 입력으로 모델끼리 비교). 앱에서는 카메라, 화면, 판정 스레드가 CPU 를 나눠 써 1 보다 느릴 수 있어 3 으로 확인
- 2 는 같은 tflite 라도 CPU (ARM), 라이브러리 버전에 따라 점수가 조금 다를 수 있어 확인. 다르면 Mac (WIDER dev) 에서 정한 기준값 (`SMILE_TH`, `EYE_TH`, `OCC_TH`) 을 그대로 쓸 수 없음

### 준비 (라즈베리파이)

1. 저장소를 받고 앱 가상환경을 만듦 (`app/README.md`). 모델 파일은 저장소에 있음 (`app/models/`, 현재 v0.0.7)
2. WIDER 평가 사진을 `evaluation/work/wider_face/` 에 풂 (위 "준비: WIDER 평가 사진")
3. 방열판, 팬, 전원 상태를 기록해 둠 (결과 폴더 이름의 `--tag` 로)
4. 측정 중에는 다른 프로그램을 끔 (브라우저 등)

### 1. 속도, CPU, 메모리 (`measure_speed.py`)

```
cd <저장소 루트>
app/.venv/bin/python evaluation/speed/measure_speed.py --tag fan                  # WIDER 사진 100장, 3회 반복
app/.venv/bin/python evaluation/speed/measure_speed.py --source camera --tag fan  # 카메라 프레임 100장 (카메라 앞에 사람이 있을 때)
```

- 앱의 단계별 판정 요청 그대로 `Model.predict` 를 호출해 시간을 잼 (사진 읽기, 리사이즈, 화면 그리기는 제외)
  - PREVIEW: 보임 + 손 들기 / PREPARE: 웃음 / CHECKING: 보임 + 웃음 + 눈
  - 세부 시간: `face` (YuNet + 얼굴 자르기), `landmark` (얼굴마다 자르기 + FaceLandmarker), `occlusion`, `smile`, `eye` (자르기 + CNN), `hand` (HandLandmarker)
- 모델 (기본 4개)
  - `yunet_landmark` (베이스라인)
  - `yunet_cnn_fp32` (앱 그대로), `yunet_cnn_dynamic`, `yunet_cnn_int8` (CNN 만 `app/models/{MODEL_VERSION}/*_{dynamic,int8}.tflite` 로 바꿈. int8 은 XNNPACK 을 쓰지 못함)
- 입력
  - `wider` (기본): WIDER dev 사진을 카메라 가로 크기 (640) 로 줄임. 1 ~ 10명 이상으로 인원이 다양해 인원 수별 지연을 볼 수 있음
  - `camera`: 실제 카메라 프레임. 먼저 모두 찍어 두고 같은 프레임으로 모델끼리 잼
- CPU, 메모리: 단계를 도는 동안 0.2초마다 기록
  - 프로세스 CPU %: 여러 코어를 쓰면 100 을 넘음 (4코어면 최대 400). 코어 %: 코어별 사용률 평균 (다른 프로그램 포함)
  - 프로세스 메모리 (RSS), 시스템 사용 메모리
  - 주의: 한 프로세스에 모델을 모두 올려 두고 재므로 RSS 는 올린 모델 전부의 합. 모델 하나의 메모리는 하나만 따로 실행 (`--models yunet_cnn --quant fp32`)
- 반복마다 모델 순서를 바꿈 (발열 영향을 나눔). 모델, 단계마다 처음 10장은 워밍업으로 버림
- 옵션: `--models yunet_cnn` (한 모델만), `--quant fp32 int8` (형식 고르기), `--frames`, `--repeat`, `--warmup`, `--width`

### 2. 성능 (정확도)

```
cd <저장소 루트>
H=$(hostname)
for s in dev holdout; do
  rm -rf evaluation/work/results/results_models_$s       # 이전 점수를 쓰지 않도록
  EVAL_SUBSET=$s app/.venv/bin/python evaluation/wider/compare_models.py | tee evaluation/speed/results/${H}_compare_models_$s.log
done
app/.venv/bin/python evaluation/wider/bench_quant.py $H | tee evaluation/speed/results/${H}_bench_quant.log
```

- `compare_models.py`: 앱 설정 그대로 사진 조건 충족 precision, recall 과 얼굴 판정 (yunet_landmark vs yunet_cnn fp32)
  - Mac 결과: `docs/model_versions.md` v0.0.7 "사진 단위" (dev precision 70/77, recall 70/143 / holdout 49/60, 49/126)
- `bench_quant.py`: CNN 형식 (fp32, dynamic, int8) 별 AUC 와 fp32 대비 점수 차이 (WIDER dev + holdout)
- 결과 로그를 `speed/results/` 에 남겨 git 에 올림

### 3. 앱 로그 (카메라, 화면, 촬영, CHECKING)

```
cd app
.venv/bin/python -m photo_app       # 손 들기 -> 촬영 -> 결과를 5회 이상. 인원 수를 바꿔 가며 (1, 2, 4명 ...)
.venv/bin/python -m analysis        # logs/*.jsonl 요약 (카메라 FPS, 화면 FPS, 판정 시간, 촬영, CHECKING)
```

- 앱 모델은 `app/photo_app/config.py` 의 `MODEL` (기본 `yunet_cnn`). 베이스라인과 비교하려면 `yunet_landmark` 로 바꿔 한 번 더
- 로그 형식, 요약 표: `docs/dev_spec.md` 9절

### 결과 보기

```
python3 evaluation/speed/show_results.py                                         # results/ 전부
python3 evaluation/speed/show_results.py evaluation/speed/results/<폴더>         # 지정
```

- 1 의 결과 폴더별로
  - 기기, 입력, 반복, 측정 전후 CPU 온도와 스로틀, 모델 설정 (CNN 형식, XNNPACK 사용 여부)
  - 단계, 모델별 사진당 중앙값, 평균, p90, 판정 FPS, 목표 충족 (판정 지연 200 ms 이하, 판정 FPS 5 이상) o/x, CPU %, 코어 %, RSS, 시스템 메모리
  - 인원 수 구간별 사진당 중앙값과 `WARN_PEOPLE`
- 결과 폴더 (`speed/results/{기기 이름}_{시각}[_{tag}]/`)
  - `summary.json`: 기기 (모델명, CPU 수, 메모리, 패키지 버전), 설정, 단계별 통계 (시간, 세부 시간, CPU, 메모리), 온도와 스로틀
  - `frames.csv`: 프레임별 시간 (반복, 모델, 단계, 얼굴 수, 전체 ms, 세부 ms)
- 2 의 결과는 `speed/results/{기기 이름}_*.log`
- 결과는 git 에 올려 기기, 설정끼리 비교
- 스로틀이 생기면 (`throttled=0x0` 이 아님) 방열, 전원을 확인하고 다시 측정
- `colima_*`: 비교용 Mac Docker (colima, 4 CPU, 8GB) 측정 (CPU, 메모리, CNN 형식 기록 전 버전). 컨테이너 시계가 UTC 라 폴더 시각이 9시간 이름

## WIDER 단체 사진 정확도 (`wider/`)

- 라벨 (`wider/labels/`, image 열은 WIDER 폴더 기준 경로)
  - `labels.csv`: 얼굴별 보임, 가린 이유, 웃음, 눈 뜸 (`label_tool.py` 로 라벨링)
  - `split.csv`: dev / holdout (`make_split.py`). 기준값은 dev 로 고르고 holdout 은 확인용
  - `images.csv`: 사진별 사용 부적절 표시, `recheck_occluded.csv`: 보임 기준 변경 뒤 재검수한 얼굴
  - 처음 라벨 시트: `make_label_sheet.py` (WIDER 원본 배포의 `wider_face_split/` 필요)
- 스크립트 (모두 `evaluate.py` 의 라벨 읽기, 얼굴 짝짓기를 씀. `EVAL_SUBSET=dev|holdout`)

| 스크립트 | 내용 |
|---|---|
| `compare_models.py` | 앱 설정 그대로 yunet_landmark (베이스라인) vs yunet_cnn: 사진 조건 충족 precision, recall, 얼굴 판정 |
| `compare_models_stage.py` | 단계별 (보임, 웃음, 눈) AUC 와 추론 시간 (이 PC) |
| `version_auc.py` | CNN 버전별 smile, eye AUC (얼굴 높이, 흑백 구간별) |
| `eye_retrain_compare.py`, `smile_retrain_compare.py` | 새로 학습한 모델 vs 앱 모델 (AUC, 기준값별 판정, 선글라스 얼굴) |
| `occlusion_cnn_eval.py`, `occlusion_yunet.py` | 가림 판정 |
| `app_captures_retrain.py` | 앱 dev 모드 촬영 사진 (`app/photos/eval/`) 에 두 모델 점수. 증상 확인용 (기준값 근거로 쓰지 않음) |
| `compare_cnn.py`, `photo_compare.py`, `requirements_check.py`, `box_height_check.py`, `app_setting_table.py`, `bench_quant.py` | 이전 버전 비교, 요구사항 확인, 양자화 속도 |

- 결과는 `work/results/` 에 씀. `compare_models.py` 는 `work/results/results_models_{dev,holdout}/faces.csv` 가 있으면 점수를 다시 계산하지 않고 요약만 냄 (모델을 바꿨으면 폴더를 지우고 실행)
