# 모델 버전 기록

- CNN 분류 모델(smile, eye, occlusion)의 버전별 파일, 입출력 형식, 성능, 알려진 문제
- 모델 파일 위치: `app/models/{버전}/`. 현재 버전 (v0.0.7) 만 git 에 있음 (`.gitignore` 예외). 이전 버전은 팀 드라이브
- 드라이브 버전 (vN): 팀 드라이브 모델 폴더의 버전 폴더. 학습 실행 폴더 (`Team-11/models/{smile,eye,occlusion}/{실행 시각}/`) 를 통째로 복사해 둠
- 비교 기준(베이스라인): `yunet_landmark` 모델 (YuNet + 잘라낸 얼굴 FaceLandmarker blendshape 룰)
- 학습 노트북, 데이터 준비: `training/` (`training/README.md`). 평가 스크립트: `evaluation/` (`evaluation/README.md`). 데이터는 팀 드라이브 `Team-11/dataset/`, 출처는 `data/README.md`

## 버전 목록

| 앱 버전 | smile | eye | 날짜 | 비고 |
|---|---|---|---|---|
| v0.0.1 | 드라이브 v13 (`smile_fp32.tflite`) | 드라이브 v10 (`eye_float32.tflite`) | 2026-09-25 | 첫 CNN 모델 |
| v0.0.2 | 드라이브 (v15), 학습 20260926-0411 (`smile_fp32.tflite`) | 드라이브 (v16), 학습 20260926-0514 (`eye_fp32.tflite`) | 2026-09-26 | 전처리 데이터, 지식 증류, EMA, 좌우 뒤집기 평균. eye 는 합성 선글라스 증강 |
| v0.0.3 | v0.0.2 와 같음 | 드라이브 (v17), 학습 20260926-1209 (`eye_fp32.tflite`) | 2026-09-26 | eye 재학습: 투명 안경 증강, 4단계 눈 뜸 정도 라벨, 기하 지표 교사 |
| v0.0.4 | v0.0.3 와 같음 | v0.0.3 와 같음 | 2026-09-27 | occlusion(가림) CNN 추가: 드라이브 (v18), 학습 20260926-1611 (`occlusion_fp32.tflite`) |
| v0.0.5 | v0.0.3 와 같음 | v0.0.3 와 같음 | 2026-09-27 | occlusion 재학습: 드라이브 (v19), 학습 20260926-1728. 피부색 가림, 두 눈 가림 추가 |
| v0.0.6 | 드라이브 (v20), 학습 20260926-1958 (`smile_fp32.tflite`) | 드라이브 (v21), 학습 20260926-2040 (`eye_fp32.tflite`) | 2026-09-27 | 선글라스 렌즈 반사 대응. smile 입력을 얼굴 아래쪽 0.45 로, eye 학습에 COFW 실제 선글라스 85장 추가. occlusion 은 v0.0.5 와 같음 |
| v0.0.7 | v0.0.6 와 같음 | 드라이브 (v22), 학습 20260927-0023 (`eye_fp32.tflite`) | 2026-09-27 | eye 재학습: 웃는 눈 목표 1 (0.7 에서), GENKI 웃는 눈 293장 추가, 베이스라인 교사 (1 - blendshape 눈 감음). smile, occlusion 은 v0.0.6 와 같음 |

## v0.0.1

### smile (드라이브 v13)

- 파일: `app/models/v0.0.1/smile_fp32.tflite` (3,732,900 bytes)
  - 2026-09-27 드라이브 v12 에서 v13 으로 바꿈 (thinkcat). v12 는 `smile_fp32_v12.tflite` 로 보관
- 학습 노트북: `smile_training.ipynb` (팀 제공)
- 모델: MobileNetV3Small (ImageNet 사전학습) + 전역 평균 풀링 + 이진 분류
- 학습 데이터: GENKI-4K (64x64)
- 입력: `face_rgb_0_255`, 128x128, RGB, 0~255
- 출력: 안 웃음 점수 (sigmoid, **1 = 안 웃음**)
- 성능

| 데이터 | v13 AUC | v12 AUC | 비고 |
|---|---|---|---|
| 팀 GENKI test (600 / 600, 이미지 전체를 128 로) | 0.943 | 0.895 | `tmp/v001_test/measure_smile_file.py` |
| WIDER dev (가로 640, 보이는 얼굴) | 0.848 | 0.800 | `tmp/group_photo_eval/smile_runs_auc.py` |
| WIDER holdout | 0.831 | 0.790 | 베이스라인 blendshape 0.933 |

- v12 노트북 기록: GENKI 검증 0.935, 시험 0.921
- 알려진 문제
  - 출력 방향이 "안 웃음"이라 웃음 판정에 쓸 때 뒤집어야 함
  - 학습 test 와 단체 사진의 차이가 큼 (v13 -0.112, v12 -0.105. v0.0.2 는 -0.071)
  - 아래는 v12 로 확인한 것
  - 단체 사진에서 흑백 얼굴을 웃음으로 오판 (60%, 컬러 16%)
  - 기울어진 얼굴, 40px 미만 얼굴(AUC 0.80)에서 오판 증가
  - 학습 증강이 약함 (회전 약 ±9도, 흐림과 저해상도 증강 없음), 미세조정이 끝까지 수렴하지 않음
  - int8 변환 실패 (XNNPACK 준비 오류)

### eye (드라이브 v10)

- 파일: `app/models/v0.0.1/eye_float32.tflite` (1,922,820 bytes)
- 학습 노트북: 확인 못 함
- 학습 데이터: opened 1,498장 (161x161, LFW 계열로 추정), closed 1,192장 (크기 여러 가지, CEW 계열로 추정)
- 입력: `upper_face_rgb_0_255`, 160x160, RGB, 0~255
  - 얼굴 정사각형 크롭의 위쪽 60% 를 160x160 으로 늘려 넣음 (추정. 예시 이미지와 단체 사진에서 가장 잘 맞은 방식)
- 출력: 눈 뜸 점수 (sigmoid, **1 = 뜸**)
- 성능

| 데이터 | AUC | 비고 |
|---|---|---|
| 예시 이미지 (opened, closed) | 0.99 | 학습에 쓴 이미지일 수 있어 과대평가 가능 |
| WIDER 단체 사진 (평가 라벨) | 0.73 | 베이스라인 blendshape 0.93 |

- 알려진 문제
  - 입력 형식(위쪽 비율, 늘림 여부)이 추정값. 학습 코드로 확인 필요
  - 단체 사진에서 40px 미만 얼굴(AUC 0.55)과 100px 이상 큰 얼굴에서 오판 증가
  - 학습 데이터의 클래스별 치우침: closed 가 더 선명, 얼굴이 더 꽉 참, 흑백 비율 높음 (조건값만으로 클래스 구분 AUC 0.977)
    - 치우침을 줄인 전처리 데이터: `training/data/eye/preprocessed/preprocess.py` 로 만든 드라이브 `Team-11/dataset/eye/preprocessed.zip` (종합 AUC 0.891)

## v0.0.2

- 공통
  - 모델: MobileNetV3Small (ImageNet 사전학습), 한 단계 학습 60 epoch, EMA 가중치
  - 지식 증류: 정답 0.3 + 교사(FaceLandmarker blendshape 점수를 train 에서 확률로 보정) 0.7 을 섞은 목표로 학습 (DISTILL_ALPHA 0.3)
  - 좌우 뒤집기 평균(TTA)이 모델 안에 포함. 앱에서 따로 할 일 없음
  - 파일: fp32, dynamic(가중치 8bit, 입출력 float), int8. 각 모델의 `*_inference_config.json` 에 입력, 출력, 기준값
  - int8 은 XNNPACK 에서 준비 오류. `OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES` 로 불러야 함

### smile (학습 20260926-0411)

- 파일: `app/models/v0.0.2/smile_{fp32,dynamic,int8}.tflite`, `smile_inference_config.json`
- 학습 노트북: `training/notebooks/smile_train.ipynb`
- 학습 데이터: GENKI-4K 전처리본 3,774장 (드라이브 `Team-11/dataset/smile/preprocessed.zip`. YuNet 으로 다시 자름, 64x64)
- 입력: `face_rgb_0_255`, 128x128, RGB, 0~255 (YuNet 박스 중심 정사각형, 한 변 = 박스 높이)
- 출력: 웃음 점수 (sigmoid, **1 = 웃음**. v0.0.1 과 방향이 반대)
- 기준값: 0.757 (val 에서 웃음 precision >= 0.95 인 값 중 recall 최대)
- 성능

| 데이터 | AUC | 비고 |
|---|---|---|
| GENKI 전처리본 검증 | 0.977 | 노트북 기록 |
| GENKI 전처리본 시험 | 0.979 | 노트북 기록 |
| WIDER 단체 사진 dev | 0.910 | 베이스라인 blendshape 0.932, v0.0.1 0.815 (v12, 당시 라벨) |

### eye (학습 20260926-0514)

- 파일: `app/models/v0.0.2/eye_{fp32,dynamic,int8}.tflite`, `eye_inference_config.json`
- 학습 노트북: `training/notebooks/eye_train.ipynb`
- 학습 데이터: opened/closed 전처리본 2,607장 (드라이브 `Team-11/dataset/eye/preprocessed.zip` 의 COFW 추가 전 판. YuNet 으로 다시 자름, 64x64, 흑백 제외)
- 증강: 합성 선글라스 10% (라벨은 뜸). 눈 좌표 드라이브 `Team-11/dataset/eye/eye_landmarks.csv`
- 입력: `upper_face_rgb_0_255`, 160x160, RGB, 0~255 (얼굴 정사각형의 위쪽 60% 를 160x160 으로 늘림)
- 출력: 눈 뜸 점수 (sigmoid, **1 = 뜸**. 선글라스류도 뜸)
- 기준값: 0.459 (val 에서 눈 뜸 precision >= 0.95 인 값 중 recall 최대)
- 성능

| 데이터 | AUC | 비고 |
|---|---|---|
| 전처리본 검증 | 0.995 | 노트북 기록 |
| 전처리본 시험 | 0.993 | 노트북 기록. 합성, 실제 선글라스 모두 뜸 판정 |
| WIDER 단체 사진 dev | 0.954 | 베이스라인 blendshape 0.934, v0.0.1 0.73 |

- WIDER dev 세부 (가로 640, 기준 0.5)
  - 뜬 눈을 감음으로 오판: 4% (blendshape 14%). 30~50px 얼굴에서 차이가 큼 (2~6%, blendshape 22~27%)
  - 감은 눈 recall: 27/39 (blendshape 32/39)
  - 선글라스류 얼굴 뜸 판정: 36/36 (blendshape 27/30)

### smile + eye 사진 단위 (WIDER dev 515장, 정답 충족 138장)

- `evaluation/wider/photo_compare.py`. 가로 640, YuNet 0.7, yaw >= 0.5 는 보이지 않음 (조건 미충족)
- 보이는 얼굴 중 판단 불가(-1) 라벨이 있는 사진 47장은 정답을 모름 -> 미충족으로 계산 / 제외 (468장) 두 가지
- 노트북 기준값: 학습 노트북이 val 에서 precision >= 0.95 인 값 중 recall 최대로 정한 값 (smile 0.757, eye 0.459)

| 방식 | 충족 precision (-1 미충족) | 충족 precision (-1 제외) | 충족 recall (둘 다) |
|---|---|---|---|
| blendshape (베이스라인) | 49/55 (89%) | 49/55 (89%) | 49/138 (36%) |
| CNN v0.0.2, 노트북 기준값 | 56/62 (90%) | 56/61 (92%) | 56/138 (41%) |
| CNN v0.0.2, 기준 0.5 | 76/90 (84%) | 76/88 (86%) | 76/138 (55%) |

- 얼굴당 처리 시간 (Mac, fp32, 중앙값): CNN smile+eye 6.3 ms, FaceLandmarker 13.3 ms
- 알려진 문제
  - holdout 미평가 (dev 로 모델, 설정을 골랐으므로 최종 확인 필요)
  - smile 은 40px 이상 얼굴에서 blendshape 보다 낮음
  - eye 의 감은 눈 recall 이 blendshape 보다 낮음 (감은 눈이 39개뿐이라 불확실)
  - CNN 은 판정 불가(None)가 없어 옆모습, 가려진 얼굴도 판정함. yaw 규칙으로 걸러야 함
  - 잘못 통과한 사진의 대부분은 가려진 얼굴을 YuNet 이 못 찾은 경우 (분류 모델과 무관)
  - 라즈베리파이 처리 시간 미측정

## v0.0.3

- smile: v0.0.2 와 같은 파일 (`app/models/v0.0.3/smile_*` 는 v0.0.2 복사본)
- 앱 기준값 (yunet_cnn config): `SMILE_TH` 0.5, `EYE_TH` 0.8. 정한 근거는 `docs/dev_spec.md` 10절 13번, 14번

### eye (학습 20260926-1209)

- 파일: `app/models/v0.0.3/eye_{fp32,dynamic,int8}.tflite`, `eye_inference_config.json`
- 학습 노트북: `training/notebooks/eye_train.ipynb` (생성 스크립트 `training/notebooks/build_eye_nb.py`)
- 학습 데이터: v0.0.2 와 같은 전처리본 2,607장 (train 1,823 / val 392 / test 392)
- 4단계 눈 뜸 정도 라벨 (드라이브 `Team-11/dataset/eye/levels.csv` 의 COFW 추가 전 판, 사람이 검수. `training/data/eye/levels/review.py`, `make_levels.py`)
  - 1: 눈이 보임 (감정 표현으로 찌푸린 눈 포함, 선글라스 포함) 1,371장
  - 2: 웃는 눈 (^^) 126장
  - 3: 아래를 봄, 거의 감김 (--) 45장
  - 4: 확실히 감음 1,065장
  - 이진 라벨: 1, 2 는 뜸 / 3, 4 는 감음
- 학습 목표: 단계 목표값 (1: 1.0, 2: 0.7, 3: 0.3, 4: 0.0) 0.5 + 교사 점수 0.5 (DISTILL_ALPHA 0.5)
  - 교사: FaceLandmarker 랜드마크의 기하 지표 4가지 (눈 열림, 눈 위쪽 휜 정도, 입꼬리 올라감, 고개 숙임 pitch) 를 백분위로 바꿔 가중 평균 (0.4, 0.4, 0.1, 0.1) 후 sigmoid 로 보정 (train 에서 뜸 1, 2 vs 감음 3, 4 로 로지스틱 회귀)
  - 교사 파일: 드라이브 `Team-11/dataset/eye/teacher_geometry.csv` (`training/data/eye/levels/geometry_teacher.py`)
- 증강: 합성 선글라스 10% (라벨은 뜸), 투명 안경 30% (라벨 그대로. 안경테, 반사를 뜸의 단서로 쓰지 않도록)
- 입력, 출력: v0.0.2 eye 와 같음 (`upper_face_rgb_0_255`, 160x160, 출력 1 = 뜸)
- 노트북 기준값: 0.817 (val 에서 감은 눈 recall >= 0.95 인 값 중 눈 뜸 recall 최대). 앱은 0.8 사용
- 성능

| 데이터 | AUC | 비고 |
|---|---|---|
| 전처리본 검증 | 0.989 | 노트북 기록 |
| 전처리본 시험 | 0.997 | 노트북 기록 |
| WIDER 단체 사진 dev (가로 640) | 0.952 | v0.0.2 0.950, blendshape 0.932 |
| WIDER 단체 사진 holdout (가로 640) | 0.931 | v0.0.2 0.912, blendshape 0.921 |

- 단계별 점수 (val+test 중앙값): 1: 0.97, 2: 0.91, 3: 0.40, 4: 0.09
  - 인접 단계 순서 AUC: 1>2 0.86, 2>3 0.84, 3>4 0.79
- WIDER 얼굴 단위 (가로 640, 기준 0.8)

| 데이터 | 감은 눈 recall | 뜬 눈을 감음으로 | 선글라스류 뜸 판정 |
|---|---|---|---|
| dev | 38/40 (95%). v0.0.2 37/40 (92%) | 239/1209 (20%). v0.0.2 16% | 23/36. v0.0.2 25/36 |
| holdout | 49/53 (92%). v0.0.2 44/53 (83%) | 276/1262 (22%). v0.0.2 18% | 28/38. v0.0.2 27/38 |

### smile + eye 사진 단위 (앱 설정)

- `evaluation/wider/app_setting_table.py`. 가로 640, YuNet 0.7, yaw >= 0.5 는 보이지 않음, SMILE_TH 0.5, EYE_TH 0.8
- 판단 불가(-1) 라벨이 있는 사진은 미충족으로 / 제외 두 가지 (두 방식 결과가 거의 같아 precision 만 나눠 적음)

| 방식 | dev precision (-1 미충족 / 제외) | dev recall | holdout precision (-1 미충족 / 제외) | holdout recall |
|---|---|---|---|---|
| blendshape (베이스라인) | 49/55 (89%) / 49/55 (89%) | 49/138 (36%) | 43/48 (90%) / 43/48 (90%) | 43/124 (35%) |
| CNN v0.0.2 | 59/66 (89%) / 59/65 (91%) | 59/138 (43%) | 39/46 (85%) / 39/46 (85%) | 39/124 (31%) |
| CNN v0.0.3 | 59/64 (92%) / 59/64 (92%) | 59/138 (43%) | 41/46 (89%) / 41/46 (89%) | 41/124 (33%) |

- 알려진 문제
  - 계속 웃는 사람의 웃는 눈(^^)은 점수가 낮게 나와 0.8 에서 감음으로 판정될 수 있음 (단계 2 중앙값 0.91 이지만 사람마다 차이가 큼)
  - 절대 기준 하나로는 웃는 눈과 감기는 눈을 모두 가르기 어려움. 촬영 안에서 같은 사람끼리 비교하는 판정은 설계 전
  - holdout 사진 단위 recall 은 blendshape 보다 낮음 (33% vs 35%)
  - 라즈베리파이 처리 시간 미측정

## v0.0.4

- smile, eye: v0.0.3 와 같은 파일 (`app/models/v0.0.4/smile_*`, `eye_*` 는 복사본)
- 앱 기준값 (yunet_cnn config): `OCC_TH` 0.59. 정한 근거는 `docs/dev_spec.md` 10절 12번

### occlusion (학습 20260926-1611)

- 파일: `app/models/v0.0.4/occlusion_{fp32,dynamic,int8}.tflite`, `occlusion_inference_config.json`
- 학습 노트북: `training/notebooks/occlusion_train.ipynb` (생성 스크립트 `training/notebooks/build_occlusion_nb.py`)
- 가림 기준 (2026-09-27 thinkcat): 입이 안 보이거나 두 눈이 모두 안 보이면 가림. 한쪽 눈만 가림, 수염, 선글라스류는 보임
- 학습 데이터: 드라이브 `Team-11/dataset/occlusion/preprocessed.zip` (`training/data/occlusion/make_dataset.py`. 64x64, 분할은 manifest 에 고정)
  - COFW 실제 얼굴: 가림 71장 (thinkcat 검수, `review.py`), 보임 약 1,400장 (검수에서 보임 + 가림 거의 없는 COFW, LFPW)
    - 출처: Caltech Occluded Faces in the Wild (COFW), X. P. Burgos-Artizzu, P. Perona, P. Dollár, "Robust face landmark estimation under occlusion", ICCV 2013. CC BY 4.0 (https://data.caltech.edu/records/bc0bf-nc666)
  - 합성 (smile, eye 전처리본 얼굴): 앞사람 겹침, COFW 입 가림 부위 옮겨 붙이기 (가림) / 그대로, 선글라스, 가리개를 이마, 볼, 옆에 붙임 (보임)
  - 가림 / 보임: train 1,401 / 2,959, val 294 / 567, test 791 / 1,375
- 입력: `face_rgb_0_255`, 128x128, RGB, 0~255 (YuNet 박스 중심 정사각형, smile 과 같음)
- 출력: 가림 점수 (sigmoid, **1 = 가림**). 좌우 뒤집기 평균 포함
- 학습: MobileNetV3Small, COFW 얼굴 샘플 가중치 3배, 클래스 가중치
- 노트북 기준값: 0.592 (val 에서 보이는 얼굴을 가림으로 보는 비율 <= 0.02 인 값 중 가림 recall 최대)
- 성능

| 데이터 | AUC | 가림 recall | 보이는 얼굴 오판 | 비고 |
|---|---|---|---|---|
| test 합성 | 0.988 | 87% | 0.6% | 노트북 기록 |
| test COFW 실제 | 0.953 | 60% (21/35) | 1.0% | 노트북 기록 |
| WIDER dev (가로 640) | 0.905 | 29% (13/45) | 1.7% | YuNet 신뢰도 규칙 AUC 0.897 |
| WIDER holdout (가로 640) | 0.912 | 50% (23/46) | 1.4% | YuNet 신뢰도 규칙 AUC 0.917 |

- 알려진 문제
  - 헬멧, 가면, 창살, 어둡고 흐린 작은 얼굴은 못 잡음 (학습 데이터에 없음)
  - 손으로 입을 가린 얼굴 일부가 점수 0.3~0.4 로 기준 아래
  - 가림 학습 데이터 대부분이 합성이라 합성 test 점수는 실제보다 높게 나옴. 판단은 WIDER 로
  - 라즈베리파이 처리 시간 미측정 (Mac 에서 얼굴당 약 2~3ms)

## v0.0.5

- smile, eye: v0.0.3 와 같은 파일 (복사본)
- 앱 기준값 (yunet_cnn config): `OCC_TH` 0.86 (WIDER dev 에서 정함. 근거는 `docs/dev_spec.md` 10절 12번)

### occlusion (학습 20260926-1728)

- 파일: `app/models/v0.0.5/occlusion_{fp32,dynamic,int8}.tflite`, `occlusion_inference_config.json`
- 학습 노트북: `training/notebooks/occlusion_train.ipynb` (생성: `python training/notebooks/build_occlusion_nb.py training/notebooks/occlusion_train.ipynb training/data/eye/augment_review/eyewear_fn.py`)
- v0.0.4 와 다른 점 (학습 데이터 드라이브 `Team-11/dataset/occlusion/preprocessed.zip`, `training/data/occlusion/make_dataset.py`)
  - 피부색을 맞춘 입 가림 (transfer_skin) 과 같은 방식의 보임 (decoy_skin): 본인 손처럼 색 차이가 없는 가림
  - 두 눈 가림: 앞사람 머리가 위에서 (eyes_person), 손, 물건이 가로 띠로 (eyes_transfer, eyes_transfer_skin). 어두운 가리개 포함 (선글라스와 구분)
  - 합성 선글라스는 zip 에서 빼고 노트북에서 학습 중 무작위로 (10%, 라벨 그대로)
  - 가림 / 보임: train 1,546 / 2,791, val 345 / 513, test 873 / 1,252
- 입력, 출력: v0.0.4 와 같음 (`face_rgb_0_255`, 128x128, 출력 1 = 가림)
- 노트북 기준값: 0.569 (앱은 WIDER dev 로 정한 0.86 사용)
- 성능

| 데이터 | AUC | 가림 recall | 보이는 얼굴 오판 | 비고 |
|---|---|---|---|---|
| test COFW 실제 | 0.961 | 74% (v0.0.4 60%) | 1.0% | 노트북 기록, 노트북 기준값 |
| test 합성 | 0.980 | 83% | 2.8% | 노트북 기록, 노트북 기준값 |
| WIDER dev (가로 640, 0.86) | 0.929 | 40% (18/45) | 2.0% | v0.0.4 AUC 0.905 |
| WIDER holdout (가로 640, 0.86) | 0.969 | 63% (29/46) | 1.7% | v0.0.4 AUC 0.912 |

- 알려진 문제
  - 헬멧, 가면, 창살, 어둡고 흐린 작은 얼굴은 여전히 약함
  - 합성 선글라스를 씌운 test 에서 보이는 얼굴 오판이 2.4% -> 5.0% 로 늘어남 (WIDER 실제 선글라스 얼굴은 dev 1/38, holdout 0/38)
  - 라즈베리파이 처리 시간 미측정

## v0.0.6

- occlusion: v0.0.5 와 같은 파일 (복사본)
- 앱 기준값 (yunet_cnn config): `SMILE_TH` 0.31, `SMILE_BOTTOM` 0.45, `EYE_TH` 0.84 (근거는 `docs/dev_spec.md` 10절 17번)

### smile (드라이브 v20, 학습 20260926-1958)

- 파일: `app/models/v0.0.6/smile_{fp32,dynamic,int8}.tflite`, `smile_inference_config.json`
- 학습 노트북: `training/notebooks/smile_train.ipynb` (생성: `python training/notebooks/build_smile_nb.py training/notebooks/smile_train.ipynb training/data/eye/augment_review/eyewear_fn.py`)
- v0.0.2 smile 과 다른 점
  - 입력: 얼굴 정사각형의 아래쪽 `SMILE_BOTTOM` 0.45 를 128x128 로 늘림 (코끝 ~ 턱. 눈, 선글라스 렌즈 제외)
    - 0.45 근거: 학습 데이터 눈 중심 높이 약 0.37 (99% 가 0.42 이내), 합성 렌즈 아래 끝 약 0.52
  - 합성 선글라스 증강 10% (렌즈 종류 dark, reflect, mirror, light. 눈 좌표 드라이브 `Team-11/dataset/smile/eye_landmarks.csv`, `training/data/smile/landmarks/make_eye_landmarks.py`). 입력에 렌즈가 거의 들어가지 않아 영향은 작음
  - 60 epoch 끝까지 학습 (조기 종료 patience 15)
- 입력, 출력: `face_rgb_0_255` (이름은 그대로), 128x128 RGB 0~255, 출력 1 = 웃음, 좌우 뒤집기 평균 포함
- 노트북 기준값: 0.318 (앱은 WIDER dev 로 정한 0.31)
- 성능 (WIDER 가로 640, 보이는 얼굴. `evaluation/wider/smile_retrain_compare.py`)

| 데이터 | AUC | 웃음 recall | 안 웃음을 웃음으로 | 비고 |
|---|---|---|---|---|
| test (노트북) | 0.982 | 94% | - | precision 94%, 노트북 기준값 |
| WIDER dev (0.31) | 0.900 | 77% | 11% | v0.0.5 (0.5): AUC 0.904, 79%, 11% |
| WIDER holdout (0.31) | 0.902 | 77% | 10% | v0.0.5 (0.5): AUC 0.889, 77%, 15% |
| WIDER 선글라스 dev / holdout (0.31) | 0.886 / 0.883 | 18/22 / 12/18 | 2/16 / 0/18 | v0.0.5: 0.875 / 0.840, 18/22 / 14/18, 5/16 / 5/18 |
| 앱 촬영 93장, 선글라스, 안 웃음 (증상 확인용) | - | - | 11장 | v0.0.5: 37장 |

- 사진 단위: 아래 "smile + eye 사진 단위" 참고

- 알려진 문제
  - 앱 촬영의 큰 얼굴(200px 이상)은 반사가 있으면 여전히 0.31~0.46 이 나와 일부 웃음으로 판정
  - 눈가 단서 없이 입, 볼로만 판정
  - 라즈베리파이 처리 시간 미측정

### eye (드라이브 v21, 학습 20260926-2040)

- 파일: `app/models/v0.0.6/eye_{fp32,dynamic,int8}.tflite`, `eye_inference_config.json`
- 학습 노트북: `training/notebooks/eye_train.ipynb` (생성: `python training/notebooks/build_eye_nb.py training/notebooks/eye_train.ipynb training/data/eye/augment_review/eyewear_fn.py`)
- v0.0.3 eye 와 다른 점
  - 실제 선글라스 추가: COFW 선글라스 얼굴 91장을 검수 (`training/data/eye/cofw_sunglasses/review.py`. 선글라스 78, 눈 보임 뜸 11, 제외 2) -> eye 전처리와 같은 방식 (`training/data/eye/cofw_sunglasses/preprocess_cofw.py`), 흑백 4장 제외 -> opened 85장 (`cofw_*.png`)
    - 학습 데이터 드라이브 `Team-11/dataset/eye/preprocessed.zip` (2,692장), 단계 라벨 드라이브 `Team-11/dataset/eye/levels.csv` (COFW 는 단계 1). 교사 점수, 눈 좌표 없음 (정답만으로 학습, 합성 선글라스 안 씌움)
    - 실제 선글라스: 13장 -> 98장 (train 약 57장)
    - COFW: Burgos-Artizzu, Perona, Dollár, ICCV 2013, CC BY 4.0
  - 합성 선글라스 렌즈 종류 추가 (dark 0.4, reflect 0.35, mirror 0.1, light 0.15. light 는 눈이 비쳐 라벨 유지)
  - 60 epoch 끝까지 학습 (조기 종료 patience 15)
- 입력, 출력: v0.0.3 와 같음 (`upper_face_rgb_0_255`, 얼굴 위쪽 0.6 을 160x160, 출력 1 = 뜸, 선글라스류 포함)
- 노트북 기준값: 0.712 (앱은 WIDER dev 로 정한 0.84)
- 성능 (WIDER 가로 640, 보이는 얼굴. `evaluation/wider/eye_retrain_compare.py`, 앱 기준은 dev 감은 눈 recall 95%)

| 데이터 | AUC | 감은 눈 recall | 뜬 눈을 감음으로 | 선글라스를 뜸으로 | 비고 |
|---|---|---|---|---|---|
| test (노트북) | 0.989 | 95% | - | 실제 36/36, 합성 99~100% | 노트북 기준값 |
| WIDER dev (0.84) | 0.947 | 41/43 (95%) | 20% | 33/38 (87%) | v0.0.3 (0.8): 0.947, 95%, 21%, 66% |
| WIDER holdout (0.84) | 0.931 | 55/59 (93%) | 20% | 36/38 (95%) | v0.0.3 (0.8): 0.933, 95%, 23%, 71% |
| 앱 촬영 114장, 선글라스 (증상 확인용, 기준 0.8) | - | - | - | 72장 (200px 이상 32장 중 20장) | v0.0.3: 36장 (0장) |

- 알려진 문제
  - 앱 촬영 120~200px 얼굴의 반사는 여전히 약함 (27장 중 10장만 뜸)
  - holdout 감은 눈 recall 은 v0.0.3 보다 1명 적음 (55/59, 56/59)

### smile + eye 사진 단위 (앱 설정)

- `evaluation/wider/compare_models.py` (가로 640, YuNet 0.7, 보임은 yaw 와 가림 CNN). 결과는 로컬 `evaluation/work/results/compare_models_v0.0.6_{dev,holdout}.log`, smile 만 바꾼 결과 `compare_models_v0.0.6_smile_only_*.log`

| 대상 | v0.0.5 precision, recall | smile 만 1958 (eye 0.8) | v0.0.6 (smile 1958, eye 2040 0.84) | yunet_landmark (베이스라인) |
|---|---|---|---|---|
| dev | 59/63 (94%), 59/143 (41%) | 59/65 (91%), 59/143 (41%) | 63/67 (94%), 63/143 (44%) | 50/55 (91%), 50/143 (35%) |
| holdout | 41/45 (91%), 41/126 (33%) | 44/48 (92%), 44/126 (35%) | 47/56 (84%), 47/126 (37%) | 43/48 (90%), 43/126 (34%) |

- holdout precision 하락 (잘못 통과 4장 -> 9장): 새로 잘못 통과한 5장은 모두 이전 eye 가 뜬 눈을 감음으로 잘못 봐서 막혀 있던 사진
  - 새 eye 가 감은 눈을 뜸으로 본 사진은 없음 (1장의 감은 눈은 이전 eye 도 뜸으로 판정)
  - 드러난 원인: YuNet 이 못 찾은 얼굴 2장, 안 웃은 얼굴을 웃음으로 (0.42) 1장, 감은 눈을 두 eye 모두 뜸으로 1장, 옆모습을 yaw 규칙이 못 거름 1장

## v0.0.7

- smile, occlusion: v0.0.6 와 같은 파일 (복사본)
- 앱 기준값 (yunet_cnn config): `EYE_TH` 0.84 (v0.0.6 과 같은 값. 근거는 `docs/dev_spec.md` 10절 18번)

### eye (드라이브 v22, 학습 20260927-0023)

- 파일: `app/models/v0.0.7/eye_{fp32,dynamic,int8}.tflite`, `eye_inference_config.json`
- 학습 노트북: `training/notebooks/eye_train.ipynb`
- v0.0.6 eye 와 다른 점
  - 웃는 눈 (단계 2) 목표를 1 로 (`LEVEL_TARGETS` {1: 1, 2: 1, 3: 0, 4: 0}. 전에는 0.7, 감기는 눈 0.3)
    - 웃는 눈 목표 0.7 이 앱 기준 0.84 보다 낮아 웃느라 가늘어진 눈을 감음으로 판정하는 문제
  - GENKI 웃는 눈 추가: GENKI 웃음 사진 (컬러) 중 기하 눈 열림이 작은 300장을 검수 (thinkcat) -> 293장 (뜸 136 (선글라스 31 포함), 웃는 눈 147, 감기는 눈 9, 감은 눈 1)
    - 스크립트: `training/data/eye/genki_smile/` (`select_candidates.py`, `review.py`, `build_dataset.py`)
    - 학습 데이터 드라이브 `Team-11/dataset/eye/preprocessed.zip` (2,985장), `levels.csv` (source genki_review)
  - 교사 점수를 베이스라인으로: 1 - max(eyeBlinkLeft, eyeBlinkRight) (드라이브 `Team-11/dataset/eye/teacher_baseline.csv`. COFW 선글라스는 빈칸)
  - 증류 `DISTILL_ALPHA` 0.5, 60 epoch 끝까지 (patience 15)
- 노트북 기준값: 0.783 (앱은 0.84)
- 성능 (WIDER 가로 640, 보이는 얼굴. `evaluation/wider/eye_retrain_compare.py`)

| 데이터 | AUC | 감은 눈 recall | 뜬 눈을 감음으로 | 선글라스를 뜸으로 | 비고 |
|---|---|---|---|---|---|
| test (노트북) | 0.985 | 94% | - | 실제 24/24, 합성 98~100% | 노트북 기준값 |
| WIDER dev (0.84) | 0.956 | 41/43 (95%) | 17.5% | 33/38 | v0.0.6: 0.947, 41/43, 20.2%, 33/38 |
| WIDER holdout (0.84) | 0.941 | 54/59 (92%) | 19.4% | 34/38 | v0.0.6: 0.931, 55/59, 20.2%, 36/38 |
| 앱 촬영 156장, 선글라스 (증상 확인용) | - | - | - | 94장 (200px 이상 51장 중 23장) | v0.0.6: 93장 (29장) |

- 기준 0.84 로 둔 이유: dev 에서 고른 0.814 는 뜬 눈 오판이 더 줄지만 (dev 15.7%) holdout 감은 눈을 2명 더 놓침 (53/59). 감은 눈을 놓치지 않는 쪽 우선 (`docs/dev_spec.md` 10절 13번)
- 같은 과정의 다른 학습 (쓰지 않음): 20260926-2324 (웃는 눈 목표 1, GENKI 없음) WIDER 는 v0.0.6 와 같음 / 20260927-0005 (GENKI + 베이스라인 교사, 웃는 눈 목표 0.7) 뜬 눈 오판이 조금 늘고 앱 촬영 반사 선글라스 29 -> 19장
- 알려진 문제
  - 웃느라 눈이 많이 가늘어진 얼굴은 여전히 감음 (표지 사진 두 명 0.50, 0.61. v0.0.6 0.22, 0.48)
  - 앱 촬영 큰 얼굴 (200px 이상) 의 반사 선글라스는 v0.0.6 보다 약함 (29 -> 23장)
  - 추가한 GENKI 사진이 거의 뜸 (283 대 10). 출처로 클래스를 구분하는 치우침 가능성 (WIDER 감은 눈 recall 로 확인: holdout 55 -> 54)
- 사진 단위 (앱 설정, `evaluation/wider/compare_models.py`. 결과 로컬 `evaluation/work/results/compare_models_v0.0.7_{dev,holdout}.log`)

| 대상 | v0.0.6 precision, recall | v0.0.7 precision, recall |
|---|---|---|
| dev | 63/67 (94%), 63/143 (44%) | 70/77 (91%), 70/143 (49%) |
| holdout | 47/56 (84%), 47/126 (37%) | 49/60 (82%), 49/126 (39%) |

## 새 버전을 추가할 때

- `app/models/{버전}/` 에 파일을 넣고 이 문서의 버전 목록과 절을 추가
- 저장소에는 현재 버전만: `.gitignore` 의 `!app/models/{버전}/` 예외를 새 버전으로 바꾸고, 이전 버전 폴더는 git 에서 뺌 (`git rm -r --cached`, 파일은 팀 드라이브에 둠)
- 적을 것: 드라이브 버전, 학습 노트북, 학습 데이터, 입력과 출력 형식 (특히 출력 방향), 성능, 알려진 문제
- 성능은 같은 조건으로 비교: WIDER 평가 라벨에서 `evaluation/wider/compare_cnn.py` (얼굴 판정 AUC), `photo_compare.py` (사진 조건 충족 precision, recall)
