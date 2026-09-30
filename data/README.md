# data

학습, 평가에 쓴 데이터의 출처와 라벨 정보. 데이터 파일은 저장소에 두지 않음 (팀 드라이브 `Team-11/dataset/`).
데이터를 만드는 스크립트는 `training/` (`training/README.md`), 모델별 결과는 `docs/model_versions.md`.

## 학습 데이터

| 모델 | 클래스 | 데이터 | 출처 | 규모 |
|---|---|---|---|---|
| smile | 웃음 / 안 웃음 | GENKI-4K | GitHub `truongnmt/smile-detection` 의 GENKI4K (https://github.com/truongnmt/smile-detection/tree/master/GENKI4K) | 4,000장 -> 전처리 후 3,774장 |
| eye | 감음 | Closed Eyes In The Wild (CEW) | CEW (LFW 를 바탕으로 만든 데이터셋) | 1,192장 -> 흑백 제외 후 1,120장 |
| eye | 뜸 | LFW | Kaggle `jessicali9530/lfw-dataset` (https://www.kaggle.com/datasets/jessicali9530/lfw-dataset). CEW 에 뜬 눈 데이터가 없어 원본 LFW 에서 1,500명 | 1,498장 -> 흑백 제외 후 1,487장 |
| eye (v0.0.6 추가) | 뜸 (선글라스) | COFW | Burgos-Artizzu, Perona, Dollár, ICCV 2013, CC BY 4.0 | 91장 검수 -> 85장 |
| eye (v0.0.7 추가) | 웃는 눈 등 | GENKI-4K 웃음 사진 | smile 과 같은 GENKI | 눈이 가늘게 잡힌 300장 검수 -> 293장 (뜸 283, 감음 10) |
| occlusion | 가림 / 보임 | COFW 실제 가림 + 합성 | COFW, smile / eye 데이터에 가리개 합성 | train 4,337 / val 858 / test 2,125 |

- 전처리 (공통): YuNet 으로 얼굴을 다시 찾아 박스 높이 정사각형으로 자르고 64x64 로 저장. eye 는 흑백 제외
- eye 의 감은 눈 (CEW) 과 뜬 눈 (LFW) 은 모은 경로가 달라 선명도, 얼굴이 꽉 찬 정도, 흑백 비율이 다름 (조건값만으로 클래스 구분 AUC 0.977). 전처리로 줄임 (0.891)

## 라벨

- smile: GENKI-4K 원본 라벨
- eye: 폴더 라벨 (뜸 / 감음) + 눈 뜬 정도 4단계 (`levels.csv`, 2,985장. v0.0.7 기준)
  - 단계: 1 뜸 1,592 / 2 웃는 눈 273 / 3 감기는 눈 54 / 4 감은 눈 1,066
  - 라벨 출처: 규칙 자동 부여 2,162, 사람 검수 437, 선글라스 검수 8, COFW 선글라스 85, GENKI 웃는 눈 검수 293 (thinkcat)
- COFW (eye 선글라스, occlusion 가림): 팀 검수 (thinkcat)
  - 가림 기준 : 입이 안 보이거나 두 눈이 모두 안 보이면 가림. 한쪽 눈만 가림, 선글라스류는 보임
  - 선글라스로 눈이 안 보이면 눈 뜸으로 봄
- 합성 데이터 (occlusion 가리개, 학습 중 합성 선글라스): 만들 때 정한 라벨
- 보조 값 (사람 라벨 아님)
  - 지식 증류 교사 점수: FaceLandmarker blendshape, 눈 기하 지표로 만듦
  - 합성 선글라스용 눈 좌표: YuNet 랜드마크

## 평가 데이터

| 데이터 | 출처 | 규모 | 라벨 |
|---|---|---|---|
| WIDER FACE 단체 사진 | WIDER FACE (CC BY-NC-ND) | 사진 1,040장 (dev 520 / holdout 520), 얼굴 3,938개 | 팀 직접 라벨링 (`evaluation/wider/label_tool.py`) |

- 얼굴별 라벨
  - 보임: 보임 2,925 / 안 보임 969 (옆모습 802, 사람에 가림 69, 손 8, 마스크 1, 기타 89)
  - 웃음: 웃음 2,037 / 안 웃음 840 / 판단 불가 1,015
  - 눈: 뜸 2,616 / 감음 112 / 선글라스 82 / 판단 불가 1,080
- 기준값은 dev 로 고르고 holdout 은 확인용
- WIDER FACE 는 CC BY-NC-ND: 팀 내부 평가에만 쓰고 사진, 라벨, 결과 그림을 외부에 공개하지 않음
