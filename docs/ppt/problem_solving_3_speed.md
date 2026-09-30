# 문제해결과정 3: PREVIEW 판정 속도 개선 (YuNet + 손 검출)

## 문제해결과정 3 요약본

- 전제: 판정 속도 5 FPS 이상 (1회 200ms 이하) (목표, 팀장이 그냥 정한거)
- 문제 인식: PREVIEW 가 세 단계 중 가장 느림. (평균적으로 프레임 한장 검증시 150~160ms, PREPARE 50~60ms, CHECKING 90~100ms)
- 원인 분석: PREVIEW 시간의 대부분이 손 검출 (1위) 과 얼굴 검출 YuNet (2위)
- 해결 방안
  - 손 검출: HandLandmarker 를 손바닥 1클래스 YOLO11n (HaGRID 학습) 으로 교체
  - YuNet: 입력 해상도를 줄여 속도와 검출률의 최적점을 찾음. YuNet 은 모든 단계에서 실행되어 전 단계에 효과
- 적용 결과: 손 검출 시간 64% 감소, PREVIEW 6.2 FPS -> 9.7 FPS
- 한계 및 다음 단계
  - YuNet 축소는 검토까지만. 앱 반영과 라즈베리파이 측정은 남음
  - 새 손 검출기의 정확도 비교 기록 없음

## 문제해결과정 3 지표본

### 측정 조건

- 기기: 라즈베리파이 5 (4 CPU, 팬)
- 입력: WIDER 사진 100장 (가로 640), 워밍업 10장, 3회 반복
- 모델: yunet_cnn v0.0.7 fp32
- 단계별 판정 요청: PREVIEW 보임 + 손 들기 / PREPARE 웃음 / CHECKING 보임 + 웃음 + 눈
- 스크립트, 결과
  - `evaluation/speed/measure_speed.py`, `evaluation/speed/show_results.py`
  - 교체 전: `evaluation/speed/results/raspberrypi_20260927_153749_fan`
  - 교체 후: `evaluation/speed/results/raspberrypi_20260928_104711_fan`
  - YuNet 입력 크기: `evaluation/wider/yunet_input_size.py`, `evaluation/work/results/yunet_input_size_{dev,holdout}.log`

### 전제: 속도 목표

- 판정 속도 5 FPS 이상, 1회 200ms 이하 (`docs/dev_spec.md` 9절)

### 문제 인식: 단계별 판정 시간 (교체 전)

| 단계 | 평균 | 판정 FPS |
|---|---|---|
| PREVIEW | 160.4ms | 6.2 |
| PREPARE | 68.2ms | 14.7 |
| CHECKING | 91.8ms | 10.9 |

- PREVIEW 9명 이상 사진: 중앙값 208ms 로 목표 초과

### 원인 분석: PREVIEW 세부 시간 (교체 전)

| 항목 | 평균 | 비중 |
|---|---|---|
| 손 검출 (HandLandmarker) | 92.9ms | 58% |
| YuNet + 얼굴 자르기 | 56.6ms | 35% |
| 가림 CNN | 10.8ms | 7% |
| 합계 | 160.4ms | |

### 해결 방안: 손 검출 교체

- Ultralytics YOLO11n, 손바닥 1클래스, 입력 320x320, HaGRID 로 학습
- 손바닥 점수 0.5 이상이면 손 들기 (손가락 룰 제거)

### 해결 방안: YuNet 입력 크기 검토 (Mac, WIDER holdout)

| 입력 가로 | 검출 시간 | 보이는 얼굴 검출률 (50px 이상) | 30px 미만 얼굴 검출률 |
|---|---|---|---|
| 640 (현재) | 100% | 96.9% | 94.1% |
| 560 | 82% | 96.7% | 94.1% |
| 480 | 65% | 95.6% | 88.2% |
| 400 | 53% | 95.6% | 88.2% |
| 320 | 38% | 93.4% | 82.4% |

- 480: 검출 시간 35% 감소, 50px 이상 검출률 1.3%p 하락, 30px 미만 5.9%p 하락

### 적용 결과: 손 검출 교체 (라즈베리파이, PREVIEW)

| 항목 | 교체 전 | 교체 후 | 변화 |
|---|---|---|---|
| 손 검출 | 92.9ms | 33.4ms | -59.5ms (-64%) |
| PREVIEW 합계 | 160.4ms | 103.2ms | -57.2ms (-36%) |
| PREVIEW 판정 FPS | 6.2 | 9.7 | +3.5 |

### 한계 및 다음 단계

- YuNet 480 적용 시 라즈베리파이에서 약 20ms 감소 예상 (56.6ms x 35%). 실측 없음
- 작은 얼굴 (30px 미만) 검출률 하락 (94.1% -> 88.2%) 과의 균형 판단 필요
- 새 손 검출기의 정확도 (mAP, 손 들기 판정) 기록 없음
