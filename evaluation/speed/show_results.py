"""속도 측정 결과 보기 (measure_speed.py 의 결과 폴더)
- 결과 폴더마다
  - 기기, 입력, 반복, 측정 전후 CPU 온도와 스로틀
  - 단계, 모델별 사진당 중앙값, 평균, p90, 판정 FPS, 목표 (판정 지연 200 ms, 판정 FPS 5) 충족 o/x
  - 단계, 모델별 CPU (프로세스 평균 %, Linux 는 전체 코어 평균 %), 메모리 (프로세스 최대 RSS, Linux 는 시스템 사용 최대)
  - 모델 설정 (yunet_cnn 의 CNN 형식, XNNPACK 사용 여부)
  - 인원 수 구간별 사진당 중앙값 (frames.csv 의 얼굴 수) 과 WARN_PEOPLE (중앙값이 200 ms 를 넘는 가장 적은 인원 구간)
- 인자가 없으면 results/ 의 폴더 전부 (오래된 순)

실행 (저장소 루트): python3 evaluation/speed/show_results.py [결과 폴더 ...]
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
LATENCY_MS = 200   # 판정 지연 목표 (docs/dev_spec.md 9절)
MIN_FPS = 5        # 판정 FPS 목표
PEOPLE = ((1, 1), (2, 2), (3, 4), (5, 6), (7, 8), (9, 99))   # 인원 수 구간


def people_label(lo: int, hi: int) -> str:
    if lo == hi:
        return f"{lo}명"
    return f"{lo}명 이상" if hi >= 99 else f"{lo}~{hi}명"


def show(d: Path):
    s = json.loads((d / "summary.json").read_text())
    dev, args = s["device"], s["args"]
    print(f"\n=== {d.name}")
    print(f"기기: {dev['model']} ({dev['cpu_count']} CPU, {dev['system']}), 입력 {args['source']} {args['frames']}장 "
          f"(워밍업 {args['warmup']}), 반복 {args['repeat']}")
    temps = [t["temp"] for t in s["thermal"] if t["temp"]]
    throttled = sorted({t["throttled"] for t in s["thermal"] if t["throttled"]})
    if temps:
        print(f"CPU 온도: {temps[0]} -> {temps[-1]}, 스로틀: {', '.join(throttled)}")
    for model, cfg in s["model_settings"].items():
        print(f"  {model}: {cfg}")
    print(f"\n{'단계':<9} {'모델':<18} {'중앙값':>8} {'평균':>8} {'p90':>8} {'FPS':>6}  지연<={LATENCY_MS}  FPS>={MIN_FPS}"
          f"  {'CPU%':>6} {'코어%':>6} {'RSS MB':>7} {'시스템 MB':>9}")
    for stage in ("PREVIEW", "PREPARE", "CHECKING"):
        for model, st in s["stages"].items():
            x = st[stage]
            r = x.get("resource", {})
            num = lambda k, fmt: format(r[k], fmt) if r.get(k) is not None else "-"  # noqa: E731
            print(f"{stage:<9} {model:<18} {x['median_ms']:>7.1f} {x['mean_ms']:>8.1f} {x['p90_ms']:>8.1f} {x['fps']:>6.1f}"
                  f"  {'o' if x['median_ms'] <= LATENCY_MS else 'x':>8}  {'o' if x['fps'] >= MIN_FPS else 'x':>7}"
                  f"  {num('proc_cpu_mean', '.0f'):>6} {num('cpu_total_mean', '.0f'):>6} {num('rss_max_mb', '.0f'):>7}"
                  f" {num('sys_used_max_mb', '.0f'):>9}")
    rows = list(csv.DictReader(open(d / "frames.csv")))
    by = defaultdict(list)
    for r in rows:
        by[(r["stage"], r["model"], int(r["faces"]))].append(float(r["total_ms"]))
    head = "".join(f"{people_label(lo, hi):>10}" for lo, hi in PEOPLE)
    print(f"\n인원 수별 사진당 중앙값 ms (괄호: 사진 수). WARN_PEOPLE: 중앙값이 {LATENCY_MS} ms 를 넘는 가장 적은 인원 구간")
    print(f"{'단계':<9} {'모델':<18}{head}   WARN_PEOPLE")
    for stage in ("PREVIEW", "PREPARE", "CHECKING"):
        for model in s["stages"]:
            cells, warn = [], "-"
            for lo, hi in PEOPLE:
                ms = [v for n in range(lo, hi + 1) for v in by.get((stage, model, n), [])]
                if not ms:
                    cells.append(f"{'-':>10}")
                    continue
                med = statistics.median(ms)
                cells.append(f"{med:>6.0f}({len(ms) // args['repeat']:>2})")
                if med > LATENCY_MS and warn == "-":
                    warn = f"{lo}명"
            print(f"{stage:<9} {model:<18}{''.join(cells)}   {warn}")


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or sorted(p for p in RESULTS.iterdir() if (p / "summary.json").exists())
    if not dirs:
        raise SystemExit(f"결과 없음: {RESULTS}")
    for d in dirs:
        show(d)


if __name__ == "__main__":
    main()
