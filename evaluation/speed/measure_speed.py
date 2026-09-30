"""라즈베리파이 추론 속도, CPU, 메모리 측정 (앱의 단계별 판정 요청 그대로)
- 모델: --models (기본 yunet_landmark yunet_cnn). 앱과 같은 설정 (각 모델 config: NUM_THREADS, MODEL_VERSION 등)
  - yunet_cnn 은 --quant 형식마다 따로 측정 (기본 fp32 dynamic int8. 이름 yunet_cnn_fp32 등)
    - fp32: 앱 코드 그대로 / dynamic, int8: 같은 폴더 (app/models/{MODEL_VERSION}/) 의 {smile,eye,occlusion}_{형식}.tflite 로 CNN 만 바꿈
    - int8: 입출력 int8 (scale, zero point 로 변환). XNNPACK 준비에 실패하면 XNNPACK 을 끄고 실행
  - 손 YOLO 는 --hand 형식마다 따로 측정 (기본 fp32. 모든 모델에 적용)
    - 손 모델 폴더 (각 모델 config 의 손 모델 경로와 같은 폴더) 의 HAND_FILES 파일로 손 검출기만 바꿈
    - 이름: 형식이 2개 이상이거나 fp32 가 아니면 뒤에 _hand_{형식} (예: yunet_cnn_fp32_hand_int8)
    - 모델 x CNN 형식 x 손 형식 조합마다 측정하므로 조합이 많으면 오래 걸림
- 단계: 앱 app.py 의 판정 요청 그대로 Model.predict 호출
  - PREVIEW: 보임 + 손 들기 / PREPARE: 웃음 / CHECKING: 보임 + 웃음 + 눈 (촬영 뒤 결과 고르기)
- 입력 (--source)
  - wider: WIDER 평가 사진 dev 중 앞 --frames 장을 가로 --width 로 줄임 (evaluation/work/wider_face). 얼굴 수가 사진마다 다름
  - camera: 카메라에서 --frames 장을 먼저 모두 찍어 두고 같은 프레임으로 측정 (찍는 시간은 빼고, 모델끼리 같은 입력)
- 순서: 반복(--repeat) 마다 모델 순서를 바꿔 발열, 클럭 변화의 영향을 나눔. 모델마다 처음 --warmup 장은 버림
- CPU, 메모리: 단계를 도는 동안 0.2초마다 기록 (워밍업 포함)
  - 프로세스 CPU (%): 이 프로세스의 CPU 시간 / 걸린 시간. 여러 코어를 쓰면 100 을 넘음 (4코어면 최대 400)
  - 프로세스 메모리 (RSS, MB)
  - Linux (라즈베리파이) 만: 코어별 CPU 사용률 (%, /proc/stat. 다른 프로그램 포함), 시스템 사용 메모리 (MB, /proc/meminfo)
- 기록 (evaluation/speed/results/{기기 이름}_{시각}[_{tag}]/)
  - summary.json: 기기 정보 (모델명, CPU 수, 메모리, 파이썬, 패키지 버전), 설정, 단계별 통계 (시간, CPU, 메모리), 측정 전후 CPU 온도와 스로틀 상태
  - frames.csv: 프레임별 시간 (반복, 모델, 단계, 프레임, 얼굴 수, 전체 ms, 세부 ms)
  - 터미널: 모델, 단계별 사진당 중앙값, 평균, p90, 판정 FPS (1000 / 평균), 얼굴 1명당 시간, CPU, 메모리, 세부 평균
- 세부 시간: Prediction.timings (face: YuNet + 얼굴 자르기, occlusion, smile, eye: CNN 합계, landmark: FaceLandmarker, hand: 손 YOLO)
- 라즈베리파이 외 기기 (Mac 등) 에서도 실행됨. 온도, 스로틀은 vcgencmd 가 있을 때만
- 결과 보기 (인원 수별 지연, 목표 충족): evaluation/speed/show_results.py

실행 (저장소 루트): app/.venv/bin/python evaluation/speed/measure_speed.py [--source wider|camera] [--quant fp32 dynamic int8] [--hand fp32 int8 w8a32] [--repeat 3] [--tag 메모]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

CODE = Path(__file__).resolve().parent   # 이 스크립트 위치
sys.path.insert(0, str(CODE.parent))   # evaluation/
from paths import LABELS, REPO, WIDER  # noqa: E402

sys.path.insert(0, str(REPO / "app"))
from photo_app import config  # noqa: E402
from photo_app.app import AGENT_REQUESTS, CHECK_REQUEST  # noqa: E402
from photo_app.models import create_model  # noqa: E402
from photo_app.models.util.hand import HandDetector  # noqa: E402
from photo_app.models.yunet_cnn import config as cnn_config  # noqa: E402
from photo_app.states import State  # noqa: E402

STAGES = {"PREVIEW": AGENT_REQUESTS[State.PREVIEW], "PREPARE": AGENT_REQUESTS[State.PREPARE], "CHECKING": CHECK_REQUEST}
RESULTS = CODE / "results"
PACKAGES = ("opencv-python", "opencv-contrib-python", "numpy", "mediapipe", "ai-edge-litert")
QUANTS = ("fp32", "dynamic", "int8")
HAND_FILES = {"fp32": "hand_yolo.tflite", "int8": "hand_yolo_int8.tflite", "w8a32": "hand_yolo_w8a32.tflite"}   # 손 YOLO 형식별 파일
SAMPLE_SEC = 0.2   # CPU, 메모리 기록 간격
LINUX = os.path.exists("/proc/stat")


def parse_args():
    ap = argparse.ArgumentParser(description="앱 단계별 추론 속도, CPU, 메모리 측정")
    ap.add_argument("--models", nargs="+", default=["yunet_landmark", "yunet_cnn"])
    ap.add_argument("--quant", nargs="+", choices=QUANTS, default=list(QUANTS), help="yunet_cnn 의 CNN 형식")
    ap.add_argument("--hand", nargs="+", choices=list(HAND_FILES), default=["fp32"], help="손 YOLO 형식 (모든 모델에 적용)")
    ap.add_argument("--source", choices=("wider", "camera"), default="wider")
    ap.add_argument("--frames", type=int, default=100, help="측정 프레임 수 (워밍업 포함)")
    ap.add_argument("--warmup", type=int, default=10, help="모델, 단계마다 버리는 처음 프레임 수")
    ap.add_argument("--repeat", type=int, default=3, help="반복 횟수 (반복마다 모델 순서를 바꿈)")
    ap.add_argument("--width", type=int, default=config.CAMERA_W, help="wider 사진을 줄일 가로 크기 (카메라 해상도)")
    ap.add_argument("--tag", default="", help="결과 폴더 이름에 붙일 메모 (예: 방열판, 전원)")
    return ap.parse_args()


class QClassifier:
    """LiteRT 분류 CNN (dynamic, int8). yunet_cnn.model.Classifier 와 같은 입출력 (RGB 0~255 -> 점수 0~1)"""

    def __init__(self, path, size: int, num_threads: int):
        self.size = size
        self.xnnpack = True
        try:
            self._open(path, num_threads, OpResolverType.AUTO)
            self(np.zeros((size, size, 3), np.uint8))
        except RuntimeError:   # MobileNetV3 int8 은 XNNPACK 준비 실패 (allocate_tensors 또는 첫 실행)
            self.xnnpack = False
            self._open(path, num_threads, OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)

    def _open(self, path, num_threads, resolver):
        self.it = Interpreter(model_path=str(path), num_threads=num_threads, experimental_op_resolver_type=resolver)
        self.it.allocate_tensors()
        self.inp, self.out = self.it.get_input_details()[0], self.it.get_output_details()[0]

    def __call__(self, rgb) -> float:
        x = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype(np.float32)[None]
        if np.issubdtype(self.inp["dtype"], np.integer):
            scale, zero = self.inp["quantization"]
            x = np.clip(np.rint(x / scale + zero), -128, 127).astype(self.inp["dtype"])
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"]).astype(np.float32)
        if np.issubdtype(self.out["dtype"], np.integer):
            scale, zero = self.out["quantization"]
            y = (y - zero) * scale
        return float(y.ravel()[0])


def build_models(names: list[str], quants: list[str], hands: list[str]) -> tuple[dict, dict]:
    """{측정 이름: 모델}, {측정 이름: 설정 메모}. yunet_cnn 은 CNN 형식마다, 모든 모델은 손 형식마다
    - 메모의 "model": 모델 이름 (config 조회용)
    """
    models, notes = {}, {}
    hand_suffix = hands != ["fp32"]
    for name in names:
        for q in (quants if name == "yunet_cnn" else [None]):
            for hand in hands:
                model = create_model(name)
                note = {"model": name}
                if q:
                    note["quant"] = q
                if q and q != "fp32":
                    folder = cnn_config.SMILE_PATH.parent
                    for attr, task, size in (("_smile", "smile", cnn_config.SMILE_SIZE), ("_eye", "eye", cnn_config.EYE_SIZE),
                                             ("_occ", "occlusion", cnn_config.OCC_SIZE)):
                        clf = QClassifier(folder / f"{task}_{q}.tflite", size, cnn_config.NUM_THREADS)
                        setattr(model, attr, clf)
                        note[f"{task}_xnnpack"] = clf.xnnpack
                note["hand"] = set_hand(model, name, hand)
                key = name + (f"_{q}" if q else "") + (f"_hand_{hand}" if hand_suffix else "")
                models[key] = model
                notes[key] = note
    return models, notes


def set_hand(model, name: str, hand: str) -> str:
    """모델의 손 검출기를 hand 형식 파일로 바꿈 (설정값은 그 모델 config 그대로). 손 모델 파일 이름을 돌려줌"""
    cfg = __import__(f"photo_app.models.{name}.config", fromlist=["config"])
    default = getattr(cfg, "HAND_PATH", None) or cfg.HAND_MODEL_PATH
    path = default.parent / HAND_FILES[hand]
    model._hands = HandDetector(path, cfg.NUM_HANDS, cfg.HAND_SCORE_TH, cfg.HAND_IOU_TH, cfg.NUM_THREADS)
    return path.name


def wider_frames(n: int, width: int) -> list[np.ndarray]:
    dev = [r["image"] for r in csv.DictReader(open(LABELS / "split.csv", encoding="utf-8")) if r["subset"] == "dev"]
    frames = []
    for path in dev:
        src = cv2.imread(str(WIDER / path))
        if src is None:
            continue
        s = width / src.shape[1]
        frames.append(cv2.resize(src, (width, round(src.shape[0] * s)), interpolation=cv2.INTER_AREA))
        if len(frames) == n:
            break
    if len(frames) < n:
        raise SystemExit(f"WIDER 사진 {len(frames)}장뿐 ({WIDER}). 평가 사진 zip 을 풀었는지 확인 (evaluation/README.md)")
    return frames


def camera_frames(n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"카메라를 열 수 없음: {config.CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_H)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    for _ in range(10):   # 노출 안정
        cap.read()
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def vcgencmd(*args: str) -> str | None:
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=2).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def meminfo() -> dict[str, int]:
    """/proc/meminfo (kB)"""
    out = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        out[k] = int(v.split()[0])
    return out


def device_info() -> dict:
    model = None
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        if os.path.exists(p):
            model = Path(p).read_text(errors="ignore").strip("\x00\n ")
            break
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"host": socket.gethostname(), "model": model or platform.machine(), "system": platform.platform(),
            "cpu_count": os.cpu_count(), "mem_total_mb": round(meminfo()["MemTotal"] / 1024) if LINUX else None,
            "python": platform.python_version(), "cv2": cv2.__version__, "packages": versions}


def thermal() -> dict:
    return {"temp": vcgencmd("measure_temp"), "throttled": vcgencmd("get_throttled")}


def model_settings(name: str) -> dict:
    """모델 config 중 속도에 영향을 주는 값"""
    try:
        cfg = __import__(f"photo_app.models.{name}.config", fromlist=["config"])
    except ImportError:
        return {}
    keys = ("MODEL_VERSION", "NUM_THREADS", "SCORE_TH", "NUM_HANDS", "HAND_SCORE_TH", "SMILE_SIZE", "EYE_SIZE", "OCC_SIZE")
    return {k: getattr(cfg, k) for k in keys if hasattr(cfg, k)}


class ResourceSampler:
    """단계를 도는 동안 SAMPLE_SEC 마다 CPU, 메모리 기록 (별도 스레드)"""

    def __init__(self):
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _proc_cpu() -> float:
        t = os.times()
        return t.user + t.system

    @staticmethod
    def _cores() -> list[tuple[int, int]] | None:
        """코어별 (busy, total) jiffies. Linux 만"""
        if not LINUX:
            return None
        out = []
        for line in open("/proc/stat"):
            if line.startswith("cpu") and line[3].isdigit():
                v = [int(x) for x in line.split()[1:]]
                idle = v[3] + v[4]   # idle + iowait
                out.append((sum(v) - idle, sum(v)))
        return out

    @staticmethod
    def _rss_mb() -> float:
        if LINUX:
            for line in open("/proc/self/status"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # macOS: bytes (현재가 아닌 최대)
        return peak / 1024 / 1024

    def _run(self):
        wall, cpu, cores = time.perf_counter(), self._proc_cpu(), self._cores()
        while True:
            stopped = self._stop.wait(SAMPLE_SEC)   # 멈출 때도 마지막 구간을 한 번 기록 (짧은 단계도 기록이 남음)
            wall2, cpu2, cores2 = time.perf_counter(), self._proc_cpu(), self._cores()
            s = {"proc_cpu": (cpu2 - cpu) / (wall2 - wall) * 100, "rss_mb": self._rss_mb()}
            if cores is not None:
                s["cores"] = [100 * (b2 - b1) / (t2 - t1) if t2 > t1 else 0.0 for (b1, t1), (b2, t2) in zip(cores, cores2)]
                m = meminfo()
                s["sys_used_mb"] = (m["MemTotal"] - m["MemAvailable"]) / 1024
            self.samples.append(s)
            wall, cpu, cores = wall2, cpu2, cores2
            if stopped:
                break

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def run(model, frames, warmup):
    """단계마다 frames 전부 predict -> {단계: ([(프레임, 얼굴 수, 전체 ms, timings), ...] 워밍업 제외, 자원 기록)}"""
    out = {}
    for stage, req in STAGES.items():
        rows = []
        with ResourceSampler() as sampler:
            for i, frame in enumerate(frames):
                t = time.perf_counter()
                pred = model.predict(frame, **req)
                ms = (time.perf_counter() - t) * 1000
                if i >= warmup:
                    rows.append((i, len(pred.faces), ms, dict(pred.timings)))
        out[stage] = (rows, sampler.samples)
    return out


def resource_stats(samples: list[dict]) -> dict:
    if not samples:
        return {}
    cpu = [s["proc_cpu"] for s in samples]
    rss = [s["rss_mb"] for s in samples]
    out = {"proc_cpu_mean": float(np.mean(cpu)), "proc_cpu_max": float(np.max(cpu)), "rss_max_mb": float(np.max(rss))}
    if "cores" in samples[0]:
        cores = np.array([s["cores"] for s in samples])
        out.update({"cores_mean": [float(v) for v in cores.mean(0)], "cores_max": [float(v) for v in cores.max(0)],
                    "cpu_total_mean": float(cores.mean()), "sys_used_max_mb": float(max(s["sys_used_mb"] for s in samples))})
    return out


def stats(rows, samples) -> dict:
    ms = np.array([r[2] for r in rows])
    per_face = [r[2] / r[1] for r in rows if r[1]]
    parts = defaultdict(list)
    for r in rows:
        for k, v in r[3].items():
            parts[k].append(v)
    return {"n": len(ms), "median_ms": float(np.median(ms)), "mean_ms": float(ms.mean()),
            "p90_ms": float(np.percentile(ms, 90)), "fps": float(1000 / ms.mean()),
            "per_face_median_ms": float(np.median(per_face)) if per_face else None,
            "faces_mean": float(np.mean([r[1] for r in rows])),
            "parts_mean_ms": {k: float(np.mean(v)) for k, v in parts.items()},
            "resource": resource_stats(samples)}


def main():
    args = parse_args()
    frames = wider_frames(args.frames, args.width) if args.source == "wider" else camera_frames(args.frames)
    info = device_info()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS / (f"{info['host']}_{stamp}" + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"기기: {info['model']} ({info['cpu_count']} CPU, 메모리 {info['mem_total_mb'] or '-'} MB), 입력 {args.source} "
          f"{len(frames)}장 {frames[0].shape[1]}x{frames[0].shape[0]}, 반복 {args.repeat}")
    models, notes = build_models(args.models, args.quant, args.hand)
    names = list(models)
    for name, note in notes.items():
        print(f"  {name}: {note}")
    all_rows, all_samples = defaultdict(list), defaultdict(list)   # (모델, 단계) -> 행, 자원 기록
    thermals = [{"when": "start", **thermal()}]
    with open(out_dir / "frames.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["repeat", "model", "stage", "frame", "faces", "total_ms", "timings"])
        for rep in range(args.repeat):
            order = names if rep % 2 == 0 else list(reversed(names))
            for name in order:
                for stage, (rows, samples) in run(models[name], frames, args.warmup).items():
                    all_rows[(name, stage)] += rows
                    all_samples[(name, stage)] += samples
                    for i, faces, ms, timings in rows:
                        w.writerow([rep, name, stage, i, faces, round(ms, 3), json.dumps({k: round(v, 3) for k, v in timings.items()})])
                thermals.append({"when": f"repeat {rep} {name}", **thermal()})
                print(f"  반복 {rep + 1} {name} 끝 {thermals[-1]['temp'] or ''}", flush=True)
    base = {name: model_settings(notes[name]["model"]) for name in names}
    summary = {"device": info, "args": vars(args), "time": stamp,
               "model_settings": {name: {**base[name], **notes.get(name, {})} for name in names}, "thermal": thermals,
               "stages": {name: {stage: stats(all_rows[(name, stage)], all_samples[(name, stage)]) for stage in STAGES}
                          for name in names}}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n{'단계':<9} {'모델':<26} {'중앙값':>8} {'평균':>8} {'p90':>8} {'FPS':>6} {'얼굴당':>7} {'CPU%':>6} {'RSS MB':>7}   세부 평균 (ms)")
    for stage in STAGES:
        for name in names:
            s = summary["stages"][name][stage]
            r = s["resource"]
            face = f"{s['per_face_median_ms']:.1f}" if s["per_face_median_ms"] is not None else "-"
            detail = ", ".join(f"{k} {v:.1f}" for k, v in s["parts_mean_ms"].items())
            print(f"{stage:<9} {name:<26} {s['median_ms']:>7.1f} {s['mean_ms']:>8.1f} {s['p90_ms']:>8.1f} {s['fps']:>6.1f} "
                  f"{face:>7} {r.get('proc_cpu_mean', 0):>6.0f} {r.get('rss_max_mb', 0):>7.0f}   {detail}")
    if any(t["throttled"] not in (None, "throttled=0x0") for t in thermals):
        print("주의: 스로틀 발생 (summary.json 의 thermal). 방열, 전원 확인")
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    main()
