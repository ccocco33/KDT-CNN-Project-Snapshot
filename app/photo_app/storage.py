"""사진 저장
- 세션 폴더 생성: {SAVE_DIR}/{YYYYmmdd_HHMMSS}/
- 촬영 사진 저장: frame_{번호:03d}.jpg
- 결과 사진 외 삭제
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

from . import config


class Session:
    def __init__(self, root: Path = config.SAVE_DIR, now=datetime.now):
        self.dir = Path(root) / now().strftime("%Y%m%d_%H%M%S")

    def save(self, frames: list, numbers: list[int] | None = None) -> list[Path]:
        """사진 저장 -> 경로 목록. numbers: 파일 이름 번호 (없으면 0 부터 차례로)"""
        self.dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, frame in zip(numbers if numbers is not None else range(len(frames)), frames):
            path = self.dir / f"frame_{i:03d}.jpg"
            cv2.imwrite(str(path), frame)
            paths.append(path)
        return paths

    def keep_only(self, keep: list[Path]) -> None:
        """keep 외 사진 삭제
        - 남은 사진이 없으면 세션 폴더도 삭제
        """
        if not self.dir.exists():
            return
        keep = {Path(p) for p in keep}
        for path in self.dir.glob("*.jpg"):
            if path not in keep:
                path.unlink()
        if not any(self.dir.iterdir()):
            self.dir.rmdir()
