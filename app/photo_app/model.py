"""모델 인터페이스
- 프레임 1장 요청 -> 판정 결과 1개 반환
- 요청 인자(smile, eye, hand, visibility)는 돌려받을 결과 선택. 계산 범위는 구현이 결정
- 웃음, 눈 뜸, 손 들기, 보임 판정 기준(임계값 등)은 Model 구현 내부에 둠
- 얼굴 크기 기준(min_face_h)은 제품 요구사항이라 앱이 guidance, ready_to_start 에 넘김
- 구현은 photo_app/models/ 에 두고 config.MODEL 로 선택
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Face:
    bbox: tuple[int, int, int, int]   # x, y, w, h
    smile: bool | None                # 웃음 여부. smile 미요청 또는 판정 불가면 None
    eyes_open: bool | None            # 눈 뜸 여부. eye 미요청 또는 판정 불가면 None
    visible: bool | None = None       # 입과 눈(한쪽 이상)이 보이는지 (옆모습, 입 가림, 두 눈 가림이면 False. 한쪽 눈만 가림, 선글라스류는 True). visibility 미요청이면 None
                                      # - False 면 웃음, 눈을 판정하지 않음 (smile, eyes_open 은 None)
    eye_score: float | None = None    # 눈 뜸 정도 (0~1, 클수록 뜸). eye 미요청 또는 판정 불가면 None
                                      # - 결과 사진 선택에서 같은 촬영의 사진끼리 비교하는 용도. 척도는 모델마다 다름
    occlusion: float | None = None    # 가림 점수 (0~1, 클수록 가림). 가림 모델이 없거나 실행하지 않았으면 None (dev 표시, 로그용)
    scores: dict[str, tuple[float, float, bool]] = field(default_factory=dict)
                                      # 판정 항목별 (점수, 기준값, 통과 여부). 모델 평가 화면(dev) 표시, 저장용
                                      # - 이름 예: yaw, occ, smile, eye (cnn) / yaw, smile, blink (landmark 계열). 실행한 항목만
                                      # - 통과 = 보임 쪽, 웃음, 눈 뜸 쪽 판정


@dataclass
class Hand:
    bbox: tuple[int, int, int, int]   # x, y, w, h
    raised: bool                      # 손 들기 여부
    pose: str = ""                    # 판정 이유 (dev 표시용. 예: "open up", "fist", "not up"). 없으면 빈 칸


@dataclass
class Prediction:
    faces: list[Face]                 # 항상 반환
    hands: list[Hand] | None          # hand 미요청이면 None
    hand_raised: bool | None          # 손 들기 여부 (키보드 흉내 포함). hand 미요청이면 None
    elapsed_ms: float                 # 요청 처리 시간
    requested: frozenset[str] = field(default_factory=frozenset)   # 요청한 결과 {"smile", "eye", "hand", "visibility"}
    timings: dict[str, float] = field(default_factory=dict)        # 단계별 시간(ms). 키: face, hand, smile, eye (구현마다 다름)

    def meets_condition(self) -> bool:
        """사진의 조건 충족 여부
        - 얼굴 1명 이상, 모든 얼굴이 웃고 눈을 뜸
        - smile, eye 를 함께 요청한 결과에서만 사용 가능
        """
        if not {"smile", "eye"} <= self.requested:
            raise ValueError("meets_condition 은 predict(smile=True, eye=True) 로 받은 결과에서만 사용 가능")
        return len(self.faces) > 0 and all(f.smile and f.eyes_open for f in self.faces)

    def eye_score(self) -> float | None:
        """사진의 눈 뜸 점수: 얼굴 중 가장 낮은 eye_score (눈을 가장 작게 뜬 사람 기준)
        - 얼굴이 없거나, 점수가 없는 얼굴이 있으면 None
        """
        scores = [f.eye_score for f in self.faces]
        if not scores or None in scores:
            return None
        return min(scores)

    def guidance(self, min_face_h: int) -> list[tuple[Face, str]]:
        """안내 대상 얼굴과 이유 [(얼굴, NOT_VISIBLE | TOO_SMALL), ...]
        - NOT_VISIBLE: visible 이 False
        - TOO_SMALL: bbox 높이 < min_face_h
        - 한 얼굴이 둘 다 해당하면 NOT_VISIBLE 만
        - visibility 를 요청한 결과에서만 사용 가능
        """
        if "visibility" not in self.requested:
            raise ValueError("guidance 는 predict(visibility=True) 로 받은 결과에서만 사용 가능")
        out = []
        for f in self.faces:
            if f.visible is False:
                out.append((f, NOT_VISIBLE))
            elif f.bbox[3] < min_face_h:
                out.append((f, TOO_SMALL))
        return out

    def ready_to_start(self, min_face_h: int) -> bool:
        """촬영 시작 가능 여부
        - 손 들기, 얼굴 1명 이상, 안내 대상 얼굴 없음
        - hand, visibility 를 함께 요청한 결과에서만 사용 가능
        """
        if not {"hand", "visibility"} <= self.requested:
            raise ValueError("ready_to_start 는 predict(hand=True, visibility=True) 로 받은 결과에서만 사용 가능")
        return bool(self.hand_raised) and len(self.faces) > 0 and not self.guidance(min_face_h)


NOT_VISIBLE = "not_visible"   # 안내 이유: 얼굴이 보이지 않음 (옆모습, 가림)
TOO_SMALL = "too_small"       # 안내 이유: 얼굴이 너무 작음


def requested(smile: bool, eye: bool, hand: bool, visibility: bool) -> frozenset[str]:
    """요청 인자 -> Prediction.requested"""
    on = (("smile", smile), ("eye", eye), ("hand", hand), ("visibility", visibility))
    return frozenset(name for name, flag in on if flag)


class Model:
    def predict(self, frame, *, smile: bool, eye: bool, hand: bool, visibility: bool) -> Prediction:
        """프레임 판정
        - 얼굴 bbox 는 항상 반환
        - smile: 얼굴별 웃음 여부
        - eye: 얼굴별 눈 뜸 여부
        - hand: 손 bbox 목록, 손 들기 여부
        - visibility: 얼굴별 보임 여부. 보이지 않는 얼굴은 웃음, 눈 None
        - 요청하지 않은 결과는 None
        """
        raise NotImplementedError
