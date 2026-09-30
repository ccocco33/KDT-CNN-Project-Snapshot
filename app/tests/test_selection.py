import pytest

from photo_app import config
from photo_app.model import Face, Prediction, requested
from photo_app.selection import EyeHold, closeness, face_margin, select, select_closest

BOX = (0, 0, 60, 60)


def scored(scores):
    """[(사진 번호, 판정 결과), ...] 찍힌 순서. 사진마다 눈 뜸 점수가 scores 인 얼굴 1명"""
    return [(i, Prediction([Face(BOX, True, True, True, s)], None, None, 0.0, requested(True, True, False, True)))
            for i, s in enumerate(scores)]


def test_empty():
    """조건에 맞는 사진이 없음"""
    # given
    ok = []

    # when
    picked = select(ok, 5)

    # then
    assert picked == []


def test_fewer_than_n_returns_all():
    """설정한 사진 수보다 적으면 전부 반환 (찍힌 순서)"""
    # given
    ok = scored([0.9, 0.1, 0.5])

    # when
    picked = select(ok, 5)

    # then
    assert picked == [0, 1, 2]


def test_exactly_n_returns_all():
    """설정한 사진 수와 같으면 전부 반환"""
    # given
    ok = scored([0.5] * 5)

    # when
    picked = select(ok, 5)

    # then
    assert picked == [0, 1, 2, 3, 4]


def test_more_than_n_picks_best_in_each_section():
    """설정한 사진 수보다 많으면 n개 구간마다 점수가 가장 높은 사진"""
    # given
    ok = scored([0.1, 0.9, 0.3, 0.8, 0.2, 0.4, 0.7, 0.6, 0.5, 0.95])   # 구간: [0,1] [2,3] [4,5] [6,7] [8,9]

    # when
    picked = select(ok, 5)

    # then
    assert picked == [1, 3, 5, 6, 9]


def test_more_than_n_skips_low_eye_score():
    """눈을 감으려는 사진(점수가 낮은 사진)은 같은 구간에 더 좋은 사진이 있으면 빠짐"""
    # given
    ok = scored([0.94] * 7 + [0.90] + [0.94] * 7)

    # when
    picked = select(ok, 5)

    # then
    assert 7 not in picked


def test_tie_keeps_earlier_photo():
    """점수가 같으면 먼저 찍힌 사진"""
    # given
    ok = scored([0.5] * 10)

    # when
    picked = select(ok, 5)

    # then
    assert picked == [0, 2, 4, 6, 8]


def test_more_than_n_one_per_section_in_order():
    """설정한 사진 수보다 많을 때, 구간마다 1장씩 중복 없이 찍힌 순서로"""
    for m in range(6, 60):
        # given
        ok = scored([(i * 7919 % 13) / 13 for i in range(m)])

        # when
        picked = select(ok, 5)

        # then
        assert len(picked) == 5 and len(set(picked)) == 5, m
        assert picked == sorted(picked), m
        assert picked[0] < m // 5 and picked[-1] >= 4 * m // 5, m


def test_score_is_lowest_face_in_photo():
    """사진의 점수는 눈을 가장 작게 뜬 사람 기준"""
    # given
    def photo(a, b):
        faces = [Face(BOX, True, True, True, a), Face(BOX, True, True, True, b)]
        return Prediction(faces, None, None, 0.0, requested(True, True, False, True))
    ok = [(0, photo(0.99, 0.70)), (1, photo(0.90, 0.90))] + [(i, photo(0.9, 0.9)) for i in range(2, 10)]

    # when
    picked = select(ok, 5)

    # then
    assert picked[0] == 1


def face(smile=True, eyes_open=True, visible=True, scores=None):
    return Face(BOX, smile, eyes_open, visible, None, None, scores or {})


def pred(*faces):
    return Prediction(list(faces), None, None, 0.0, requested(True, True, False, True))


def test_face_margin_is_smallest_gap_to_threshold():
    """가장 모자란 값: 웃음, 눈 점수의 기준 대비 차이 중 최솟값"""
    # given
    f = face(eyes_open=False, scores={"smile": (0.9, 0.31, True), "eye": (0.5, 0.84, False), "yaw": (0.1, 0.6, True)})

    # when
    margin = face_margin(f)

    # then
    assert margin == pytest.approx(0.5 - 0.84)


def test_face_margin_blink_lower_is_pass():
    """눈 감음 점수 (blink) 는 기준 - 점수"""
    # given
    f = face(eyes_open=False, scores={"smile": (0.9, 0.5, True), "blink": (0.7, 0.5, False)})

    # when
    margin = face_margin(f)

    # then
    assert margin == pytest.approx(0.5 - 0.7)


@pytest.mark.parametrize("f", [face(visible=False, smile=None, eyes_open=None), face(smile=None)])
def test_face_margin_not_judged(f):
    """보이지 않거나 판정하지 못한 얼굴은 -1"""
    # given: f

    # when
    margin = face_margin(f)

    # then
    assert margin == -1.0


def test_face_margin_without_scores_uses_pass():
    """점수가 없는 모델 (fake) 은 통과 여부로"""
    # given
    passed, failed = face(), face(eyes_open=False)

    # when
    margins = (face_margin(passed), face_margin(failed))

    # then
    assert margins == (0.0, -1.0)


def test_closeness_prefers_more_passing_faces():
    """조건을 충족한 얼굴 수가 많은 사진이 더 가까움 (모자란 값보다 먼저)"""
    # given
    two_pass = pred(face(), face(), face(eyes_open=False, scores={"eye": (0.1, 0.84, False)}))
    one_pass = pred(face(), face(eyes_open=False, scores={"eye": (0.8, 0.84, False)}),
                    face(eyes_open=False, scores={"eye": (0.8, 0.84, False)}))

    # when
    order = closeness(two_pass) > closeness(one_pass)

    # then
    assert order


def closest_cands(eyes):
    """[(사진 번호, 판정 결과), ...] 얼굴 1명, 눈 점수 eyes (기준 0.84 미만이면 눈 감음)"""
    return [(i, pred(face(eyes_open=e >= 0.84, scores={"smile": (0.9, 0.31, True), "eye": (e, 0.84, e >= 0.84)})))
            for i, e in enumerate(eyes)]


def test_select_closest_picks_best_per_interval():
    """찍힌 순서로 n개 구간, 구간마다 가장 가까운 사진"""
    # given
    cands = closest_cands([0.1, 0.5, 0.2, 0.7, 0.3, 0.1, 0.6, 0.2, 0.4, 0.3])

    # when
    picked = select_closest(cands, 5)

    # then
    assert picked == [1, 3, 4, 6, 8]


def test_select_closest_skips_photos_without_faces():
    """얼굴이 없는 사진은 고르지 않음. 모두 없으면 빈 결과"""
    # given
    empty = [(0, pred()), (1, pred())]
    mixed = empty + closest_cands([0.5])

    # when
    picked = (select_closest(empty, 5), select_closest(mixed, 5))

    # then
    assert picked == ([], [0])


# ---- EyeHold (눈 감음 허용) ----

def eye(closed=True, x=0, visible=True, smile=True, h=100):
    """눈 판정한 얼굴 1명 (박스 x 위치, 높이 h). 감음 0.05, 뜸 0.95"""
    s = 0.05 if closed else 0.95
    return Face((x, 0, 80, h), smile, not closed, visible, s, scores={"eye": (s, 0.84, not closed), "smile": (0.99, 0.31, smile)})


def photo(*faces, eye_req=True):
    return Prediction(list(faces), None, None, 0.0, requested(True, eye_req, False, True))


def run(hold, times, make):
    """times 마다 make() 판정을 apply 한 결과 목록"""
    return [hold.apply(t, make()) for t in times]


def test_eye_hold_short_blink_not_allowed():
    """0.3초 미만 눈 감음은 허용하지 않음"""
    # given
    hold = EyeHold()

    # when
    out = run(hold, [0.0, 0.1, 0.2], lambda: photo(eye()))

    # then
    assert not any(p.meets_condition() for p in out)


def test_eye_hold_allows_from_hold_sec_and_keeps_raw():
    """0.3초가 된 프레임부터 허용. 원래 판정, eye_score, eye 점수는 그대로"""
    # given
    hold = EyeHold()
    raw = photo(eye())
    run(hold, [10.0, 10.1, 10.2], lambda: raw)

    # when
    out = hold.apply(10.3, raw)

    # then
    assert out.meets_condition() and out.faces[0].scores["eye_hold"][2]
    assert raw.faces[0].eyes_open is False and "eye_hold" not in raw.faces[0].scores
    assert out.faces[0].eye_score == 0.05 and out.faces[0].scores["eye"][2] is False


def test_eye_hold_needs_min_samples(monkeypatch):
    """시간이 지나도 관측 수가 EYE_HOLD_MIN_SAMPLES 미만이면 허용하지 않음"""
    # given
    monkeypatch.setattr(config, "EYE_HOLD_MIN_SAMPLES", 4)
    hold = EyeHold()
    hold.apply(0.0, photo(eye()))
    hold.apply(0.15, photo(eye()))

    # when
    out = hold.apply(0.3, photo(eye()))   # 0.3초지만 3회

    # then
    assert not out.meets_condition()


def test_eye_hold_open_resets():
    """눈을 뜨면 누적 초기화"""
    # given
    hold = EyeHold()
    run(hold, [0.0, 0.1, 0.2], lambda: photo(eye()))
    hold.apply(0.3, photo(eye(closed=False)))

    # when
    out = hold.apply(0.4, photo(eye()))

    # then
    assert not out.meets_condition()


def test_eye_hold_missing_face_resets():
    """얼굴이 사라지면 누적 초기화"""
    # given
    hold = EyeHold()
    run(hold, [0.0, 0.1, 0.2], lambda: photo(eye()))
    hold.apply(0.3, photo())

    # when
    out = hold.apply(0.4, photo(eye()))

    # then
    assert not out.meets_condition()


def test_eye_hold_long_gap_resets():
    """이웃 프레임 간격이 EYE_HOLD_MAX_GAP_SEC 보다 크면 누적 초기화"""
    # given
    hold = EyeHold()
    hold.apply(0.0, photo(eye()))

    # when
    out = run(hold, [0.4, 0.5, 0.6], lambda: photo(eye()))

    # then
    assert not any(p.meets_condition() for p in out)


def test_eye_hold_time_not_increasing_resets():
    """촬영 시각이 늘지 않으면 예외 없이 누적 초기화"""
    # given
    hold = EyeHold()
    run(hold, [1.0, 1.1, 1.2], lambda: photo(eye()))

    # when
    out = hold.apply(1.2, photo(eye()))

    # then
    assert not out.meets_condition() and out.faces[0].scores["eye_hold"][0] == 0.0


def test_eye_hold_follows_faces_when_order_swaps():
    """검출 순서가 바뀌어도 박스로 같은 사람을 이어 봄"""
    # given
    hold = EyeHold()
    run(hold, [0.0, 0.1, 0.2], lambda: photo(eye(x=0), eye(closed=False, x=200)))

    # when
    out = hold.apply(0.3, photo(eye(closed=False, x=200), eye(x=0)))

    # then
    assert out.meets_condition()
    assert [f.scores["eye_hold"][2] for f in out.faces] == [False, True]


def test_eye_hold_other_person_blink_blocks_photo():
    """다른 사람이 짧게 감으면 그 사람은 허용하지 않아 사진은 조건 미충족"""
    # given
    hold = EyeHold()
    run(hold, [0.0, 0.1, 0.2], lambda: photo(eye(x=0), eye(closed=False, x=200)))

    # when
    out = hold.apply(0.3, photo(eye(x=0), eye(x=200)))

    # then
    assert out.faces[0].eyes_open and not out.faces[1].eyes_open
    assert not out.meets_condition()


def test_eye_hold_ambiguous_match_not_linked():
    """두 얼굴과 비슷하게 겹치면 잇지 않음 (누적 초기화)"""
    # given
    hold = EyeHold()
    run(hold, [0.0, 0.1, 0.2], lambda: photo(eye(x=0), eye(x=100)))

    # when
    out = hold.apply(0.3, photo(eye(x=49), eye(x=51)))

    # then
    assert not any(f.eyes_open for f in out.faces)


@pytest.mark.parametrize("face", [
    eye(visible=False),                                             # 안 보임
    Face((0, 0, 80, 100), True, None, True, None, scores={}),       # 눈 판정 불가
    eye(h=30),                                                      # MIN_FACE_H 미만
])
def test_eye_hold_unjudged_face_not_allowed_and_resets(face):
    """안 보임, 눈 판정 불가, 작은 얼굴은 허용하지 않고 누적 초기화"""
    # given
    hold = EyeHold()
    run(hold, [0.0, 0.1, 0.2], lambda: photo(eye()))

    # when
    out = hold.apply(0.3, photo(face))
    after = hold.apply(0.4, photo(eye()))

    # then
    assert not out.faces[0].scores["eye_hold"][2]
    assert not after.meets_condition()


def test_eye_hold_smile_still_required():
    """눈을 허용해도 웃음 조건은 그대로"""
    # given
    hold = EyeHold()

    # when
    out = run(hold, [0.0, 0.1, 0.2, 0.3], lambda: photo(eye(smile=False)))[-1]

    # then
    assert out.faces[0].eyes_open and not out.meets_condition()


def test_eye_hold_skips_without_eye_request():
    """eye 를 요청하지 않은 판정은 그대로 돌려주고 누적 초기화"""
    # given
    hold = EyeHold()
    run(hold, [0.0, 0.1, 0.2], lambda: photo(eye()))
    raw = photo(eye(), eye_req=False)

    # when
    out = hold.apply(0.3, raw)
    after = hold.apply(0.4, photo(eye()))

    # then
    assert out is raw and not after.meets_condition()


def test_eye_hold_counts_allowed():
    """allowed: 허용한 얼굴 x 프레임 수"""
    # given
    hold = EyeHold()

    # when
    run(hold, [0.0, 0.1, 0.2, 0.3, 0.4], lambda: photo(eye(x=0), eye(x=200)))

    # then
    assert hold.allowed == 4   # 0.3, 0.4 초 x 2명
