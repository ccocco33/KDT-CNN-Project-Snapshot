LENS_SHAPES = ("ellipse", "rounded_rect", "aviator", "cat_eye")


def lens_polygon(cx, cy, a, b, angle_deg, shape, side, n=48):
    """렌즈 외곽 좌표 (정수 배열)
    - a, b: 반폭, 반높이. side: -1 오른눈(이미지 왼쪽), +1 왼눈(이미지 오른쪽). 캣아이의 바깥쪽 방향
    - ellipse: 타원 / rounded_rect: 초타원(n=4) / aviator: 아래가 넓은 물방울 / cat_eye: 바깥쪽 위가 올라감
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    if shape == "rounded_rect":
        x, y = a * np.sign(c) * np.abs(c) ** 0.5, b * np.sign(s) * np.abs(s) ** 0.5
    elif shape == "aviator":
        x, y = a * c * (1 + 0.18 * s), b * s   # 아래(s > 0)가 넓음
        y = np.where(s > 0, y * 1.15, y)
    elif shape == "cat_eye":
        x, y = a * c, b * s
        y = y - 0.35 * b * np.clip(c * side, 0, None) * (s < 0)   # 바깥쪽 위를 끌어올림
    else:
        x, y = a * c, b * s
    r = np.deg2rad(angle_deg)
    xr, yr = x * np.cos(r) - y * np.sin(r), x * np.sin(r) + y * np.cos(r)
    return np.stack([cx + xr, cy + yr], 1).round().astype(np.int32)


SUNGLASS_STYLES = {"dark": 0.40, "reflect": 0.35, "mirror": 0.10, "light": 0.15}   # 렌즈 종류와 뽑힐 확률
MIRROR_COLORS = ((40, 120, 200), (220, 170, 40), (190, 190, 190), (80, 160, 60))      # RGB: 파랑, 금색, 은색, 초록


def pick_sunglass_style(rng, allow_light=True):
    """SUNGLASS_STYLES 확률로 렌즈 종류 하나. allow_light=False 면 light 제외"""
    names = [k for k in SUNGLASS_STYLES if allow_light or k != "light"]
    p = np.array([SUNGLASS_STYLES[k] for k in names])
    return names[int(rng.choice(len(names), p=p / p.sum()))]


def _lens_reflections(shape, centers, w, h, dist, rng, strength):
    """렌즈 안 밝은 반사 (사각형 창/화면, 비스듬한 띠, 밝은 점). 0~1 세기 맵"""
    refl = np.zeros(shape, np.float32)
    for cx, cy in centers:
        for _ in range(int(rng.integers(1, 3))):
            kind = rng.integers(3)
            ox, oy = rng.uniform(-0.5, 0.4) * w, rng.uniform(-0.6, 0.3) * h
            if kind == 0:
                rw, rh = w * rng.uniform(0.3, 0.7), h * rng.uniform(0.25, 0.6)
                cv2.rectangle(refl, (int(cx + ox), int(cy + oy)), (int(cx + ox + rw), int(cy + oy + rh)), 1.0, -1)
            elif kind == 1:
                cv2.line(refl, (int(cx - w * 0.6), int(cy + oy + h * 0.4)), (int(cx + w * 0.4), int(cy + oy - h * 0.5)), 1.0,
                         max(1, int(h * rng.uniform(0.15, 0.35))))
            else:
                cv2.circle(refl, (int(cx + ox), int(cy + oy)), max(1, int(h * rng.uniform(0.1, 0.25))), 1.0, -1)
    return cv2.GaussianBlur(refl, (0, 0), max(0.4, dist * rng.uniform(0.02, 0.06))) * rng.uniform(*strength)


def add_sunglasses(img, right_eye, left_eye, rng, min_alpha=0.70, style="dark", bgr=False):
    """두 눈 위에 합성 선글라스. img: uint8 이미지, 눈 좌표는 img 픽셀 기준
    - 렌즈 모양 LENS_SHAPES 중 하나, 크기, 색, 불투명도, 반사광은 무작위
    - min_alpha: 렌즈 불투명도 하한 (dark). 감은 눈 이미지는 0.9 이상 (눈이 비치면 "뜸" 라벨과 모순)
    - style: 렌즈 종류 (SUNGLASS_STYLES, pick_sunglass_style 로 고름)
      - dark: 고르게 어두운 렌즈 + 위쪽 옅은 반사
      - reflect: 어두운 렌즈 (불투명도 0.95) + 밝은 반사 (실내 화면, 창문이 비친 모습)
      - mirror: 색 미러 렌즈 (불투명도 0.97, 위에서 아래로 어두워짐) + 밝은 반사
      - light: 옅은 렌즈 (불투명도 0.35 ~ 0.6) + 약한 반사. 눈이 비침 -> 눈 라벨은 바꾸지 않아야 함 (투명 안경과 같음)
    - bgr: img 가 BGR(OpenCV) 이면 True. 기본은 RGB (학습 노트북). 미러 렌즈 색의 채널 순서만 바뀜
    """
    if style != "dark":
        return _styled_sunglasses(img, right_eye, left_eye, rng, style, bgr)
    (rx, ry), (lx, ly) = right_eye, left_eye
    dist = math.hypot(lx - rx, ly - ry)
    angle = math.degrees(math.atan2(ly - ry, lx - rx))
    shape = LENS_SHAPES[rng.integers(len(LENS_SHAPES))]
    w = dist * rng.uniform(0.36, 0.46)       # 렌즈 반폭
    h = w * rng.uniform(0.62, 0.85)          # 렌즈 반높이
    dy = h * rng.uniform(-0.10, 0.15)        # 렌즈 중심 높이 (눈 기준)
    base = rng.uniform(5, 50)                # 렌즈 밝기 (어두움)
    tint = base + rng.uniform(-5, 1, 3) * rng.uniform(0, 4)   # 채널마다 조금씩 달라 갈색, 회색, 녹색 기운
    alpha = rng.uniform(min_alpha, 0.97)     # 렌즈 불투명도 (낮으면 눈이 살짝 비침)
    frame = tuple(int(v) for v in rng.uniform(0, 40, 3))
    thick = max(1, int(round(dist * 0.05)))

    lens = np.zeros(img.shape[:2], np.uint8)
    polys, centers = [], []
    for (ex, ey), side in (((rx, ry), -1), ((lx, ly), 1)):
        c = (ex, ey + dy)
        centers.append(c)
        poly = lens_polygon(c[0], c[1], w, h, angle, shape, side)
        polys.append(poly)
        cv2.fillPoly(lens, [poly], 255)
    out = img.astype(np.float32)
    m = (lens > 0)[..., None]
    # 위쪽이 더 진한 렌즈 (그라데이션): 렌즈 위 -> 아래로 불투명도 alpha -> alpha * 0.85
    ys = np.arange(img.shape[0], dtype=np.float32)[:, None, None]
    top = min(ry, ly) + dy - h
    grad = np.clip((ys - top) / (2 * h), 0, 1)
    a = alpha * (1 - 0.15 * grad)
    out = np.where(m, out * (1 - a) + np.clip(tint, 0, 255) * a, out)
    # 렌즈 위쪽 옅은 반사광
    shine = np.zeros(img.shape[:2], np.uint8)
    for cx, cy in centers:
        cv2.ellipse(shine, (int(cx), int(cy - h * 0.45)), (int(w * 0.6), max(1, int(h * 0.2))), angle, 0, 360, 255, -1)
    shine = cv2.GaussianBlur((shine & lens).astype(np.float32) / 255, (0, 0), max(0.5, dist * 0.05))[..., None]
    out = out + shine * rng.uniform(10, 45)
    out = np.clip(out, 0, 255).astype(np.uint8)
    cv2.polylines(out, polys, True, frame, thick, cv2.LINE_AA)
    cv2.line(out, (int(centers[0][0] + w * 0.9), int(centers[0][1])), (int(centers[1][0] - w * 0.9), int(centers[1][1])),
             frame, thick, cv2.LINE_AA)
    return out


def _styled_sunglasses(img, right_eye, left_eye, rng, style, bgr):
    """add_sunglasses 의 reflect, mirror, light 렌즈"""
    (rx, ry), (lx, ly) = right_eye, left_eye
    dist = math.hypot(lx - rx, ly - ry)
    angle = math.degrees(math.atan2(ly - ry, lx - rx))
    shape = LENS_SHAPES[rng.integers(len(LENS_SHAPES))]
    w = dist * rng.uniform(0.36, 0.46)
    h = w * rng.uniform(0.62, 0.85)
    dy = h * rng.uniform(-0.10, 0.15)
    lens = np.zeros(img.shape[:2], np.uint8)
    polys, centers = [], []
    for (ex, ey), side in (((rx, ry), -1), ((lx, ly), 1)):
        c = (ex, ey + dy)
        centers.append(c)
        poly = lens_polygon(c[0], c[1], w, h, angle, shape, side)
        polys.append(poly)
        cv2.fillPoly(lens, [poly], 255)
    m = (lens > 0).astype(np.float32)[..., None]
    out = img.astype(np.float32)
    if style == "light":
        a = rng.uniform(0.35, 0.6)
        out = out * (1 - m * a) + rng.uniform(20, 160, 3) * m * a
        r = _lens_reflections(img.shape[:2], centers, w, h, dist, rng, (0.3, 0.6))[..., None] * m
        out = out * (1 - r) + 255 * r
    elif style == "reflect":
        out = out * (1 - m * 0.95) + rng.uniform(5, 45) * m * 0.95
        r = _lens_reflections(img.shape[:2], centers, w, h, dist, rng, (0.6, 1.0))[..., None] * m
        out = out * (1 - r) + rng.uniform(170, 255, 3) * r
    elif style == "mirror":
        color = np.array(MIRROR_COLORS[rng.integers(len(MIRROR_COLORS))], np.float32)
        color = (color[::-1] if bgr else color) * rng.uniform(0.7, 1.1)
        ys = np.arange(img.shape[0], dtype=np.float32)[:, None, None]
        grad = np.clip((ys - (min(c[1] for c in centers) - h)) / (2 * h), 0, 1)
        out = out * (1 - m * 0.97) + color * (1.1 - 0.6 * grad) * m * 0.97
        r = _lens_reflections(img.shape[:2], centers, w, h, dist, rng, (0.6, 1.0))[..., None] * m
        out = out * (1 - r) + 255 * r
    else:
        raise ValueError(f"알 수 없는 선글라스 종류: {style}")
    out = np.clip(out, 0, 255).astype(np.uint8)
    frame = tuple(int(v) for v in (rng.uniform(0, 60, 3) if rng.random() < 0.7 else rng.uniform(120, 200, 3)))
    thick = max(1, int(round(dist * 0.05)))
    cv2.polylines(out, polys, True, frame, thick, cv2.LINE_AA)
    cv2.line(out, (int(centers[0][0] + w * 0.9), int(centers[0][1])), (int(centers[1][0] - w * 0.9), int(centers[1][1])),
             frame, thick, cv2.LINE_AA)
    return out


CLEAR_FRAME_COLORS = ((20, 20, 20), (50, 45, 45), (90, 60, 40), (130, 90, 60), (170, 170, 170), (190, 150, 60))   # RGB: 검정, 진회색, 갈색, 밝은 갈색, 은색, 금색


def add_clear_glasses(img, right_eye, left_eye, rng, bgr=False):
    """두 눈 위에 투명 안경 (테 + 옅은 렌즈 색 + 반사광). img: 이미지 uint8, 눈 좌표는 img 픽셀 기준
    - 눈이 비쳐 보이므로 라벨은 그대로 (뜬 눈은 뜸, 감은 눈은 감음). 안경테, 렌즈 색, 반사를 뜸/감음의 단서로 쓰지 않도록
    - 렌즈 모양 LENS_SHAPES 중 하나, 크기, 테 두께는 무작위. 테 색은 CLEAR_FRAME_COLORS 중 하나 (검정, 갈색, 은색, 금색 계열)
    - bgr: img 가 BGR(OpenCV) 이면 True. 기본은 RGB (학습 노트북). 테 색의 채널 순서만 바뀜
    - 렌즈 색: 불투명도 0 ~ 0.25
    - 반사광 (70%): 렌즈 안 무작위 위치의 흐린 타원, 색(초록, 보라, 흰색 등)과 세기(0.15 ~ 0.45) 무작위. 눈을 완전히 가리지 않음
    """
    (rx, ry), (lx, ly) = right_eye, left_eye
    dist = math.hypot(lx - rx, ly - ry)
    angle = math.degrees(math.atan2(ly - ry, lx - rx))
    shape = LENS_SHAPES[rng.integers(len(LENS_SHAPES))]
    w = dist * rng.uniform(0.38, 0.50)       # 렌즈 반폭 (선글라스보다 약간 큼)
    h = w * rng.uniform(0.70, 0.95)          # 렌즈 반높이
    dy = h * rng.uniform(-0.10, 0.15)        # 렌즈 중심 높이 (눈 기준)

    lens = np.zeros(img.shape[:2], np.uint8)
    polys, centers = [], []
    for (ex, ey), side in (((rx, ry), -1), ((lx, ly), 1)):
        c = (ex, ey + dy)
        centers.append(c)
        poly = lens_polygon(c[0], c[1], w, h, angle, shape, side)
        polys.append(poly)
        cv2.fillPoly(lens, [poly], 255)
    inside = (lens > 0).astype(np.float32)[..., None]
    out = img.astype(np.float32)
    # 옅은 렌즈 색
    a = rng.uniform(0.0, 0.25)
    out = out * (1 - inside * a) + rng.uniform(60, 255, 3) * inside * a
    # 반사광
    if rng.random() < 0.7:
        refl = np.zeros(img.shape[:2], np.float32)
        for cx, cy in centers:
            if rng.random() < 0.8:
                ox, oy = rng.uniform(-0.5, 0.5) * w, rng.uniform(-0.5, 0.5) * h
                axes = (max(1, int(w * rng.uniform(0.3, 0.8))), max(1, int(h * rng.uniform(0.2, 0.6))))
                cv2.ellipse(refl, (int(cx + ox), int(cy + oy)), axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        refl = cv2.GaussianBlur(refl, (0, 0), max(0.5, dist * 0.06)) * inside[..., 0]
        ra = (refl * rng.uniform(0.15, 0.45))[..., None]
        out = out * (1 - ra) + rng.uniform(80, 255, 3) * ra
    out = np.clip(out, 0, 255).astype(np.uint8)
    # 테, 브리지: 실제 안경테 색 + 약간의 흔들림
    base = CLEAR_FRAME_COLORS[rng.integers(len(CLEAR_FRAME_COLORS))]
    base = base[::-1] if bgr else base
    frame = tuple(int(np.clip(v + rng.uniform(-15, 15), 0, 255)) for v in base)
    thick = max(1, int(round(dist * rng.uniform(0.03, 0.05))))
    cv2.polylines(out, polys, True, frame, thick, cv2.LINE_AA)
    cv2.line(out, (int(centers[0][0] + w * 0.9), int(centers[0][1])), (int(centers[1][0] - w * 0.9), int(centers[1][1])),
             frame, thick, cv2.LINE_AA)
    return out
