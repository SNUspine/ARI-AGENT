import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


LINE_COLOR = (0, 255, 255)  # yellow
DOT_COLOR = (0, 0, 255)     # red (BGR)
SVA_COLOR = (255, 165, 0)   # orange
C7UP_COLOR = (0, 255, 0)    # green (BGR)
BASE_SIZE = 512


def _sf(img_shape):
    return max(img_shape[0], img_shape[1]) / BASE_SIZE


def _overlay_rect(img, pt1, pt2, color=(0, 0, 0), alpha=0.45):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _line_intersect(p1, d1, p2, d2):
    """Find intersection of two lines defined by point + direction."""
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-10:
        return None
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    t = (dx * d2[1] - dy * d2[0]) / denom
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def _bgr_to_rgb(bgr):
    """Convert BGR tuple to RGB tuple."""
    return (bgr[2], bgr[1], bgr[0])


_FONT_CACHE = {}

def _load_font(font_size):
    if font_size in _FONT_CACHE:
        return _FONT_CACHE[font_size]
    candidates = [
        "malgun.ttf",
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    font = None
    for path in candidates:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[font_size] = font
    return font


def _put_text_pil(img, text, pos, font_size, color_bgr):
    """Draw text using PIL for Unicode support."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = _load_font(font_size)
    color_rgb = _bgr_to_rgb(color_bgr)
    draw.text(pos, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_cobb_angle(image, keypoints, measurements):
    img = image.copy()
    sf = _sf(img.shape)
    thin = max(1, int(round(1.0 * sf)))
    dot_r = max(2, int(round(2 * sf)))
    font_size = int(round(16 * sf))

    cobb = measurements["cobb_angle"]
    c2_slope = measurements["c2_slope"]
    c7_slope = measurements["c7_slope"]
    sva_px = measurements["sva_px"]
    sva_mm = measurements.get("sva_mm")

    c2a = np.array(keypoints["C2A"], dtype=np.float64)
    c2p = np.array(keypoints["C2P"], dtype=np.float64)
    c7a = np.array(keypoints["C7A"], dtype=np.float64)
    c7p = np.array(keypoints["C7P"], dtype=np.float64)
    c2_mid = (c2a + c2p) / 2.0
    c7_mid = (c7a + c7p) / 2.0

    # -- 1. Endplate lines (extend to image edges) --
    diag = np.sqrt(img.shape[0]**2 + img.shape[1]**2)

    def _extend_to_edge(pa, pp):
        dx, dy = pp[0] - pa[0], pp[1] - pa[1]
        length = np.sqrt(dx * dx + dy * dy) + 1e-9
        factor = diag / length
        return (int(round(pa[0] - dx * factor)), int(round(pa[1] - dy * factor))), \
               (int(round(pp[0] + dx * factor)), int(round(pp[1] + dy * factor)))

    c2e1, c2e2 = _extend_to_edge(c2a, c2p)
    c7e1, c7e2 = _extend_to_edge(c7a, c7p)
    cv2.line(img, c2e1, c2e2, LINE_COLOR, thin, cv2.LINE_AA)
    cv2.line(img, c7e1, c7e2, LINE_COLOR, thin, cv2.LINE_AA)

    # Keypoint dots (small, red)
    for pt in [c2a, c2p, c7a, c7p]:
        cv2.circle(img, (int(round(pt[0])), int(round(pt[1]))),
                   dot_r, DOT_COLOR, -1, cv2.LINE_AA)

    # C7UP dot (green) - right upper corner of C7
    if "C7UP" in keypoints:
        c7up = np.array(keypoints["C7UP"], dtype=np.float64)
        cv2.circle(img, (int(round(c7up[0])), int(round(c7up[1]))),
                   dot_r, C7UP_COLOR, -1, cv2.LINE_AA)

    # -- 2. Find intersection of the two endplate lines --
    c2_dir = (c2p[0] - c2a[0], c2p[1] - c2a[1])
    c7_dir = (c7p[0] - c7a[0], c7p[1] - c7a[1])
    ix_pt = _line_intersect(c2a, c2_dir, c7a, c7_dir)

    if ix_pt is not None:
        ix, iy = ix_pt

        # -- 3. Angle arc at intersection (between the two endplate lines) --
        arc_r = int(round(50 * sf))

        # Calculate angles of the endplate directions
        ang_c2 = np.degrees(np.arctan2(c2_dir[1], c2_dir[0]))
        ang_c7 = np.degrees(np.arctan2(c7_dir[1], c7_dir[0]))

        # Draw arc between the two angles (smaller arc)
        a1 = min(ang_c2, ang_c7)
        a2 = max(ang_c2, ang_c7)
        if a2 - a1 > 180:
            a1, a2 = a2, a1 + 360
        cv2.ellipse(img, (int(round(ix)), int(round(iy))),
                    (arc_r, arc_r), 0, a1, a2, LINE_COLOR, thin, cv2.LINE_AA)

    # -- 4. SVA (orange vertical + horizontal arrow to C7UP) --
    c2m = (int(round(c2_mid[0])), int(round(c2_mid[1])))
    if "C7UP" in keypoints:
        c7up = keypoints["C7UP"]
        c7up_pt = (int(round(c7up[0])), int(round(c7up[1])))
        # C2 중점에서 C7UP Y 레벨까지 수직선
        cv2.line(img, c2m, (c2m[0], c7up_pt[1]), SVA_COLOR, thin, cv2.LINE_AA)
        # C7UP까지 수평 화살표
        cv2.arrowedLine(img, (c2m[0], c7up_pt[1]), c7up_pt, SVA_COLOR, thin,
                        tipLength=0.15, line_type=cv2.LINE_AA)
    else:
        c7m = (int(round(c7_mid[0])), int(round(c7_mid[1])))
        cv2.line(img, c2m, (c2m[0], c7m[1]), SVA_COLOR, thin, cv2.LINE_AA)
        cv2.arrowedLine(img, (c2m[0], c7m[1]), c7m, SVA_COLOR, thin,
                        tipLength=0.15, line_type=cv2.LINE_AA)

    # -- 5. Measurement text block (top-right, semi-transparent) --
    if sva_mm is not None:
        sva_text = f"C2-C7 SVA: {sva_mm:.1f} mm"
    else:
        sva_text = f"C2-C7 SVA: {sva_px:.1f} px"
    # Cobb angle에 +/- 부호 표시
    cobb_sign = "+" if cobb >= 0 else ""
    lines = [
        (f"C2-C7 Cobb: {cobb_sign}{cobb:.1f}°", LINE_COLOR),
        (f"C2 Slope: {c2_slope:.1f}°", LINE_COLOR),
        (f"C7 Slope: {c7_slope:.1f}°", LINE_COLOR),
        (sva_text, SVA_COLOR),
    ]

    pad = int(6 * sf)
    lh = int(font_size * 1.5)

    # Calculate text width using PIL
    font = _load_font(font_size)

    max_tw = 0
    for text, _ in lines:
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        max_tw = max(max_tw, tw)

    bx = img.shape[1] - max_tw - pad * 3
    bx = max(pad, bx)
    by_base = pad * 2
    total_h = lh * len(lines)

    _overlay_rect(img,
                  (bx - pad, by_base - pad),
                  (bx + max_tw + pad, by_base + total_h + pad),
                  alpha=0.4)

    # Draw text using PIL
    for i, (text, color) in enumerate(lines):
        img = _put_text_pil(img, text, (bx, by_base + i * lh), font_size, color)

    return img
