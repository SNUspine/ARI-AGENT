import os
import re
import numpy as np
import cv2
from scipy import ndimage
from sympy import Point, Line
import onnxruntime as ort

from download_model import get_model_path


TARGET_HEIGHT = 768
PAD_MULTIPLE = 256
CLAHE_CLIP_LIMIT = 100
CLAHE_TILE_SIZE = 32
HEATMAP_THRESHOLD = 127
MIN_COMPONENT_AREA = 5
SCALE_BAR_MM = 50  # 스케일 바 기본값 (50mm)
MIN_SPINE_WIDTH = 150  # 최소 척추 영역 너비 (픽셀)

# C7 영역 추가 대조 보정 파라미터
C7_REGION_RATIO = 0.45   # 하위 45%를 C7 영역으로 처리
C7_CLAHE_CLIP   = 60     # 전역(100)보다 낮되 국소적으로 세밀하게
C7_CLAHE_TILE   = 16     # 전역(32)보다 작은 타일 → 더 세밀한 국소 대조


def detect_spine_regions(gray_img):
    """
    Detect multiple spine regions in a single image.
    Uses vertical projection to find dark vertical dividing lines.

    Returns: List of (x_start, x_end) tuples for each detected region.
             Returns [(0, width)] if only one region detected.
    """
    h, w = gray_img.shape[:2]

    # Skip if image is too narrow for multiple spines
    if w < MIN_SPINE_WIDTH * 2:
        return [(0, w)]

    # Calculate vertical projection (sum of pixel values along each column)
    # Exclude top 10% and bottom 10% to avoid text regions
    roi_top = int(h * 0.1)
    roi_bottom = int(h * 0.9)
    roi = gray_img[roi_top:roi_bottom, :]

    vertical_projection = np.mean(roi, axis=0)

    # Smooth the projection
    kernel_size = max(5, w // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(vertical_projection.reshape(1, -1), (kernel_size, 1), 0)[0]

    # Method 1: Look for very dark vertical bands (near-black dividers)
    # This catches explicit black gaps between images
    very_dark_threshold = 20  # Almost black
    is_very_dark = smoothed < very_dark_threshold

    # Find continuous very dark regions
    dark_dividers = []
    in_dark = False
    dark_start = 0

    for i in range(w):
        if is_very_dark[i] and not in_dark:
            dark_start = i
            in_dark = True
        elif not is_very_dark[i] and in_dark:
            dark_end = i
            dark_width = dark_end - dark_start
            center = (dark_start + dark_end) // 2
            # Divider should be in middle 70% of image
            if w * 0.15 < center < w * 0.85:
                # Any width is ok for very dark dividers
                if dark_width >= 5:
                    dark_dividers.append((dark_start, dark_end, center))
            in_dark = False

    # Method 2: If no very dark dividers, look for sharp brightness transitions
    if not dark_dividers:
        # Calculate gradient (brightness changes)
        gradient = np.abs(np.diff(smoothed))

        # Find significant drops followed by rises (valley pattern)
        mean_grad = np.mean(gradient)
        std_grad = np.std(gradient)
        high_gradient_threshold = mean_grad + 2 * std_grad

        # Look for sharp transitions in middle portion
        for i in range(int(w * 0.2), int(w * 0.8)):
            # Check for significant brightness drop
            if gradient[i] > high_gradient_threshold:
                # Look for corresponding rise nearby
                for j in range(i + 10, min(i + 150, w - 1)):
                    if gradient[j] > high_gradient_threshold:
                        # Found potential divider between i and j
                        center = (i + j) // 2
                        # Verify the middle region is dark
                        middle_brightness = np.mean(smoothed[i:j+1])
                        side_brightness = (smoothed[max(0, i-20)] + smoothed[min(w-1, j+20)]) / 2
                        if middle_brightness < side_brightness * 0.5:
                            dark_dividers.append((i, j+1, center))
                        break

    # If no dividers found, return single region
    if not dark_dividers:
        return [(0, w)]

    # Sort by center position and remove overlapping dividers
    dark_dividers.sort(key=lambda x: x[2])

    # Remove overlapping dividers (keep first)
    filtered_dividers = []
    last_end = -100
    for start, end, center in dark_dividers:
        if start > last_end + 50:  # At least 50px gap between dividers
            filtered_dividers.append((start, end, center))
            last_end = end

    # Build spine regions from dividers
    regions = []
    prev_end = 0

    for dark_start, dark_end, center in filtered_dividers:
        if dark_start - prev_end >= MIN_SPINE_WIDTH:
            regions.append((prev_end, dark_start))
        prev_end = dark_end

    # Add last region
    if w - prev_end >= MIN_SPINE_WIDTH:
        regions.append((prev_end, w))

    # If only one valid region, return full image
    if len(regions) <= 1:
        return [(0, w)]

    return regions


def split_image_for_regions(color_img, regions):
    """
    Split a color image into multiple sub-images based on detected regions.

    Returns: List of (sub_image, x_offset) tuples
    """
    sub_images = []
    for x_start, x_end in regions:
        sub_img = color_img[:, x_start:x_end].copy()
        sub_images.append((sub_img, x_start))
    return sub_images


def extract_metadata_from_image(gray_img):
    """
    일반 이미지에서 OCR로 메타데이터 추출.
    - 우측 상단: 날짜 (YYYY-MM-DD), 성별/나이 (M/F 044Y)
    - 좌측 하단: study description (C-spine Lat(neut))
    Returns: dict (sex, age, study_date, study_id, study_description)
    """
    try:
        import pytesseract
        import sys
        # Tesseract 경로 설정 (Windows) - 번들 버전 우선
        tesseract_paths = []
        if getattr(sys, 'frozen', False):
            exe_dir = os.path.dirname(sys.executable)
            # 1. _internal 폴더 내부의 Tesseract-OCR (번들 버전)
            tesseract_paths.append(os.path.join(exe_dir, '_internal', 'Tesseract-OCR', 'tesseract.exe'))
        # 2. 시스템 설치 버전
        tesseract_paths.append(r'C:\Program Files\Tesseract-OCR\tesseract.exe')

        for tesseract_path in tesseract_paths:
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                break
    except ImportError:
        return None

    h, w = gray_img.shape[:2]
    result = {
        "sex": "",
        "age": "",
        "study_date": "",
        "study_id": "",
        "study_description": "",
    }

    # 우측 상단 영역 (상위 15%, 우측 40%) - 기본
    top_right = gray_img[0:int(h * 0.15), int(w * 0.6):]
    # 좌측 하단 영역 (하위 10%, 좌측 40%)
    bottom_left = gray_img[int(h * 0.9):, 0:int(w * 0.4)]

    # 이진화 전처리 (OCR 정확도 향상)
    _, top_right_bin = cv2.threshold(top_right, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bottom_left_bin = cv2.threshold(bottom_left, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        # 우측 상단 OCR (이진화 적용)
        top_text = pytesseract.image_to_string(top_right_bin, config='--psm 6')

        # 날짜 추출 (DOB 제외)
        for line in top_text.split('\n'):
            # DOB/OB 라인 건너뛰기
            if re.search(r'[DO]O?B[:\s]', line, re.IGNORECASE):
                continue

            # 1차: 정확한 YYYY-MM-DD (2020~2030 범위)
            date_match = re.search(r'(20[2-3]\d)-(\d{1,2})-(\d{1,2})', line)
            if date_match:
                y, m, d = date_match.groups()
                result["study_date"] = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
                break

            # 2차: Y/W + 4자리 (Y1925-12-3 → 2025-12-03)
            noisy_year = re.search(r'[YyWw]\d(\d{3})-(\d{1,2})-(\d{1,2})', line)
            if noisy_year:
                rest, m, d = noisy_year.groups()
                year = '20' + rest[-2:]  # 925 → 2025
                result["study_date"] = f"{year}-{m.zfill(2)}-{d.zfill(2)}"
                break

            # 3차: 연도 앞자리 누락 (026-02-02 → 2026-02-02)
            partial_year = re.search(r'^0?(\d{2})-(\d{1,2})-(\d{1,2})', line)
            if partial_year:
                y, m, d = partial_year.groups()
                result["study_date"] = f"20{y}-{m.zfill(2)}-{d.zfill(2)}"
                break

            # 4차: YYYY-MM (일자 누락, 01로 설정)
            date_no_day = re.search(r'(20[2-3]\d)-(\d{1,2})(?![0-9-])', line)
            if date_no_day:
                y, m = date_no_day.groups()
                result["study_date"] = f"{y}-{m.zfill(2)}-01"
                break

        # 성별/나이 패턴: M044Y 또는 F065Y
        # OCR 오인식 패턴:
        # - M -> W, VV, N, H, IVI, WW
        # - F -> P, f, R, r
        # - 0 -> O, 6 -> O, 5 -> S

        # 1차: 정확한 패턴 M/F + 0 + 숫자 + Y (예: M044Y, F065Y)
        sex_age_match = re.search(r'([MF])\s*0?(\d{2,3}Y)', top_text)
        if sex_age_match:
            result["sex"] = sex_age_match.group(1)
            age_num = sex_age_match.group(2)
            if len(age_num) == 3:
                age_num = '0' + age_num
            result["age"] = age_num
        else:
            # 2차: M/F + 공백 + O/0 + 2자리숫자 (예: "F O53" -> F, 053)
            space_pattern = re.search(r'([MF])\s+[O0](\d{2})', top_text)
            if space_pattern:
                result["sex"] = space_pattern.group(1)
                result["age"] = '0' + space_pattern.group(2) + 'Y'
            else:
                # 3차: OCR 오류 대응 - 줄 시작의 P/W 등 + O/0 + 문자 + Y + ID
                # "POO SY 25828182" -> P가 sex (F로 변환)
                # "WOO 44Y 25828182" -> W가 sex (M로 변환)
                id_pattern = re.search(r'^([PpFfRrWwMmNnHhVvIi|1])[O0]+\s*[A-Za-z0-9]*[SsYy]\s+\d{7,}', top_text, re.MULTILINE)
                if id_pattern:
                    sex_char = id_pattern.group(1).upper()
                    if sex_char in ['M', 'W', 'N', 'H', 'V', 'I', '|', '1']:
                        result["sex"] = 'M'
                    elif sex_char in ['F', 'P', 'R']:
                        result["sex"] = 'F'
                else:
                    # IVI, VV 등 M 오인식 패턴
                    m_pattern = re.search(r'^([|lI1][VvWw][|lI1]|[VvWw]{2})[O0]+\s*[A-Za-z0-9]*[SsYy]\s+\d{7,}', top_text, re.MULTILINE)
                    if m_pattern:
                        result["sex"] = 'M'

        # 성별을 못 찾으면 원본 이미지로 재시도 (이진화가 일부 텍스트 손상 가능)
        if not result["sex"]:
            top_text_orig = pytesseract.image_to_string(top_right, config='--psm 6')
            # 3차 패턴 재시도
            id_pattern = re.search(r'^([PpFfRrWwMmNnHhVvIi|1])[O0]+\s*[A-Za-z0-9]*[SsYy]\s+\d{7,}', top_text_orig, re.MULTILINE)
            if id_pattern:
                sex_char = id_pattern.group(1).upper()
                if sex_char in ['M', 'W', 'N', 'H', 'V', 'I', '|', '1']:
                    result["sex"] = 'M'
                elif sex_char in ['F', 'P', 'R']:
                    result["sex"] = 'F'
            else:
                # IVI, VV 등 M 오인식 패턴
                m_pattern = re.search(r'^([|lI1][VvWw][|lI1]|[VvWw]{2})[O0]+\s*[A-Za-z0-9]*[SsYy]\s+\d{7,}', top_text_orig, re.MULTILINE)
                if m_pattern:
                    result["sex"] = 'M'

        # 그래도 못 찾으면 더 넓은 영역으로 재시도 (원본 + 이진화 둘 다)
        if not result["sex"]:
            top_right_wide = gray_img[0:int(h * 0.20), int(w * 0.5):]

            # 원본과 이진화 둘 다 시도
            for img_version in [top_right_wide, cv2.threshold(top_right_wide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]]:
                top_text_wide = pytesseract.image_to_string(img_version, config='--psm 6')

                # M/F + 0 + 2자리 나이 + ID번호 연속 (예: F0595620619, W0445620619)
                continuous_match = re.search(r'([MFWwNnHh])0(\d{2})\d{6,}', top_text_wide)
                if continuous_match:
                    sex_char = continuous_match.group(1).upper()
                    if sex_char in ['M', 'W', 'N', 'H']:
                        result["sex"] = 'M'
                    else:
                        result["sex"] = 'F'
                    result["age"] = '0' + continuous_match.group(2) + 'Y'
                    break

                # F가 i=, l=, |= 등으로 인식된 경우 (예: "i= 059405620619")
                f_misread = re.search(r'[il|=]+\s*0(\d{2})\d{6,}', top_text_wide)
                if f_misread:
                    result["sex"] = 'F'
                    result["age"] = '0' + f_misread.group(1) + 'Y'
                    break

                # M이 W, N, H, IVI, VV 등으로 인식된 경우
                # W0, N0, H0, IVI0, VV0 + 나이 + ID
                m_misread = re.search(r'([WwNnHh]|[|lI1][VvWw][|lI1]|[VvWw]{2})\s*0(\d{2})\d{6,}', top_text_wide)
                if m_misread:
                    result["sex"] = 'M'
                    result["age"] = '0' + m_misread.group(2) + 'Y'
                    break

                # M이 줄 시작에서 W, N 등으로 인식 + 공백 + 나이 패턴
                m_misread2 = re.search(r'^([WwNnHh]|[|lI1][VvWw][|lI1]|[VvWw]{2})\s*[O0]?(\d{2})[SsYy]', top_text_wide, re.MULTILINE)
                if m_misread2:
                    result["sex"] = 'M'
                    result["age"] = '0' + m_misread2.group(2) + 'Y'
                    break

        # 좌측 하단 OCR
        bottom_text = pytesseract.image_to_string(bottom_left_bin, config='--psm 6')

        # C-spine 관련 텍스트 찾기 (가장 완전한 패턴 우선)
        lines = [l.strip() for l in bottom_text.strip().split('\n') if l.strip()]
        # "C-spine Lat(neut)" 같은 완전한 설명 찾기
        for line in lines:
            if re.search(r'C-?spine', line, re.IGNORECASE):
                result["study_description"] = line
                break
        else:
            # C-spine 없으면 Lat(...) 패턴 찾기
            desc_match = re.search(r'(Lat\s*\([^)]+\))', bottom_text, re.IGNORECASE)
            if desc_match:
                result["study_description"] = desc_match.group(0).strip()
            elif lines:
                result["study_description"] = lines[-1]
    except Exception:
        pass

    # 값이 하나라도 있으면 반환
    if any(result.values()):
        return result
    return None


def detect_scale_bar(gray_img):
    """
    이미지 왼쪽 영역에서 스케일 바(밝은 수직선)를 감지하여 mm/pixel 비율 반환.
    감지 실패 시 None 반환.
    """
    h, w = gray_img.shape[:2]
    # 왼쪽 12% 영역만 검사
    left_width = int(w * 0.12)
    left_region = gray_img[:, :left_width]

    # 밝은 픽셀 이진화 (threshold)
    _, binary = cv2.threshold(left_region, 180, 255, cv2.THRESH_BINARY)

    # 수직선 감지를 위해 세로로 긴 커널로 모폴로지 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 컨투어 찾기
    contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 가장 긴 수직선 찾기 (높이 기준)
    max_height = 0
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # 수직선: 높이가 너비보다 훨씬 크고, 최소 높이 조건
        if ch > cw * 5 and ch > h * 0.05:
            if ch > max_height:
                max_height = ch

    if max_height == 0:
        return None

    # 50mm = max_height pixels
    mm_per_pixel = SCALE_BAR_MM / max_height
    return mm_per_pixel


def load_image(filepath):
    """
    이미지 로드. DICOM의 경우 Pixel Spacing과 메타데이터도 함께 반환.
    Returns: (image, pixel_spacing_mm, dicom_info)
    - pixel_spacing_mm: None 또는 float (mm/pixel)
    - dicom_info: None 또는 dict (sex, age, study_date, study_id, study_description)
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".dcm":
        try:
            import pydicom
            ds = pydicom.dcmread(filepath)
            img = ds.pixel_array.astype(np.float64)
            if hasattr(ds, "PhotometricInterpretation"):
                if ds.PhotometricInterpretation == "MONOCHROME1":
                    img = img.max() - img
            # Pixel Spacing 추출 (mm 단위)
            pixel_spacing = None
            if hasattr(ds, "PixelSpacing") and ds.PixelSpacing:
                pixel_spacing = float(ds.PixelSpacing[0])
            elif hasattr(ds, "ImagerPixelSpacing") and ds.ImagerPixelSpacing:
                pixel_spacing = float(ds.ImagerPixelSpacing[0])
            # DICOM 메타데이터 추출
            dicom_info = {
                "sex": getattr(ds, "PatientSex", "") or "",
                "age": getattr(ds, "PatientAge", "") or "",
                "study_date": getattr(ds, "StudyDate", "") or "",
                "study_id": getattr(ds, "StudyID", "") or "",
                "study_description": getattr(ds, "StudyDescription", "") or "",
            }
            return img, pixel_spacing, dicom_info
        except ImportError:
            raise RuntimeError("pydicom is required for DICOM files. Install with: pip install pydicom")
    else:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            # 한글 파일명 fallback
            img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Cannot read image: {filepath}")
        # 일반 이미지에서 OCR로 메타데이터 추출 시도
        if img.ndim == 2:
            gray_for_ocr = img
        elif img.ndim == 3 and img.shape[2] >= 3:
            gray_for_ocr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_for_ocr = img
        ocr_info = extract_metadata_from_image(gray_for_ocr)
        return img, None, ocr_info


def to_grayscale_uint8(img):
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        return img
    p1, p99 = np.percentile(img, [1, 99])
    if p99 - p1 < 1e-6:
        p99 = p1 + 1
    img_norm = np.clip((img.astype(np.float64) - p1) / (p99 - p1) * 255, 0, 255)
    return img_norm.astype(np.uint8)


def auto_crop_screen_photo(gray_img, min_keep_ratio=0.4, min_crop_gain=0.05):
    """
    Remove dark borders (desk, monitor bezel) from phone photos of PACS screens.
    Samples outer edge brightness to detect desk/bezel and sets adaptive threshold.
    Returns (cropped_gray, (x1, y1, x2, y2)) or (original_gray, None).
    """
    h, w = gray_img.shape[:2]

    # Sample outermost ~3% strip on each edge to measure border brightness
    bh = max(5, h // 30)
    bw = max(5, w // 30)
    top_m    = float(np.mean(gray_img[:bh, :]))
    bottom_m = float(np.mean(gray_img[-bh:, :]))
    left_m   = float(np.mean(gray_img[:, :bw]))
    right_m  = float(np.mean(gray_img[:, -bw:]))

    darkest_edge = min(top_m, bottom_m, left_m, right_m)

    # Skip if no edge is very dark — not a phone photo / desk border scenario
    if darkest_edge > 20:
        return gray_img, None

    # Adaptive threshold: just above the measured border darkness
    dark_thresh = max(darkest_edge + 12, 15)
    dark_thresh = min(dark_thresh, 30)

    # Row/column mean profiles — scan inward from each edge
    row_means = np.mean(gray_img, axis=1).astype(np.float32)
    col_means = np.mean(gray_img, axis=0).astype(np.float32)

    y1 = 0
    while y1 < h - 1 and row_means[y1] <= dark_thresh:
        y1 += 1
    y2 = h - 1
    while y2 > y1 and row_means[y2] <= dark_thresh:
        y2 -= 1
    y2 += 1

    x1 = 0
    while x1 < w - 1 and col_means[x1] <= dark_thresh:
        x1 += 1
    x2 = w - 1
    while x2 > x1 and col_means[x2] <= dark_thresh:
        x2 -= 1
    x2 += 1

    crop_h, crop_w = y2 - y1, x2 - x1
    if crop_h < h * min_keep_ratio or crop_w < w * min_keep_ratio:
        return gray_img, None
    removed = 1.0 - (crop_h * crop_w) / (h * w)
    if removed < min_crop_gain:
        return gray_img, None

    return gray_img[y1:y2, x1:x2], (x1, y1, x2, y2)


def should_invert(gray_img):
    """
    이미지가 역상인지 판단한다.
    X선 배경(공기)은 어두운 것이 정상.
    PACS가 추가한 인위적 검정 테두리(< 10)를 제외한
    외곽 20% 영역 평균이 128 초과이면 역상으로 판단.
    Returns True if inversion is needed.
    """
    h, w = gray_img.shape[:2]
    margin_h, margin_w = int(h * 0.2), int(w * 0.2)
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:margin_h, :] = True
    border_mask[h - margin_h:, :] = True
    border_mask[:, :margin_w] = True
    border_mask[:, w - margin_w:] = True
    border_pixels = gray_img[border_mask]
    valid = border_pixels[border_pixels > 10]
    if len(valid) < 100:
        return False
    return valid.mean() > 128


def correct_inversion(gray_img):
    """역상이면 반전, 아니면 그대로 반환."""
    if should_invert(gray_img):
        return 255 - gray_img
    return gray_img


def resize_height(img, target_height):
    h, w = img.shape[:2]
    if h == target_height:
        return img, 1.0
    scale = target_height / h
    new_w = int(round(w * scale))
    resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def pad_to_multiple(img, multiple):
    h, w = img.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    padded = np.zeros((new_h, new_w), dtype=img.dtype)
    padded[:h, :w] = img
    return padded, (pad_h, pad_w)


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
    return clahe.apply(img)


def enhance_c7_region(clahe_img, padded_orig):
    """
    C7 경계 부근(하위 45%)의 밝기/대조를 국소적으로 추가 보정.

    전역 CLAHE(tile=32, clip=100) 후, C7이 위치하는 하부 영역에
    더 작은 타일(16)로 CLAHE를 재적용하여 겹치는 뼈와의 대조를 개선.
    원본 padded 이미지를 기반으로 처리해 CLAHE 이중 누적을 방지.
    """
    h = padded_orig.shape[0]
    split_y = int(h * (1.0 - C7_REGION_RATIO))
    lower = padded_orig[split_y:, :]
    clahe_local = cv2.createCLAHE(
        clipLimit=C7_CLAHE_CLIP,
        tileGridSize=(C7_CLAHE_TILE, C7_CLAHE_TILE)
    )
    result = clahe_img.copy()
    result[split_y:, :] = clahe_local.apply(lower)
    return result


def preprocess(filepath, force_invert=False, force_hflip=False):
    raw, pixel_spacing, dicom_info = load_image(filepath)
    gray = correct_inversion(to_grayscale_uint8(raw))
    if force_invert:
        gray = 255 - gray
    if force_hflip:
        gray = cv2.flip(gray, 1)  # 좌우 반전
    original_h, original_w = gray.shape[:2]
    resized, scale = resize_height(gray, TARGET_HEIGHT)
    resized_h, resized_w = resized.shape[:2]
    padded, (pad_h, pad_w) = pad_to_multiple(resized, PAD_MULTIPLE)
    clahe_img = apply_clahe(padded)
    clahe_img = enhance_c7_region(clahe_img, padded)
    tensor = clahe_img.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, np.newaxis, :, :]
    if raw.ndim == 2:
        original_color = cv2.cvtColor(to_grayscale_uint8(raw), cv2.COLOR_GRAY2BGR)
    elif raw.ndim == 3 and raw.shape[2] == 3:
        if raw.dtype != np.uint8:
            original_color = cv2.cvtColor(to_grayscale_uint8(raw), cv2.COLOR_GRAY2BGR)
        else:
            original_color = raw.copy()
    else:
        original_color = cv2.cvtColor(to_grayscale_uint8(raw), cv2.COLOR_GRAY2BGR)
    return tensor, original_color, scale, resized_h, resized_w, original_h, original_w, pixel_spacing, dicom_info


def heatmap2points(heatmaps, affinity_maps, threshold=None):
    if threshold is None:
        threshold = HEATMAP_THRESHOLD
    num_keypoints = heatmaps.shape[0]
    assert num_keypoints == 4
    keypoints_per_channel = []
    for ch in range(num_keypoints):
        hm = heatmaps[ch]
        binary = (hm > threshold).astype(np.uint8)
        labeled, num_features = ndimage.label(binary)
        candidates = []
        for i in range(1, num_features + 1):
            component = (labeled == i)
            area = component.sum()
            if area < MIN_COMPONENT_AREA:
                continue
            weighted = hm.astype(np.float64) * component
            total = weighted.sum()
            if total < 1e-6:
                continue
            ys, xs = np.where(component)
            cy = (ys.astype(np.float64) * weighted[ys, xs]).sum() / total
            cx = (xs.astype(np.float64) * weighted[ys, xs]).sum() / total
            candidates.append((cx, cy, total))
        keypoints_per_channel.append(candidates)

    pair_indices = [(0, 1), (2, 3)]
    pairs = []
    for a_ch, b_ch in pair_indices:
        a_candidates = keypoints_per_channel[a_ch]
        b_candidates = keypoints_per_channel[b_ch]
        if not a_candidates or not b_candidates:
            return None
        best_pair = None
        best_score = -1
        for ac in a_candidates:
            for bc in b_candidates:
                ax, ay, a_conf = ac
                bx, by, b_conf = bc
                score = score_pair_affinity(ax, ay, bx, by, affinity_maps, a_conf, b_conf)
                if score > best_score:
                    best_score = score
                    best_pair = ((ax, ay), (bx, by))
        if best_pair is None:
            return None
        pairs.append(best_pair)

    c2a, c2p = pairs[0]
    c7a, c7p = pairs[1]
    return {"C2A": c2a, "C2P": c2p, "C7A": c7a, "C7P": c7p}


def heatmap2points_partial(heatmaps, affinity_maps):
    """
    Detect keypoints with partial detection support.
    Returns keypoints dict even if only C2 or C7 is detected.
    Returns None only if neither C2 nor C7 can be detected.
    """
    num_keypoints = heatmaps.shape[0]
    assert num_keypoints == 4
    keypoints_per_channel = []
    for ch in range(num_keypoints):
        hm = heatmaps[ch]
        binary = (hm > HEATMAP_THRESHOLD).astype(np.uint8)
        labeled, num_features = ndimage.label(binary)
        candidates = []
        for i in range(1, num_features + 1):
            component = (labeled == i)
            area = component.sum()
            if area < MIN_COMPONENT_AREA:
                continue
            weighted = hm.astype(np.float64) * component
            total = weighted.sum()
            if total < 1e-6:
                continue
            ys, xs = np.where(component)
            cy = (ys.astype(np.float64) * weighted[ys, xs]).sum() / total
            cx = (xs.astype(np.float64) * weighted[ys, xs]).sum() / total
            candidates.append((cx, cy, total))
        keypoints_per_channel.append(candidates)

    result = {}

    # Try to detect C2 pair (channels 0, 1)
    c2a_candidates = keypoints_per_channel[0]
    c2p_candidates = keypoints_per_channel[1]
    if c2a_candidates and c2p_candidates:
        best_pair = None
        best_score = -1
        for ac in c2a_candidates:
            for bc in c2p_candidates:
                ax, ay, a_conf = ac
                bx, by, b_conf = bc
                score = score_pair_affinity(ax, ay, bx, by, affinity_maps, a_conf, b_conf)
                if score > best_score:
                    best_score = score
                    best_pair = ((ax, ay), (bx, by))
        if best_pair is not None:
            result["C2A"] = best_pair[0]
            result["C2P"] = best_pair[1]

    # Try to detect C7 pair (channels 2, 3)
    c7a_candidates = keypoints_per_channel[2]
    c7p_candidates = keypoints_per_channel[3]
    if c7a_candidates and c7p_candidates:
        best_pair = None
        best_score = -1
        for ac in c7a_candidates:
            for bc in c7p_candidates:
                ax, ay, a_conf = ac
                bx, by, b_conf = bc
                score = score_pair_affinity(ax, ay, bx, by, affinity_maps, a_conf, b_conf)
                if score > best_score:
                    best_score = score
                    best_pair = ((ax, ay), (bx, by))
        if best_pair is not None:
            result["C7A"] = best_pair[0]
            result["C7P"] = best_pair[1]

    # Return None only if neither C2 nor C7 detected
    if not result:
        return None

    return result


def score_pair_affinity(ax, ay, bx, by, affinity_maps, a_conf, b_conf):
    num_samples = 10
    xs = np.linspace(ax, bx, num_samples).astype(int)
    ys = np.linspace(ay, by, num_samples).astype(int)
    h, w = affinity_maps.shape[1], affinity_maps.shape[2]
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    dx = bx - ax
    dy = by - ay
    length = np.sqrt(dx * dx + dy * dy) + 1e-6
    ux, uy = dx / length, dy / length
    aff_score = 0
    for i in range(num_samples):
        vx = affinity_maps[0, ys[i], xs[i]]
        vy = affinity_maps[1, ys[i], xs[i]]
        aff_score += vx * ux + vy * uy
    return aff_score / num_samples + (a_conf + b_conf) * 0.0001


def should_flip_horizontal(keypoints):
    """
    검출된 키포인트 배치가 좌우 반전 이미지에서 유래했는지 판단.

    표준 측면 경추 X선: 환자가 오른쪽을 향해 anterior(A)의 x좌표 > posterior(P) x좌표.
    C2, C7 모두 A.x < P.x 이면 좌우 반전 이미지로 판단하여 True 반환.
    4개 키포인트가 모두 없으면 False 반환.
    """
    if not all(k in keypoints for k in ("C2A", "C2P", "C7A", "C7P")):
        return False
    c2_flip = keypoints["C2A"][0] < keypoints["C2P"][0]
    c7_flip = keypoints["C7A"][0] < keypoints["C7P"][0]
    return c2_flip and c7_flip


def calculate_slope(pa, pp):
    dx = float(pp[0] - pa[0])
    dy = float(pp[1] - pa[1])
    angle_deg = np.degrees(np.arctan2(dy, dx))
    return round(angle_deg, 1)


def calculate_sva(keypoints, mm_per_pixel=None):
    """
    C2-C7 SVA: C2 중점에서 내린 수직선과 C7UP 사이의 수평 거리.
    양수 = C7UP가 C2 중점보다 앞쪽 (anterior)
    """
    c2_mid_x = (keypoints["C2A"][0] + keypoints["C2P"][0]) / 2.0
    c7up_x = keypoints["C7UP"][0]
    pixel_dist = c7up_x - c2_mid_x  # 양수 = anterior
    sva_px = round(pixel_dist, 1)
    if mm_per_pixel is not None:
        sva_mm = round(pixel_dist * mm_per_pixel, 1)
        return sva_px, sva_mm
    return sva_px, None


def _find_superior_endplate_dist(gray_img, anchor_pt, direction, width):
    """
    anchor_pt에서 direction 방향으로 intensity profile을 분석하여
    superior endplate까지의 추정 거리를 반환.

    척추체(밝음) → 추간판/연부조직(어둠)으로의 전환 지점을 탐색.
    실패 시 None 반환.
    """
    h, w = gray_img.shape[:2]
    d_start = max(3, int(width * 0.40))
    d_end   = int(width * 1.50)

    profile, dists = [], []
    for d in range(d_start, d_end + 1):
        pt = np.array(anchor_pt, dtype=np.float64) + direction * d
        px, py = int(round(pt[0])), int(round(pt[1]))
        if not (1 <= px < w - 1 and 1 <= py < h - 1):
            break
        # 3×3 평균으로 노이즈 억제
        val = float(gray_img[py - 1:py + 2, px - 1:px + 2].mean())
        profile.append(val)
        dists.append(d)

    if len(profile) < 6:
        return None

    vals = np.array(profile, dtype=np.float32)
    # 3-point moving average smoothing
    kernel = np.ones(3, dtype=np.float32) / 3
    vals_sm = np.convolve(vals, kernel, mode='same')

    # 음의 gradient = intensity 감소 지점 (body → disc 전환)
    neg_grad = -np.gradient(vals_sm.astype(np.float64))
    peak_idx = int(np.argmax(neg_grad))

    # 최소 gradient 크기 확인 (noise와 구분)
    if neg_grad[peak_idx] < 3.0:
        return None

    # mode='same' convolution 경계 효과로 인한 극단값 제외
    if peak_idx == 0 or peak_idx == len(neg_grad) - 1:
        return None

    return float(dists[peak_idx])


def _refine_c7up_corner(gray_img, estimated_pt, search_radius):
    """
    C7UP 전용 코너 검출.
    수직 엣지(posterior wall)와 수평 엣지(superior endplate)가
    동시에 강한 지점 = corner score 최대 지점을 반환.
    Generic Shi-Tomasi 대신 |Gx| × |Gy| 응답으로 방향성 필터링.
    """
    h, w = gray_img.shape[:2]
    cx, cy = int(round(estimated_pt[0])), int(round(estimated_pt[1]))

    x1 = max(0, cx - search_radius)
    y1 = max(0, cy - search_radius)
    x2 = min(w, cx + search_radius)
    y2 = min(h, cy + search_radius)

    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    roi = gray_img[y1:y2, x1:x2]

    # 국소 CLAHE로 엣지 강화
    clahe = cv2.createCLAHE(clipLimit=40, tileGridSize=(4, 4))
    roi_enh = clahe.apply(roi)
    roi_f = roi_enh.astype(np.float32)

    # Sobel gradients
    gx = cv2.Sobel(roi_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_f, cv2.CV_32F, 0, 1, ksize=3)

    # |Gx| × |Gy|: 수직 엣지와 수평 엣지가 동시에 강한 지점이 진짜 코너
    corner_map = np.abs(gx) * np.abs(gy)
    corner_map = cv2.GaussianBlur(corner_map, (3, 3), 0)

    max_val = float(corner_map.max())
    mean_val = float(corner_map.mean())
    if max_val < 50 or max_val < mean_val * 3:
        return None

    best_y, best_x = np.unravel_index(corner_map.argmax(), corner_map.shape)
    return (float(best_x + x1), float(best_y + y1))


def estimate_c7up(keypoints, gray_img=None):
    """
    C7UP (superior posterior corner) 추정.
    1차: C7P에서 intensity profile로 superior endplate 위치 추정
         (척추체 밝음→추간판 어둠 전환 지점 탐색, 실패 시 너비×0.8 fallback)
    2차: |Gx|×|Gy| 기반 방향성 코너 검출로 refine
         (posterior wall + superior endplate 교차점 특정)
    """
    c7a = np.array(keypoints["C7A"])
    c7p = np.array(keypoints["C7P"])

    # C7A -> C7P 벡터
    vec = c7p - c7a
    width = np.linalg.norm(vec)

    # 수직 벡터 (위쪽 방향: y 감소)
    perp = np.array([-vec[1], vec[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-9)

    # 위쪽 방향 확인 (y가 감소하는 방향)
    if perp[1] > 0:
        perp = -perp

    # 1차: intensity profile로 추체 높이 추정, 실패 시 너비×0.8 사용
    # raw gray는 CLAHE 미적용일 수 있으므로 국소 대조 향상 후 전달
    height = width * 0.8
    if gray_img is not None:
        _clahe_est = cv2.createCLAHE(clipLimit=40, tileGridSize=(16, 16))
        gray_enhanced = _clahe_est.apply(gray_img)
        dist = _find_superior_endplate_dist(gray_enhanced, c7p, perp, width)
        if dist is not None:
            height = dist

    c7up_est = c7p + perp * height

    # 2차: 방향성 gradient 기반 코너 검출 (좁은 반경으로 오탐 억제)
    if gray_img is not None:
        search_r = max(5, int(width * 0.20))
        c7up_refined = _refine_c7up_corner(gray_img, c7up_est, search_r)
        if c7up_refined is not None:
            return c7up_refined

    return (float(c7up_est[0]), float(c7up_est[1]))


def refine_corner(gray_img, estimated_pt, search_radius=30):
    """
    추정 위치 주변에서 실제 코너를 찾아 반환. (C2 등 범용)
    """
    h, w = gray_img.shape[:2]
    cx, cy = int(round(estimated_pt[0])), int(round(estimated_pt[1]))

    # ROI 영역 설정
    x1 = max(0, cx - search_radius)
    y1 = max(0, cy - search_radius)
    x2 = min(w, cx + search_radius)
    y2 = min(h, cy + search_radius)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    roi = gray_img[y1:y2, x1:x2]

    # Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=10,
        qualityLevel=0.1,
        minDistance=5,
        blockSize=7
    )

    if corners is None or len(corners) == 0:
        return None

    # 추정 위치와 가장 가까운 코너 선택
    best_corner = None
    best_dist = float('inf')
    est_local = np.array([cx - x1, cy - y1])

    for corner in corners:
        pt = corner[0]
        dist = np.linalg.norm(pt - est_local)
        if dist < best_dist:
            best_dist = dist
            best_corner = pt

    if best_corner is not None:
        # 전역 좌표로 변환
        return (float(best_corner[0] + x1), float(best_corner[1] + y1))

    return None


def calculate_measurements(keypoints, mm_per_pixel=None):
    c2_slope = calculate_slope(keypoints["C2A"], keypoints["C2P"])
    c7_slope = calculate_slope(keypoints["C7A"], keypoints["C7P"])
    # Cobb angle = C7 slope - C2 slope (부호 포함)
    cobb_angle = round(c7_slope - c2_slope, 1)
    sva_px, sva_mm = calculate_sva(keypoints, mm_per_pixel)
    return {
        "cobb_angle": cobb_angle,
        "c2_slope": c2_slope,
        "c7_slope": c7_slope,
        "sva_px": sva_px,
        "sva_mm": sva_mm,
    }


class C2C7Inference:
    def __init__(self, session=None):
        if session is not None:
            self.session = session
        else:
            model_path = get_model_path()
            if model_path is None:
                raise FileNotFoundError("Model file not found. Place c2c7.onnx in the app directory.")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess_image(self, gray_img):
        """Preprocess a grayscale image for inference."""
        original_h, original_w = gray_img.shape[:2]
        resized, scale = resize_height(gray_img, TARGET_HEIGHT)
        resized_h, resized_w = resized.shape[:2]
        padded, (pad_h, pad_w) = pad_to_multiple(resized, PAD_MULTIPLE)
        clahe_img = apply_clahe(padded)
        clahe_img = enhance_c7_region(clahe_img, padded)
        tensor = clahe_img.astype(np.float32) / 255.0
        tensor = tensor[np.newaxis, np.newaxis, :, :]
        return tensor, scale, resized_h, resized_w

    def _run_inference_on_image(self, color_img, pixel_spacing=None, needs_invert=False):
        """Run inference on a single color image (multi-spine sub-image용).
        needs_invert: run_multi()에서 전체 이미지 기준으로 판단한 역상 여부.
        sub-image 경계는 몸통으로 채워져 있어 극성 재감지가 부정확하므로
        전체 이미지의 판단 결과를 그대로 사용하고, threshold/hflip만 재시도.

        최적화: (invert, hflip) 조합당 ONNX 추론 1회, threshold만 재시도.
        """
        gray_orig = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        orig_w = gray_orig.shape[1]

        # threshold × 좌우반전 조합으로 최대 4회 시도 (극성은 호출자가 결정)
        # (None, False)와 (64, False)는 동일 gray → 동일 ONNX 출력 캐싱
        sub_attempts = [
            (None, False),  # 1차: thr=127
            (64,   False),  # 2차: thr=64
            (None, True),   # 3차: thr=127, hflip
            (64,   True),   # 4차: thr=64, hflip
        ]
        keypoints = None
        used_hflip = False
        used_scale = 1.0
        keypoints_fallback = None
        hflip_fallback = False
        scale_fallback = 1.0

        # ONNX 출력 캐시: do_flip → (raw_outputs, scale, resized_h, resized_w)
        output_cache = {}

        for thr, do_flip in sub_attempts:
            if do_flip not in output_cache:
                gray = gray_orig.copy()
                if needs_invert:
                    gray = 255 - gray
                if do_flip:
                    gray = cv2.flip(gray, 1)
                resized, scale = resize_height(gray, TARGET_HEIGHT)
                resized_h, resized_w = resized.shape[:2]
                padded, _ = pad_to_multiple(resized, PAD_MULTIPLE)
                clahe_img = apply_clahe(padded)
                clahe_img = enhance_c7_region(clahe_img, padded)
                tensor = clahe_img.astype(np.float32) / 255.0
                tensor = tensor[np.newaxis, np.newaxis, :, :]
                raw_outputs = self.session.run(None, {self.input_name: tensor})
                output_cache[do_flip] = (raw_outputs, scale, resized_h, resized_w)

            raw_outputs, scale, resized_h, resized_w = output_cache[do_flip]
            output = raw_outputs[0][0]
            heatmaps = output[:4, :resized_h, :resized_w]
            affinity = output[4:6, :resized_h, :resized_w]
            heatmaps_uint8 = np.clip(heatmaps * 255, 0, 255).astype(np.uint8)
            keypoints = heatmap2points(heatmaps_uint8, affinity, threshold=thr)

            if keypoints is not None:
                if not do_flip and should_flip_horizontal(keypoints):
                    if keypoints_fallback is None:
                        keypoints_fallback = keypoints
                        hflip_fallback = False
                        scale_fallback = scale
                    keypoints = None
                    continue
                used_hflip = do_flip
                used_scale = scale
                break

        if keypoints is None and keypoints_fallback is not None:
            keypoints = keypoints_fallback
            used_hflip = hflip_fallback
            used_scale = scale_fallback

        if keypoints is None:
            raise RuntimeError("Could not detect all 4 keypoints (C2A, C2P, C7A, C7P)")

        keypoints_original = {}
        for name, (x, y) in keypoints.items():
            if used_hflip:
                x_orig = (orig_w - 1) - (x / used_scale)
            else:
                x_orig = x / used_scale
            keypoints_original[name] = (x_orig, y / used_scale)

        # C7UP estimation (원본 이미지 좌표 기준)
        keypoints_original["C7UP"] = estimate_c7up(keypoints_original, gray_orig)

        # mm/pixel ratio
        mm_per_pixel = pixel_spacing
        if mm_per_pixel is None:
            mm_per_pixel = detect_scale_bar(gray_orig)

        measurements = calculate_measurements(keypoints_original, mm_per_pixel)
        return measurements, keypoints_original

    def _run_from_gray(self, gray_base, original_color, pixel_spacing, dicom_info):
        """로드된 gray 이미지로부터 추론 실행 (파일 재로드/OCR 없음).

        최적화: (force_inv, force_hflip) 조합당 ONNX 추론 1회 수행 후 캐싱.
        동일 gray에 threshold만 다른 시도는 캐싱된 출력을 재사용하므로
        최대 ONNX 호출 횟수: 4회 (이전 8회에서 절감).
        """
        orig_w = gray_base.shape[1]

        # 극성(정/역) × 좌우반전 조합별로 threshold를 달리해 최대 8회 시도.
        # 같은 (force_inv, force_hflip) 조합은 동일 gray → ONNX 출력 캐싱.
        attempts = [
            (False, None,  False),  # 1차: 자동극성, thr=127
            (True,  None,  False),  # 2차: 반대극성, thr=127
            (False, 64,    False),  # 3차: 자동극성, thr=64  (1차와 동일 gray → 캐시)
            (True,  64,    False),  # 4차: 반대극성, thr=64  (2차와 동일 gray → 캐시)
            (False, None,  True),   # 5차: 좌우반전, thr=127
            (True,  None,  True),   # 6차: 좌우반전+역상, thr=127
            (False, 64,    True),   # 7차: 좌우반전, thr=64  (5차와 동일 gray → 캐시)
            (True,  64,    True),   # 8차: 좌우반전+역상, thr=64 (6차와 동일 gray → 캐시)
        ]
        keypoints = None
        hflip_applied = False
        scale_used = 1.0
        keypoints_fallback = None
        hflip_fallback = False
        scale_fallback = 1.0

        # ONNX 출력 캐시: (force_inv, force_hflip) → (raw_outputs, scale, resized_h, resized_w)
        output_cache = {}

        for force_inv, thr, force_hflip in attempts:
            cache_key = (force_inv, force_hflip)
            if cache_key not in output_cache:
                gray = gray_base.copy()
                if force_inv:
                    gray = 255 - gray
                if force_hflip:
                    gray = cv2.flip(gray, 1)
                resized, scale = resize_height(gray, TARGET_HEIGHT)
                resized_h, resized_w = resized.shape[:2]
                padded, _ = pad_to_multiple(resized, PAD_MULTIPLE)
                clahe_img = apply_clahe(padded)
                clahe_img = enhance_c7_region(clahe_img, padded)
                tensor = clahe_img.astype(np.float32) / 255.0
                tensor = tensor[np.newaxis, np.newaxis, :, :]
                raw_outputs = self.session.run(None, {self.input_name: tensor})
                output_cache[cache_key] = (raw_outputs, scale, resized_h, resized_w)

            raw_outputs, scale, resized_h, resized_w = output_cache[cache_key]
            output = raw_outputs[0][0]
            heatmaps = output[:4, :resized_h, :resized_w]
            affinity = output[4:6, :resized_h, :resized_w]
            heatmaps_uint8 = np.clip(heatmaps * 255, 0, 255).astype(np.uint8)
            keypoints = heatmap2points(heatmaps_uint8, affinity, threshold=thr)

            if keypoints is not None:
                if not force_hflip and should_flip_horizontal(keypoints):
                    # 방향 불일치 → hflip 시도를 우선하되, 결과를 fallback으로 보존
                    if keypoints_fallback is None:
                        keypoints_fallback = keypoints
                        hflip_fallback = False
                        scale_fallback = scale
                    keypoints = None
                    continue
                hflip_applied = force_hflip
                scale_used = scale
                break

        # hflip 시도가 모두 실패하면 방향 불일치 결과를 fallback으로 사용
        if keypoints is None and keypoints_fallback is not None:
            keypoints = keypoints_fallback
            hflip_applied = hflip_fallback
            scale_used = scale_fallback

        if keypoints is None:
            raise RuntimeError("Could not detect all 4 keypoints (C2A, C2P, C7A, C7P)")

        keypoints_original = {}
        for name, (x, y) in keypoints.items():
            if hflip_applied:
                x_orig = (orig_w - 1) - (x / scale_used)
            else:
                x_orig = x / scale_used
            keypoints_original[name] = (x_orig, y / scale_used)

        # C7UP 추정 (원본 이미지 기준)
        gray_original = cv2.cvtColor(original_color, cv2.COLOR_BGR2GRAY)
        keypoints_original["C7UP"] = estimate_c7up(keypoints_original, gray_original)
        # mm/pixel 비율: DICOM이면 PixelSpacing, 아니면 스케일 바 감지
        mm_per_pixel = pixel_spacing
        if mm_per_pixel is None:
            mm_per_pixel = detect_scale_bar(gray_original)
        measurements = calculate_measurements(keypoints_original, mm_per_pixel)
        return measurements, keypoints_original, original_color, dicom_info

    def run(self, filepath):
        """Run inference on a single image file."""
        raw, pixel_spacing, dicom_info = load_image(filepath)
        gray_base = to_grayscale_uint8(raw)
        gray_base, _crop = auto_crop_screen_photo(gray_base)
        gray_base = correct_inversion(gray_base)

        if raw.ndim == 2:
            original_color = cv2.cvtColor(to_grayscale_uint8(raw), cv2.COLOR_GRAY2BGR)
            if _crop is not None:
                cx1, cy1, cx2, cy2 = _crop
                original_color = original_color[cy1:cy2, cx1:cx2]
        elif raw.ndim == 3 and raw.shape[2] == 3 and raw.dtype == np.uint8:
            if _crop is not None:
                cx1, cy1, cx2, cy2 = _crop
                original_color = raw[cy1:cy2, cx1:cx2].copy()
            else:
                original_color = raw.copy()
        else:
            original_color = cv2.cvtColor(to_grayscale_uint8(raw), cv2.COLOR_GRAY2BGR)
            if _crop is not None:
                cx1, cy1, cx2, cy2 = _crop
                original_color = original_color[cy1:cy2, cx1:cx2]

        return self._run_from_gray(gray_base, original_color, pixel_spacing, dicom_info)

    def run_multi(self, filepath):
        """
        Run inference with automatic multi-spine detection.
        If multiple spines are detected in one image, analyzes each separately.

        Returns: Tuple of (results_list, original_color_image, regions)

        results_list: List of tuples, each containing:
            (measurements, keypoints, sub_image, dicom_info, region_index)
            or for errors:
            (None, None, sub_image, dicom_info, region_index, error_msg)

        regions: List of (x_start, x_end) tuples for each region

        region_index: 0 for single/first spine, 1 for second, etc.
        For single spine images, returns a list with one element.
        """
        # Load image and get metadata (1회만 수행)
        raw, pixel_spacing, dicom_info = load_image(filepath)
        gray_orig = to_grayscale_uint8(raw)

        # Remove dark borders from phone photos of PACS screens
        gray_orig, _crop = auto_crop_screen_photo(gray_orig)

        # 역상 여부를 전체 이미지 기준으로 한 번만 판단
        needs_invert = should_invert(gray_orig)

        # 표시용 original_color 및 영역 검출은 원본 기준으로 유지
        if raw.ndim == 2:
            original_color = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2BGR)
        elif raw.ndim == 3 and raw.shape[2] == 3 and raw.dtype == np.uint8:
            if _crop is not None:
                cx1, cy1, cx2, cy2 = _crop
                original_color = raw[cy1:cy2, cx1:cx2].copy()
            else:
                original_color = raw.copy()
        else:
            original_color = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2BGR)

        # Detect spine regions on original gray (dark dividers must remain dark)
        regions = detect_spine_regions(gray_orig)

        results = []

        if len(regions) == 1:
            # 단일 척추: 이미 로드된 데이터 재사용 (파일 재로드/OCR 없음)
            try:
                gray_base = correct_inversion(gray_orig)
                measurements, keypoints, _, _ = self._run_from_gray(
                    gray_base, original_color, pixel_spacing, dicom_info
                )
                results.append((measurements, keypoints, original_color, dicom_info, 0))
            except Exception as e:
                raise RuntimeError(f"Analysis failed: {str(e)}")
        else:
            # Multiple spines - process each region
            sub_images = split_image_for_regions(original_color, regions)

            for idx, (sub_img, x_offset) in enumerate(sub_images):
                try:
                    measurements, keypoints = self._run_inference_on_image(sub_img, pixel_spacing, needs_invert=needs_invert)
                    # Keypoints are in sub_image coordinates (no offset needed for drawing on sub_image)
                    results.append((measurements, keypoints, sub_img, dicom_info, idx))
                except Exception as e:
                    # If one region fails, continue with others
                    results.append((None, None, sub_img, dicom_info, idx, str(e)))

        return results, original_color, regions
