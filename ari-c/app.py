import io
import os
import zipfile
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from openpyxl import Workbook
from openpyxl.styles import Font as XLFont, Alignment, PatternFill, Border, Side
from PIL import Image

from inference import C2C7Inference
from cobb_draw import draw_cobb_angle

# File size limit (10MB)
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

st.set_page_config(page_title="ARI-C: C-lat Analyzer", layout="wide")
st.title("ARI-C: C-lat Analyzer")
st.caption("C2-C7 Cervical Spine Cobb Angle Measurement")

# Info box about file size limit
st.info(f"**Web version file size limit: {MAX_FILE_SIZE_MB}MB per upload**")


@st.cache_resource
def load_engine():
    return C2C7Inference()


def pil_to_cv2(pil_img):
    img = np.array(pil_img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv_img):
    if cv_img.ndim == 2:
        return Image.fromarray(cv_img)
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


uploaded_files = st.file_uploader(
    "Upload image files",
    type=["jpg", "jpeg", "png", "bmp", "dcm"],
    accept_multiple_files=True,
)

if uploaded_files:
    # Check total file size
    total_size = sum(f.size for f in uploaded_files)
    oversized_files = [f.name for f in uploaded_files if f.size > MAX_FILE_SIZE_BYTES]

    if total_size > MAX_FILE_SIZE_BYTES:
        st.error(f"""
        **File size limit exceeded!**

        Total uploaded size: **{total_size / (1024*1024):.1f}MB** (Limit: {MAX_FILE_SIZE_MB}MB)
        """)

        if oversized_files:
            st.warning(f"Oversized files: {', '.join(oversized_files)}")

        st.markdown("""
        ---
        ### Need to analyze larger files?

        **Get the Desktop version for unlimited file processing!**

        - No file size limits
        - Batch processing for thousands of files
        - Faster analysis with parallel processing
        - Works offline

        **Contact us to purchase:** [ari-c@example.com](mailto:ari-c@example.com)

        ---
        """)
        st.stop()

    if st.button("Analyze", type="primary"):
        engine = load_engine()
        all_results = []
        result_images = {}

        progress_bar = st.progress(0, text="Processing...")
        total = len(uploaded_files)

        for idx, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            progress_bar.progress((idx) / total, text=f"Processing: {filename} ({idx+1}/{total})")

            try:
                # Save uploaded file to temp path for inference
                suffix = os.path.splitext(filename)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                # Use run_multi for automatic multi-spine detection
                results_list, original_color, regions = engine.run_multi(tmp_path)

                os.unlink(tmp_path)

                num_regions = len(regions)

                if num_regions == 1:
                    # Single spine - simple case
                    result_tuple = results_list[0]
                    if len(result_tuple) == 6:
                        # Error
                        measurements, keypoints, sub_img, dicom_info, region_idx, error_msg = result_tuple
                        all_results.append({
                            "Filename": filename,
                            "C2-C7 Angle (°)": "",
                            "C2 Slope (°)": "",
                            "C7 Slope (°)": "",
                            "C2-C7 SVA (mm)": "",
                            "C2-C7 SVA (px)": "",
                            "Sex": "",
                            "Age": "",
                            "Study Date": "",
                            "Study ID": "",
                            "Study Desc": "",
                            "Status": f"Error: {error_msg[:60]}",
                        })
                    else:
                        measurements, keypoints, sub_img, dicom_info, region_idx = result_tuple
                        result_img = draw_cobb_angle(original_color.copy(), keypoints, measurements)
                        sva_mm = measurements.get("sva_mm") if measurements.get("sva_mm") is not None else ""
                        dicom_info = dicom_info or {}
                        all_results.append({
                            "Filename": filename,
                            "C2-C7 Angle (°)": measurements["cobb_angle"],
                            "C2 Slope (°)": measurements["c2_slope"],
                            "C7 Slope (°)": measurements["c7_slope"],
                            "C2-C7 SVA (mm)": sva_mm,
                            "C2-C7 SVA (px)": measurements["sva_px"],
                            "Sex": dicom_info.get("sex", ""),
                            "Age": dicom_info.get("age", ""),
                            "Study Date": dicom_info.get("study_date", ""),
                            "Study ID": dicom_info.get("study_id", ""),
                            "Study Desc": dicom_info.get("study_description", ""),
                            "Status": "Completed",
                        })
                        result_images[filename] = {
                            "original": original_color,
                            "result": result_img,
                        }
                else:
                    # Multiple spines - draw on each sub-image and combine
                    combined_result = original_color.copy()

                    for result_tuple in results_list:
                        if len(result_tuple) == 6:
                            # Error
                            measurements, keypoints, sub_img, dicom_info, region_idx, error_msg = result_tuple
                            region_label = f" (Region {region_idx + 1})"
                            all_results.append({
                                "Filename": filename + region_label,
                                "C2-C7 Angle (°)": "",
                                "C2 Slope (°)": "",
                                "C7 Slope (°)": "",
                                "C2-C7 SVA (mm)": "",
                                "C2-C7 SVA (px)": "",
                                "Sex": "",
                                "Age": "",
                                "Study Date": "",
                                "Study ID": "",
                                "Study Desc": "",
                                "Status": f"Error: {error_msg[:60]}",
                            })
                            continue

                        measurements, keypoints, sub_img, dicom_info, region_idx = result_tuple

                        # Draw on sub-image (keypoints are in sub-image coordinates)
                        sub_result = draw_cobb_angle(sub_img.copy(), keypoints, measurements)

                        # Place sub-result back into combined image
                        x_start, x_end = regions[region_idx]
                        combined_result[:, x_start:x_end] = sub_result

                        sva_mm = measurements.get("sva_mm") if measurements.get("sva_mm") is not None else ""
                        dicom_info = dicom_info or {}
                        region_label = f" (Region {region_idx + 1})"
                        display_name = filename + region_label

                        all_results.append({
                            "Filename": display_name,
                            "C2-C7 Angle (°)": measurements["cobb_angle"],
                            "C2 Slope (°)": measurements["c2_slope"],
                            "C7 Slope (°)": measurements["c7_slope"],
                            "C2-C7 SVA (mm)": sva_mm,
                            "C2-C7 SVA (px)": measurements["sva_px"],
                            "Sex": dicom_info.get("sex", ""),
                            "Age": dicom_info.get("age", ""),
                            "Study Date": dicom_info.get("study_date", ""),
                            "Study ID": dicom_info.get("study_id", ""),
                            "Study Desc": dicom_info.get("study_description", ""),
                            "Status": "Completed",
                        })

                    # Store combined result image
                    result_images[filename] = {
                        "original": original_color,
                        "result": combined_result,
                    }

            except Exception as e:
                all_results.append({
                    "Filename": filename,
                    "C2-C7 Angle (°)": "",
                    "C2 Slope (°)": "",
                    "C7 Slope (°)": "",
                    "C2-C7 SVA (mm)": "",
                    "C2-C7 SVA (px)": "",
                    "Sex": "",
                    "Age": "",
                    "Study Date": "",
                    "Study ID": "",
                    "Study Desc": "",
                    "Status": f"Error: {str(e)[:80]}",
                })
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        progress_bar.progress(1.0, text="Processing complete!")

        # Store results in session state
        st.session_state["all_results"] = all_results
        st.session_state["result_images"] = result_images

    # Display results from session state
    if "all_results" in st.session_state:
        all_results = st.session_state["all_results"]
        result_images = st.session_state["result_images"]

        # Measurements table
        st.subheader("Measurement Results")
        df = pd.DataFrame(all_results)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Image display
        st.subheader("Result Images")
        for filename, data in result_images.items():
            st.markdown(f"**{filename}**")
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2_to_pil(data["original"]), caption="Original", use_container_width=True)
            with col2:
                st.image(cv2_to_pil(data["result"]), caption="Result", use_container_width=True)

        # Download buttons
        st.subheader("Downloads")
        col_dl1, col_dl2 = st.columns(2)

        # Excel download
        with col_dl1:
            wb = Workbook()
            ws = wb.active
            ws.title = "C2-C7 Measurements"

            # Header styling
            header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            header_font_white = XLFont(bold=True, size=11, color="FFFFFF")
            thin_border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            center_align = Alignment(horizontal='center', vertical='center')

            headers = ["Filename", "C2-C7 Angle (°)", "C2 Slope (°)",
                       "C7 Slope (°)", "C2-C7 SVA (mm)", "C2-C7 SVA (px)",
                       "Sex", "Age", "Study Date", "Study ID", "Study Desc", "Status"]
            ws.append(headers)

            for col, header in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col)
                cell.font = header_font_white
                cell.fill = header_fill
                cell.alignment = center_align
                cell.border = thin_border

            for row in all_results:
                ws.append([
                    row["Filename"],
                    row["C2-C7 Angle (°)"],
                    row["C2 Slope (°)"],
                    row["C7 Slope (°)"],
                    row["C2-C7 SVA (mm)"],
                    row["C2-C7 SVA (px)"],
                    row.get("Sex", ""),
                    row.get("Age", ""),
                    row.get("Study Date", ""),
                    row.get("Study ID", ""),
                    row.get("Study Desc", ""),
                    row["Status"],
                ])

            for row_idx in range(2, ws.max_row + 1):
                for col in range(1, 13):
                    cell = ws.cell(row=row_idx, column=col)
                    cell.border = thin_border
                    if col > 1:
                        cell.alignment = center_align

            ws.column_dimensions['A'].width = 30
            for col in ['B', 'C', 'D', 'E', 'F']:
                ws.column_dimensions[col].width = 16
            ws.column_dimensions['G'].width = 8   # Sex
            ws.column_dimensions['H'].width = 8   # Age
            ws.column_dimensions['I'].width = 12  # Study Date
            ws.column_dimensions['J'].width = 12  # Study ID
            ws.column_dimensions['K'].width = 20  # Study Desc
            ws.column_dimensions['L'].width = 20  # Status

            excel_buf = io.BytesIO()
            wb.save(excel_buf)
            excel_buf.seek(0)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            st.download_button(
                label="Download Excel",
                data=excel_buf,
                file_name=f"C-lat_angle_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        # ZIP download (result images)
        with col_dl2:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for filename, data in result_images.items():
                    basename = os.path.splitext(filename)[0]
                    _, png_bytes = cv2.imencode(".png", data["result"])
                    zf.writestr(f"result_{basename}.png", png_bytes.tobytes())
            zip_buf.seek(0)
            st.download_button(
                label="Download Result Images (ZIP)",
                data=zip_buf,
                file_name="c2c7_results.zip",
                mime="application/zip",
            )
