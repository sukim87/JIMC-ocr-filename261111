import gc
import hashlib
import io
import os
import re
import sys

import easyocr
import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

VERSION = "V.261111"
OCR_PATCH = "숫자 인식 전처리 적용"
UPDATE_DATE = "2026-11-11"
MAX_FILES = 5


def get_resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


@st.cache_resource(show_spinner=False)
def load_ocr_reader():
    return easyocr.Reader(["ko", "en"], gpu=False)


def file_sha256(data):
    return hashlib.sha256(data).hexdigest()


def resize_for_ocr(img, max_width=1800):
    img = img.convert("RGB")
    width, height = img.size
    if width <= max_width:
        return img
    ratio = max_width / float(width)
    return img.resize((int(width * ratio), int(height * ratio)), Image.LANCZOS)


def crop_required_area(img):
    # 필요한 필드가 있는 상단 영역만 OCR한다.
    width, height = img.size
    bottom = max(1, int(height * 0.67))
    return img.crop((0, 0, width, bottom))


def clean_and_fix_order_no(order_str):
    if not order_str:
        return ""
    raw = re.sub(r"\s+", "", order_str.upper().strip())
    if raw.startswith(("PO", "P0", "MI")) or any(x in raw for x in ("MATERIAL", "MATER1AL", "MATL")):
        return ""
    if raw.startswith("ZNAJOB"):
        return raw

    match = re.search(r"([MHXP][A-Z0-9\-_]+)", raw, re.I)
    if not match:
        return ""
    value = match.group(1).rstrip("-_")

    # 코드 전체를 숫자로 강제 변환하지 않는다. 첫 문자와 끝 영문(예: X240305HD)은 보존한다.
    # 숫자가 위치하는 중간 구간에서만 OCR 혼동문자를 보정한다.
    chars = list(value)
    digit_map = {"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"}
    for i in range(1, len(chars)):
        if chars[i].isalpha():
            prev_is_digitlike = i > 1 and (chars[i-1].isdigit() or chars[i-1] in digit_map)
            next_is_digitlike = i+1 < len(chars) and (chars[i+1].isdigit() or chars[i+1] in digit_map)
            if prev_is_digitlike and next_is_digitlike and chars[i] in digit_map:
                chars[i] = digit_map[chars[i]]
    return "".join(chars)


def normalize_ocr_results(results):
    rows = []
    texts = []
    for bbox, text, prob in results:
        value = str(text).strip()
        if not value:
            continue
        texts.append(value)
        rows.append({
            "text": value,
            "top": float(bbox[0][1]),
            "left": float(bbox[0][0]),
            "prob": float(prob),
        })
    return " ".join(texts), texts, pd.DataFrame(rows)


def extract_fields(results):
    full_text, full_text_list, df = normalize_ocr_results(results)

    order_no = ""
    found = re.search(r"\b(ZNAJOB[A-Z0-9]*)\b", full_text, re.I)
    if found:
        order_no = found.group(1)
    else:
        blacklist = {"ORDER", "URDER", "NUMBER", "DELIVER", "CUSTOMER", "VENDOR",
                     "INSPECTION", "REPORT", "NOTIFICATION", "ORDERNO", "URDERNO",
                     "MATERIAL", "MATER1AL", "MATL"}
        for candidate in re.findall(r"\b([MHXP][234][A-Z0-9\-_]+)\b", full_text, re.I):
            upper = candidate.upper()
            if upper not in blacklist and not upper.startswith(("MATER", "MI", "PO", "P0")):
                order_no = candidate
                break
    order_no = clean_and_fix_order_no(re.sub(r"[\-_]$", "", order_no))

    date = ""
    for candidate in re.findall(r"(20[2-9][0-9][-/\.][0-9]{2}[-/\.][0-9]{2})", full_text):
        digits = re.sub(r"[^0-9]", "", candidate)
        if len(digits) == 8:
            date = digits
            break
    if not date:
        for text in full_text_list:
            digits = re.sub(r"[^0-9]", "", text)
            if len(digits) == 8 and digits.startswith("20"):
                date = digits
                break

    po_no = ""
    po_match = re.search(r"(PO?[0-9]{8,})", full_text, re.I)
    if po_match:
        po_no = po_match.group(1)

    vendor = ""
    customer_words = set()
    if not df.empty:
        customer_labels = df[df["text"].astype(str).str.contains("고객|Customer", na=False, case=False)]
        if not customer_labels.empty:
            row = customer_labels.iloc[0]
            targets = df[(df["top"] >= row["top"] - 20) & (df["top"] <= row["top"] + 30) &
                         (df["left"] > row["left"])].sort_values("left")
            for _, target in targets.iterrows():
                value = re.sub(r"[^가-힣a-zA-Z0-9]", "", str(target["text"]))
                if value and value.lower() not in ("고객", "customer"):
                    customer_words.add(value)

        vendor_labels = df[df["text"].astype(str).str.contains("업체소재지|소재지|Vendor", na=False, case=False)]
        if not vendor_labels.empty:
            row = vendor_labels.iloc[0]
            targets = df[(df["left"] > row["left"] + 5) &
                         (df["top"] >= row["top"] - 40) &
                         (df["top"] <= row["top"] + 50)].sort_values("left")
            for _, target in targets.iterrows():
                text = str(target["text"])
                if any(x in text for x in ("업체소재지", "Vendor", "Address", "결재", "성산구", "의창구", "강서구", "녹산산업")):
                    continue
                value = re.split(r"[\(\[\d]|경상남도|창원시|의창구|부산|강서구|녹산|경남|서울|경기|시|구|군", text)[0].strip()
                value = re.sub(r"[^가-힣a-zA-Z0-9]", "", value)
                if len(value) >= 2 and value not in customer_words:
                    vendor = value
                    break

    if not vendor:
        company_terms = ("스틸", "볼텍", "머티리얼", "에스앤피", "주식회사", "(주)", "공업", "금속",
                         "테크", "산업", "엔지니어링", "상사", "정밀", "기업", "파이프", "금동", "스틱")
        for text in full_text_list:
            value = re.split(r"[\(\[\d]|경상남도|창원시|의창구|부산|강서구|녹산|경남|서울|경기|시|구|군", text)[0].strip()
            value = re.sub(r"[^가-힣a-zA-Z0-9]", "", value)
            if value in customer_words or value.lower() in ("업체소재지", "vendor", "address", "고객", "customer"):
                continue
            if len(value) >= 2 and any(term in text for term in company_terms):
                vendor = value
                break
    vendor = vendor.replace("스틱", "스틸") if vendor else ""

    return {"order_no": order_no, "date": date, "vendor": vendor, "po_no": po_no}


def success_score(fields):
    return sum(bool(fields.get(key)) for key in ("order_no", "date", "po_no"))


def preprocess_number_image(img):
    """숫자/영문 코드 인식용 전처리: 2배 확대 + 회색조 + 대비/선명도 강화 + 자동 대비."""
    gray = ImageOps.grayscale(img)
    gray = gray.resize((max(1, gray.width * 2), max(1, gray.height * 2)), Image.LANCZOS)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.2, percent=170, threshold=2))
    return gray


def numeric_candidate_results(target, reader):
    """원본 OCR이 핵심 숫자 필드를 놓쳤을 때만 전처리 OCR을 한 번 추가한다."""
    enhanced = preprocess_number_image(target)
    arr = np.asarray(enhanced, dtype=np.uint8)
    try:
        return reader.readtext(arr, detail=1, paragraph=False)
    finally:
        del arr, enhanced


def merge_fields(primary, secondary):
    merged = dict(primary)
    for key in ("order_no", "date", "po_no"):
        if not merged.get(key) and secondary.get(key):
            merged[key] = secondary[key]
    # 업체명은 기존 원본 OCR 결과를 우선하여 전처리 때문에 한글 인식 규칙이 흔들리지 않게 한다.
    if not merged.get("vendor") and secondary.get("vendor"):
        merged["vendor"] = secondary["vendor"]
    return merged


def run_ocr_once(img, reader, angle):
    rotated = img.rotate(angle, expand=True) if angle else img
    target = crop_required_area(rotated)
    array = np.asarray(target, dtype=np.uint8)
    results = reader.readtext(array)
    fields = extract_fields(results)

    # 핵심 숫자 필드가 부족한 경우에만 숫자 인식용 전처리를 추가 적용한다.
    # 정상 문서는 기존 1회 OCR로 끝나므로 속도 저하를 최소화한다.
    if success_score(fields) < 3:
        enhanced_results = numeric_candidate_results(target, reader)
        enhanced_fields = extract_fields(enhanced_results)
        fields = merge_fields(fields, enhanced_fields)
        del enhanced_results, enhanced_fields

    del array, target, rotated, results
    return fields


def analyze_image(img, reader):
    # 정상 문서는 첫 OCR 한 번으로 종료한다. 핵심 필드 인식 실패 때만 회전 방향을 재시도한다.
    prepared = resize_for_ocr(img, max_width=1800)
    best = run_ocr_once(prepared, reader, 0)
    if success_score(best) >= 2:
        del prepared
        return best

    for angle in (180, 90, 270):
        candidate = run_ocr_once(prepared, reader, angle)
        if success_score(candidate) > success_score(best):
            best = candidate
        if success_score(best) >= 2:
            break
    del prepared
    return best


def render_first_page(file_bytes):
    pdf = pdfium.PdfDocument(file_bytes)
    try:
        page = pdf[0]
        image = page.render(scale=1.6).to_pil().convert("RGB")
        page.close()
        return image
    finally:
        pdf.close()


def analyze_file(file_bytes, file_ext, reader):
    if file_ext == "pdf":
        image = render_first_page(file_bytes)
    else:
        with Image.open(io.BytesIO(file_bytes)) as opened:
            image = opened.convert("RGB")
    try:
        return analyze_image(image, reader)
    finally:
        image.close()
        gc.collect()


def mime_for(ext):
    return {"pdf": "application/pdf", "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "application/octet-stream")


def init_state():
    st.session_state.setdefault("ocr_cache", {})
    st.session_state.setdefault("processed_results", [])
    st.session_state.setdefault("last_upload_signature", "")


def clear_results():
    st.session_state.processed_results = []
    st.session_state.last_upload_signature = ""


def main():
    st.set_page_config(page_title="인수검사서 파일명 자동 생성기 | (주)정우산기", page_icon="📄", layout="wide")
    init_state()

    with st.sidebar:
        logo_path = get_resource_path("logo.jpg")
        if not os.path.exists(logo_path):
            logo_path = get_resource_path("세로-영문-Jeongwoo.jpg")
        if os.path.exists(logo_path):
            st.image(logo_path)
        st.markdown("---")
        st.markdown("### 📞 시스템 문의 및 지원")
        st.info("**시스템 개발 및 관리자**  \n* **부서:** PS품질팀  \n* **담당자:** 김선웅  \n* **문의 내용:** 프로그램 오류, 수주번호/업체명 추출 규칙 변경 요청")
        st.markdown("---")
        st.caption("ⓒ Jeongwoo Industrial Machine Co., Ltd. All rights reserved.")

    st.title(f"📄 인수검사서 파일명 자동 생성기 `{VERSION}`")
    st.caption(f"📅 최종 업데이트: {UPDATE_DATE} | (주)정우산기")
    st.info("💡 **사용 안내**  \n인수검사 완료 파일을 첨부한 후 **분석 시작**을 누르면 파일명을 자동 생성합니다.  \n📌 **파일명 생성 기준:** `수주번호_날짜_업체명_발주서번호`")
    st.markdown("---")

    uploaded_files = st.file_uploader("PDF 또는 이미지 파일을 선택하세요 (최대 5개)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
    target_files = list(uploaded_files[:MAX_FILES]) if uploaded_files else []
    if uploaded_files and len(uploaded_files) > MAX_FILES:
        st.warning("최대 5개까지 처리합니다. 상위 5개 파일만 선택되었습니다.")

    upload_signature = "|".join(f"{f.name}:{f.size}" for f in target_files)
    if upload_signature != st.session_state.last_upload_signature and st.session_state.processed_results:
        clear_results()

    col1, col2 = st.columns([1, 1])
    analyze_clicked = col1.button("🔍 분석 시작", type="primary", use_container_width=True, disabled=not target_files)
    clear_clicked = col2.button("🧹 결과 지우기", use_container_width=True, disabled=not st.session_state.processed_results)
    if clear_clicked:
        clear_results()
        st.rerun()

    if analyze_clicked:
        # EasyOCR는 앱 접속 시 로딩하지 않고 분석 버튼을 눌렀을 때만 초기화한다.
        # 따라서 모델 다운로드/초기화가 느려도 앱 UI 자체는 먼저 정상 표시된다.
        with st.spinner("OCR AI 모델을 준비 중입니다. 최초 실행은 모델 다운로드로 시간이 걸릴 수 있습니다..."):
            reader = load_ocr_reader()

        results = []
        progress = st.progress(0.0)
        status = st.empty()
        for idx, file in enumerate(target_files):
            status.info(f"분석 중: {file.name} ({idx + 1}/{len(target_files)})")
            file.seek(0)
            file_bytes = file.read()
            ext = file.name.rsplit(".", 1)[-1].lower() if "." in file.name else "pdf"
            digest = file_sha256(file_bytes)
            cache_key = f"{VERSION}:{digest}"
            try:
                if cache_key in st.session_state.ocr_cache:
                    fields = dict(st.session_state.ocr_cache[cache_key])
                else:
                    fields = analyze_file(file_bytes, ext, reader)
                    st.session_state.ocr_cache[cache_key] = dict(fields)

                display = {
                    "order_no": fields.get("order_no") or "미인식",
                    "date": fields.get("date") or "미인식",
                    "vendor": fields.get("vendor") or "업체명확인필요",
                    "po_no": fields.get("po_no") or "미인식",
                }
                new_name = f"{display['order_no']}_{display['date']}_{display['vendor']}_{display['po_no']}.{ext}"
                results.append({"original_name": file.name, "new_name": new_name, **display,
                                "file_bytes": file_bytes, "ext": ext})
            except Exception as exc:
                st.error(f"'{file.name}' 처리 중 오류: {exc}")
            progress.progress((idx + 1) / len(target_files))
            gc.collect()

        st.session_state.processed_results = results
        st.session_state.last_upload_signature = upload_signature
        status.empty()
        st.success("🎉 모든 파일 분석이 완료되었습니다. 같은 파일은 다시 OCR하지 않습니다.")

    if st.session_state.processed_results:
        st.markdown("---")
        st.markdown("### 📊 파일별 상세 결과 및 다운로드")
        for idx, res in enumerate(st.session_state.processed_results):
            with st.expander(f"📁 {res['original_name']} ➔ {res['new_name']}", expanded=True):
                left, right = st.columns([3, 1])
                left.write(f"**수주번호:** `{res['order_no']}` | **의뢰일자:** `{res['date']}` | **업체명:** `{res['vendor']}` | **발주서번호:** `{res['po_no']}`")
                right.download_button("💾 변경된 파일 다운로드", data=res["file_bytes"], file_name=res["new_name"],
                                      mime=mime_for(res["ext"]), key=f"download_{idx}_{res['original_name']}",
                                      on_click="ignore", use_container_width=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        st.error(f"프로그램 구동 중 오류가 발생했습니다: {exc}")
        st.exception(exc)
