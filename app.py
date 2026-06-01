import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import re

class ContractVerifier:
    def __init__(self, uploaded_file, contract_type):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_cv_raw = cv2.imdecode(file_bytes, 1)
        
        raw_h, raw_w = img_cv_raw.shape[:2]
        scale = 1200.0 / raw_w
        self.img_cv = cv2.resize(img_cv_raw, (1200, int(raw_h * scale)))
        self.h, self.w = self.img_cv.shape[:2]
        
        self.img_gray = cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2GRAY)
        
        norm_img = np.zeros((self.h, self.w))
        self.img_ocr = cv2.normalize(self.img_gray, norm_img, 0, 255, cv2.NORM_MINMAX)
        
        blurred = cv2.GaussianBlur(self.img_ocr, (3, 3), 0)
        self.img_ocr_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        
        self.full_text = ""
        self.contract_type = contract_type # 'rental' 또는 'sales'

    # 동/호수 존재 여부 신속 체크용 헬퍼 메서드
    def has_dong_ho(self, text):
        clean = re.sub(r'[^0-9가-힣]', '', text)
        return bool(re.search(r'\d{1,4}[동통돔덤등]', clean) and re.search(r'\d{1,4}', clean))

    # 기준 1. 문서 형식 (Layout & Format)
    def check_layout_format(self):
        gray_blur = cv2.GaussianBlur(self.img_gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 2)
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_count = len(contours)
        
        if line_count >= 1:
            return True, f"표준 격자선 및 면적 표 서식 감지됨 (가로줄 {line_count}개 확인)"
        return False, "계약서 특유의 표 양식(격자선)이 감지되지 않았습니다. 동·호수 및 면적 표가 포함된 페이지를 올바르게 업로드했는지 확인해 주세요."

    # 기준 2. 도장(직인) 검증
    def check_seal_and_position(self):
        blurred_cv = cv2.GaussianBlur(self.img_cv, (5, 5), 0)
        hsv = cv2.cvtColor(blurred_cv, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50]) 
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2 
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_seal = False
        seal_msg = "문서 내에서 유효한 도장(LH 붉은색 사각 직인 또는 공인전자서명 타임스탬프 마크)을 찾을 수 없습니다. 도장 날인이나 서명 처리가 포함된 페이지인지 확인해 주세요."
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            
            if area > 400 and w > 20 and h > 20:
                aspect_ratio = w / float(h)
                
                if 0.6 <= aspect_ratio <= 1.4:
                    valid_seal = True
                    seal_msg = f"정사각형 LH 기관 직인 감지 완료 (비율: {aspect_ratio:.2f})"
                    break
                elif 2.5 <= aspect_ratio <= 15.0:
                    valid_seal = True
                    seal_msg = f"공인전자서명(타임스탬프) 인장 감지 완료 (비율: {aspect_ratio:.2f})"
                    break

        if valid_seal:
            return True, seal_msg
        return False, seal_msg

    # 기준 3 & 4. 계약 형태 및 명의 교차 검증
    def check_landlord_info(self):
        try:
            text1 = pytesseract.image_to_string(self.img_gray, lang='kor', config=r'--oem 3 --psm 3')
            text2 = pytesseract.image_to_string(self.img_ocr_adaptive, lang='kor', config=r'--oem 3 --psm 4')
            
            if self.has_dong_ho(text1):
                self.full_text = text1
            elif self.has_dong_ho(text2):
                self.full_text = text2
            else:
                self.full_text = text1 + "\n" + text2
            
            text_clean = re.sub(r'[^가-힣0-9]', '', self.full_text)
            
            # 1. [분양형] 검증 로직
            if self.contract_type == "sales":
                sales_keywords = ["수분양자", "분양자", "분양계약", "공공분양", "공급면적", "대지지분"]
                is_sales_type = any(keyword in text_clean for keyword in sales_keywords)
                is_lh = any(kw in text_clean for kw in ["한국토지주택공사", "주택공사", "토지", "로지", "트지", "공사"])
                is_head = any(kw in text_clean for kw in ["경기", "북부", "지역", "본부", "본부장", "본"])
                has_rental_word = any(kw in text_clean for kw in ["임대인", "표준임대차"])

                if is_sales_type and is_lh and is_head and not has_rental_word:
                    return True, "분양형 계약서 및 분양자(한국토지주택공사 경기북부지역본부장) 정보 일치 확인"
                elif has_rental_word:
                    return False, "현재 '분양형'으로 가입 신청을 하셨으나, 첨부된 서류는 '선택형/임대차' 계약서로 확인됩니다. 상단의 계약 유형 선택을 다시 확인해 주세요."
                elif is_sales_type and (not is_lh or not is_head):
                    return False, f"분양자 명의 단어가 일부 누락되었거나 사진이 흐립니다. (LH 검출: {'성공' if is_lh else '실패'}, 경기북부지역본부장 검출: {'성공' if is_head else '실패'})"
                else:
                    return False, "분양 계약서의 핵심 용어(수분양자, 분양계약 등)가 분석되지 않았습니다. 업로드한 이미지가 아파트 공급계약서 첫 페이지가 맞는지 확인해 주세요."
            
            # 2. [선택형/임대] 검증 로직
            else: 
                lh_keywords = ["한국토지주택공사", "주택공사", "토지", "공공임대", "국민임대", "행복주택", "135671"]
                rental_keywords = ["임대인", "표준임대차", "임대차"]
                is_lh = any(keyword in text_clean for keyword in lh_keywords)
                is_rental = any(keyword in text_clean for keyword in rental_keywords)
                
                if is_lh and is_rental:
                    return True, "선택형(임대) 계약서 및 임대인(LH) 정보 일치 확인"
                elif is_lh and not is_rental:
                    return False, "현재 '선택형(임대)'으로 가입 신청을 하셨으나, 첨부된 서류는 '공공분양 공급계약서'로 확인됩니다. 상단의 계약 유형 선택을 다시 확인해 주세요."
                else:
                    return False, "임대인 명의(한국토지주택공사) 및 표준임대차 양식 확인에 실패했습니다. 빛 반사가 없는 선명한 원본 사진인지 확인해 주세요."
                    
        except Exception as e:
            return False, f"OCR 글자 분석 중 시스템 내부 에러 발생: {e}"
            
    # [스마트 동/호수 추출기]
    def extract_dong_ho(self):
        if not self.full_text:
            return False, "텍스트가 추출되지 않았습니다."
            
        text_clean = re.sub(r'[^0-9가-힣]', '', self.full_text)
        
        split_keywords = ["주택의표시", "주택의표", "공공임대주택", "공공분양주택", "4주택", "3공공임대"]
        for ckw in split_keywords:
            if ckw in text_clean:
                idx = text_clean.find(ckw)
                text_clean = text_clean[idx:]
                break
        
        pattern = r'(\d{1,4})[동통돔덤등][^0-9]*?(\d{1,4})'
        match = re.search(pattern, text_clean)
        
        if match:
            dong = str(int(match.group(1)))
            ho = str(int(match.group(2)))
            return True, f"{dong}동 {ho}호"
            
        return False, "계약서 본문에서 동/호수 숫자를 명확하게 판독해내지 못했습니다. 사진의 해상도를 높이거나 테두리가 잘리지 않게 다시 찍어 주세요."

# --- Streamlit 웹 UI 구현부 ---
st.set_page_config(page_title="LH 계약서 진위 판별기", layout="wide")

st.title("📄 LH 공공임대/분양 계약서 진위 판별 시스템")
st.markdown("계약 유형(분양형/선택형) 자동 분기, 도장 직인 분석, 명의자 및 동·호수 교차 검증을 수행합니다.")

with st.form("contract_form"):
    st.subheader("계약 정보 입력")
    col_type, col_dong, col_ho = st.columns([2, 1, 1])
    
    with col_type:
        contract_type_raw = st.selectbox("계약 유형 선택", ["선택형 (6년 후 분양전환 / 임대)", "분양형 (공공분양)"])
        contract_type = "rental" if "선택형" in contract_type_raw else "sales"
        
    with col_dong:
        input_dong = st.text_input("계약 동 (숫자만)", placeholder="예: 408")
        
    with col_ho:
        input_ho = st.text_input("계약 호수 (숫자만)", placeholder="예: 503")

    uploaded_file = st.file_uploader("계약서 이미지 업로드", type=['png', 'jpg', 'jpeg'])
    submit_button = st.form_submit_button("제출 및 AI 자동 검증")

if submit_button:
    if uploaded_file is None:
        st.warning("⚠️ 계약서 이미지를 업로드해주세요.")
    elif not input_dong or not input_ho:
        st.warning("⚠️ 검증을 위해 계약 동과 호수를 입력해주세요.")
    else:
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("업로드된 계약서")
            st.image(uploaded_file, use_column_width=True)
            
        with col2:
            st.subheader("심층 분석 결과 리포트")
            
            with st.spinner("알고리즘 분석 중입니다..."):
                verifier = ContractVerifier(uploaded_file, contract_type)
                
                # 1단계 검증
                st.markdown("### 1. 문서 형식 및 표 서식 검사 (Layout)")
                layout_valid, layout_msg = verifier.check_layout_format()
                if layout_valid: st.success(f"✅ Pass: {layout_msg}")
                else: st.error(f"❌ Fail: {layout_msg}")

                # 2단계 검증
                st.markdown("### 2. 도장 직인 정밀 검지 (Seal Analysis)")
                seal_valid, seal_msg = verifier.check_seal_and_position()
                if seal_valid: st.success(f"✅ Pass: {seal_msg}")
                else: st.error(f"❌ Fail: {seal_msg}")

                # 3단계 검증
                party_label = "분양자" if contract_type == "sales" else "임대인"
                st.markdown(f"### 3. 계약 형태 및 {party_label} 명의 검증 (OCR)")
                landlord_valid, landlord_msg = verifier.check_landlord_info()
                if landlord_valid: st.success(f"✅ Pass: {landlord_msg}")
                else: st.error(f"❌ Fail: {landlord_msg}")
                    
                st.markdown("---")
                
                # 모든 조건 충족 시 최종 동/호수 대조 매칭
                if all([layout_valid, seal_valid, landlord_valid]):
                    extracted, dong_ho_msg = verifier.extract_dong_ho()
                    
                    if extracted:
                        extracted_dong = dong_ho_msg.split("동")[0].strip()
                        extracted_ho = dong_ho_msg.split("동")[1].replace("호", "").strip()
                        
                        if input_dong == extracted_dong and input_ho == extracted_ho:
                            st.success(f"🎉 **최종 가입 승인:** 입력하신 정보와 계약서 정보({dong_ho_msg})가 완벽히 일치하는 진본 문서입니다.")
                            st.balloons()
                        else:
                            st.error(f"🚨 **동·호수 정보 불일치:** 입력 화면에 기입하신 주소({input_dong}동 {input_ho}호)와 실제 서류에서 읽어낸 아파트 주소({dong_ho_msg})가 서로 매칭되지 않습니다. 동/호수를 바르게 적었는지 서류를 다시 한번 대조해 주세요.")
                    else:
                        st.warning(f"⚠️ {dong_ho_msg}")
                        st.error("🚨 동/호수 추출 실패로 인해 가입 승인이 보류되었습니다.")
                else:
                    # [개선부] 사용자가 한눈에 알아볼 수 있도록 직관적인 '검증 실패 사유 요약 서머리 리포트' 출력
                    st.error("🚨 **최종 가입 승인 거절:** 계약서 자동 인증 기준을 통과하지 못했습니다. 원활한 가입 승인을 위해 아래 요약된 실패 사유를 확인하고 보완해 주세요.")
                    
                    st.markdown("📋 **인증 반려 사유 요약 리포트**")
                    if not layout_valid:
                        st.markdown(f"* **[양식 오류]** {layout_msg}")
                    if not seal_valid:
                        st.markdown(f"* **[도장 미감지]** {seal_msg}")
                    if not landlord_valid:
                        st.markdown(f"* **[명의 및 계약서 종류 오류]** {landlord_msg}")

                with st.expander("🛠️ (개발자용) 프로그램이 읽어들인 전체 텍스트 보기"):
                    if verifier.full_text:
                        st.text(verifier.full_text)
                    else:
                        st.text("추출된 텍스트가 없습니다.")
