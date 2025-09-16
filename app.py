---

### **4. `app.py`**

사용자 인터페이스와 앱의 전체 흐름을 제어하는 메인 파일입니다.

```python
# /app.py

import streamlit as st
from utils import initialize_app_data
from cate_gome_logic import initialize_models_and_data, get_classification_report

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="카테고미(CateGOMe)",
    page_icon="🐻",
    layout="wide"
)

# --- 앱 데이터 및 모델 초기화 ---
# utils.py를 통해 파일 다운로드 및 캐싱
# 성공 시, 다운로드된 파일의 로컬 경로가 담긴 딕셔너리를 반환
local_paths = initialize_app_data()

# API 키 확인
openai_api_key = st.secrets.get("OPENAI_API_KEY")
genai_api_key = st.secrets.get("GENAI_API_KEY")

if not openai_api_key or not genai_api_key:
    st.error("API 키가 설정되지 않았습니다. Streamlit Secrets에 OPENAI_API_KEY와 GENAI_API_KEY를 설정해주세요.")
    st.stop() # API 키 없으면 앱 실행 중지

# cate_gome_logic.py의 모델 및 데이터 로더 호출
# 이 과정은 캐시된 리소스 내에서 처리되므로, 모델도 한 번만 로드됩니다.
if local_paths:
    is_initialized, message = initialize_models_and_data(openai_api_key)
    if not is_initialized:
        st.error(message)
        st.stop()
else:
    st.error("데이터 파일 초기화에 실패하여 앱을 시작할 수 없습니다.")
    st.stop()


# --- UI 구성 ---

# 1. 헤더 (로고와 소개)
if local_paths and local_paths.get("logo_main"):
    st.image(local_paths["logo_main"], width=400)
st.title("가계부 자동 분류 서비스")
st.markdown("---")
if local_paths and local_paths.get("emoji_hi"):
    st.markdown(f"안녕하세요! 가계부 이미지를 업로드해주시면 제가 알아서 분류해 드릴게요. <img src='file://{local_paths['emoji_hi']}' width='30'>", unsafe_allow_html=True)

# 2. 세션 상태 초기화 (결과 로그 저장용)
if 'results_log' not in st.session_state:
    st.session_state['results_log'] = []

# 3. 이미지 업로드 인터페이스
uploaded_file = st.file_uploader(
    "여기에 가계부 이미지를 드래그 앤 드롭하거나 클릭하여 업로드하세요.",
    type=['png', 'jpg', 'jpeg']
)

# 4. '분류시작' 버튼 및 로직 실행
if uploaded_file is not None:
    st.image(uploaded_file, caption="업로드된 이미지", use_column_width=True)
    
    if st.button("분류시작", type="primary"):
        # 스피너 (로딩 메시지) 표시
        spinner_message = "카테고미가 열심히 분류 중이에요... 잠시만 기다려주세요! "
        if local_paths and local_paths.get("emoji_categorizing"):
            spinner_message += f"<img src='file://{local_paths['emoji_categorizing']}' width='40'>"
        
        with st.spinner(spinner_message):
            try:
                # 이미지 파일을 바이트로 변환
                image_bytes = uploaded_file.getvalue()
                
                # 핵심 로직 함수 호출
                report = get_classification_report(image_bytes, openai_api_key, genai_api_key)
                
                # 결과 로그를 세션 상태에 추가 (최신 결과가 위로 오도록)
                st.session_state.results_log.insert(0, report)

            except Exception as e:
                error_message = f"죄송해요, 처리 중 오류가 발생했어요. <img src='file://{local_paths.get('emoji_sorry', '')}' width='30'>"
                st.error(error_message, icon="🚨")
                st.error(f"오류 상세 내용: {e}")

st.markdown("---")

# 5. 누적된 결과 로그 출력
st.subheader("분석 결과 로그")
if not st.session_state.results_log:
    st.info("아직 분석한 내역이 없어요. 이미지를 업로드하고 '분류시작' 버튼을 눌러주세요!")
else:
    for i, report_md in enumerate(st.session_state.results_log):
        with st.expander(f"결과 로그 #{len(st.session_state.results_log) - i} (클릭하여 확인)", expanded=(i==0)):
            st.markdown(report_md, unsafe_allow_html=True)
