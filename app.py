# **app.py** / 사용자 인터페이스와 앱의 전체 흐름을 제어하는 메인 파일
import streamlit as st
import base64
import os
from utils import initialize_app_data
from cate_gome_logic import initialize_models_and_data, get_classification_report

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="카테고미(CateGOMe)",
    page_icon="🐻",
    layout="wide"
)

# --- 헬퍼 함수들 ---
def get_base64_image(image_path):
    """이미지를 base64 문자열로 변환"""
    try:
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except:
        pass
    return None

def display_image_or_emoji(local_paths, image_key, emoji_fallback, size=30):
    """이미지 표시 시도, 실패시 이모지 대체"""
    img_base64 = get_base64_image(local_paths.get(image_key))
    if img_base64:
        return f'<img src="data:image/png;base64,{img_base64}" width="{size}">'
    return emoji_fallback

# --- 앱 데이터 및 모델 초기화 ---
# utils.py를 통해 파일 다운로드 및 캐싱
local_paths = initialize_app_data()

# API 키 확인
openai_api_key = st.secrets.get("OPENAI_API_KEY")
genai_api_key = st.secrets.get("GENAI_API_KEY")

if not openai_api_key or not genai_api_key:
    st.error("API 키가 설정되지 않았습니다. Streamlit Secrets에 OPENAI_API_KEY와 GENAI_API_KEY를 설정해주세요.")
    st.stop()

# cate_gome_logic.py의 모델 및 데이터 로더 호출
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
    try:
        from PIL import Image
        img = Image.open(local_paths["logo_main"])
        st.image(img, width=400)
    except Exception as e:
        st.markdown("# 🐻 카테고미(CateGOMe)")
else:
    st.markdown("# 🐻 카테고미(CateGOMe)")

st.title("가계부 자동 분류 서비스")
st.markdown("---")

# 인사말 - base64 이미지 또는 이모지 사용
hi_emoji = display_image_or_emoji(local_paths, "emoji_hi", "👋", 30)
st.markdown(f"안녕하세요! {hi_emoji} 가계부 이미지를 업로드해주시면 제가 알아서 분류해 드릴게요.", unsafe_allow_html=True)

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
        # 스피너 메시지 - base64 이미지 또는 이모지 사용
        categorizing_emoji = display_image_or_emoji(local_paths, "emoji_categorizing", "⏳", 40)
        spinner_message = f"카테고미가 열심히 분류 중이에요... 잠시만 기다려주세요! {categorizing_emoji}"
        
        with st.spinner("카테고미가 열심히 분류 중이에요... 잠시만 기다려주세요! ⏳"):
            # HTML을 포함한 spinner 메시지는 st.spinner에서 지원하지 않으므로
            # 별도의 st.markdown으로 표시
            st.markdown(spinner_message, unsafe_allow_html=True)
            
            try:
                # 이미지 파일을 바이트로 변환
                image_bytes = uploaded_file.getvalue()
                
                # 핵심 로직 함수 호출
                report = get_classification_report(image_bytes, openai_api_key, genai_api_key)
                
                # 결과 로그를 세션 상태에 추가 (최신 결과가 위로 오도록)
                st.session_state.results_log.insert(0, report)
                
            except Exception as e:
                sorry_emoji = display_image_or_emoji(local_paths, "emoji_sorry", "😔", 30)
                error_message = f"죄송해요, 처리 중 오류가 발생했어요. {sorry_emoji}"
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
