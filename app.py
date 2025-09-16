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

# def display_image_or_emoji(paths_dict: dict, key: str, fallback_emoji: str, width: int = 30) -> str:
#     """
#     이미지 파일이 존재하면 Base64로 인코딩된 HTML 이미지 태그를 반환하고,
#     없으면 fallback 이모지를 반환합니다.
#     """
#     try:
#         image_path = paths_dict.get(key)
#         if image_path and os.path.exists(image_path):
#             with open(image_path, "rb") as f:
#                 content = f.read()
#             b64_content = base64.b64encode(content).decode()
#             return f'<img src="data:image/png;base64,{b64_content}" width="{width}">'
#     except Exception:
#         pass  # 파일 처리 중 오류가 발생하면 fallback 이모지를 반환
#     return fallback_emoji
    

# --- 앱 데이터 및 모델 초기화 ---

# with 블록을 사용하여 초기화 과정을 명확하게 표시
with st.spinner("카테고미 서비스에 필요한 데이터를 준비 중이에요... 🤖"):
    local_paths = initialize_app_data()

# API 키 확인
openai_api_key = st.secrets.get("OPENAI_API_KEY")
genai_api_key = st.secrets.get("GENAI_API_KEY")

if not openai_api_key or not genai_api_key:
    st.error("API 키가 설정되지 않았습니다. Streamlit Secrets에 OPENAI_API_KEY와 GENAI_API_KEY를 설정해주세요.")
    st.stop()

# 모델 초기화 (데이터 다운로드 성공 여부와 관계없이 시도)
if local_paths:
    # is_initialized 변수를 사용하여 초기화 성공 여부를 명확히 관리
    is_initialized, message = initialize_models_and_data(openai_api_key)
    if not is_initialized:
        # <<< 초기화 실패 시, 구체적인 오류 메시지를 사용자에게 보여주고 앱을 중지
        st.error(f"서비스를 시작할 수 없습니다. 모델 초기화에 실패했습니다.")
        st.error(f"상세 원인: {message}")
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
        st.markdown("# 🐻 카테고미")
else:
    st.markdown("# 🐻 카테고미")

st.title("가계동향조사 항목분류 AI챗봇")
st.markdown("---")

# 인사말 - base64 이미지 또는 이모지 사용
hi_emoji = display_image_or_emoji(local_paths, "emoji_hi", "👋", 30)
st.markdown(f"안녕하세요, 가계부 이미지를 업로드해주시면 제가 알아서 분류해 드릴게요! {hi_emoji}", unsafe_allow_html=True)

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
    st.image(uploaded_file, caption="업로드된 이미지", use_container_width=True)
    
    if st.button("분류시작", type="primary"):
        spinner_message = "카테고미가 열심히 분류 중이에요... 잠시만 기다려주세요!"
        
        with st.spinner(spinner_message):
            try:
                image_bytes = uploaded_file.getvalue()
                report = get_classification_report(image_bytes, openai_api_key, genai_api_key)
                st.session_state.results_log.insert(0, report)
            except Exception as e:
                error_message = f"죄송해요, 처리 중 오류가 발생했어요."
                st.error(error_message, icon="🚨")
                st.error(f"오류 상세 내용: {e}")

st.markdown("---")


# 5. 누적된 결과 로그 출력
st.subheader("분석결과 로그")
if not st.session_state.results_log:
    st.info("아직 분류한 내역이 없어요. 이미지를 업로드하고 '분류시작' 버튼을 눌러주세요!")
else:
    for i, report_md in enumerate(st.session_state.results_log):
        with st.expander(f"결과 로그 #{len(st.session_state.results_log) - i} (클릭하여 확인)", expanded=(i==0)):
            st.markdown(report_md, unsafe_allow_html=True)
