
---

### **3단계: `app.py` 전체 교체**

`load_all_data_and_models` 함수를 호출하고, 그 결과를 `get_classification_report` 함수에 넘겨주기만 하면 됩니다. `use_container_width` 경고도 수정했습니다.

```python
# /app.py (FINAL VERSION - ALL-IN-ONE)

import streamlit as st
import os
import base64
from cate_gome_logic import load_all_data_and_models, get_classification_report

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="카테고미(CateGOMe)", page_icon="🐻", layout="wide")

# --- 헬퍼 함수 정의 ---
def display_image_or_emoji(path: str, fallback_emoji: str, width: int = 30) -> str:
    """이미지 파일이 있으면 Base64 HTML 태그를, 없으면 fallback 이모지를 반환합니다."""
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                content = base64.b64encode(f.read()).decode()
            return f'<img src="data:image/png;base64,{content}" width="{width}">'
        except Exception:
            return fallback_emoji
    return fallback_emoji

# --- API 키 및 데이터 로딩 ---
openai_api_key = st.secrets.get("OPENAI_API_KEY")
genai_api_key = st.secrets.get("GENAI_API_KEY")

if not openai_api_key or not genai_api_key:
    st.error("API 키가 설정되지 않았습니다. Streamlit Secrets을 확인해주세요.")
    st.stop()

# 단일 함수로 모든 데이터와 모델을 로드합니다.
with st.spinner("카테고미 서비스에 필요한 데이터를 준비 중이에요... 🤖"):
    loaded_data = load_all_data_and_models(openai_api_key)

# --- UI 구성 ---
# 로고 이미지는 이제 GitHub에서 직접 다운로드하지 않고, 로컬 경로를 사용합니다.
st.image("assets/CateGOMe_kor.png", width=400) # 로고 경로는 assets 폴더 기준으로 고정
st.title("가계부 자동 분류 서비스")
st.markdown("---")

hi_emoji = display_image_or_emoji("assets/emoji/CateGOMe_emoji_hi.png", "👋")
st.markdown(f"안녕하세요! 가계부 이미지를 업로드해주시면 제가 알아서 분류해 드릴게요. {hi_emoji}", unsafe_allow_html=True)

if 'results_log' not in st.session_state:
    st.session_state['results_log'] = []

uploaded_file = st.file_uploader("여기에 가계부 이미지를 드래그 앤 드롭하거나 클릭하여 업로드하세요.", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # use_container_width 경고 수정
    st.image(uploaded_file, caption="업로드된 이미지", width=None) # width=None 으로 설정 시 자동으로 맞춰짐

    if st.button("분류시작", type="primary"):
        categorizing_emoji = display_image_or_emoji("assets/emoji/CateGOMe_emoji_categorying.png", "🤓", 40)
        spinner_message = f"카테고미가 열심히 분류 중이에요... 잠시만 기다려주세요! {categorizing_emoji}"
        
        with st.spinner(spinner_message):
            try:
                image_bytes = uploaded_file.getvalue()
                # 로드된 데이터를 get_classification_report 함수에 전달
                report = get_classification_report(image_bytes, genai_api_key, loaded_data)
                st.session_state.results_log.insert(0, report)
            except Exception as e:
                sorry_emoji = display_image_or_emoji("assets/emoji/CateGOMe_emoji_sorry.png", "🚨")
                st.error(f"죄송해요, 처리 중 오류가 발생했어요. {sorry_emoji}", icon=" ")
                st.error(f"오류 상세 내용: {e}")

st.markdown("---")
st.subheader("분석 결과 로그")

if not st.session_state.results_log:
    st.info("아직 분석한 내역이 없어요. 이미지를 업로드하고 '분류시작' 버튼을 눌러주세요!")
else:
    for i, report_md in enumerate(st.session_state.results_log):
        with st.expander(f"결과 로그 #{len(st.session_state.results_log) - i} (클릭하여 확인)", expanded=(i==0)):
            st.markdown(report_md, unsafe_allow_html=True)
