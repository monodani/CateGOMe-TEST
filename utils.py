# /utils.py

import streamlit as st
import requests
import os
from pathlib import Path

# --- 설정값 (Configuration) ---
# 아래 딕셔너리에 GitHub Raw URL과 로컬 저장 경로를 추가/수정하여 관리합니다.
DATA_FILES = {
    # Vectorstores - cases
    "cases_faiss": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/vectorstores/cases/cases_index.faiss",
        "local_path": "vectorstores/cases/cases_index.faiss"
    },
    "cases_pkl": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/vectorstores/cases/cases_index.pkl",
        "local_path": "vectorstores/cases/cases_index.pkl"
    },
    # Vectorstores - classification
    "classification_faiss": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/vectorstores/classification/classification_index.faiss",
        "local_path": "vectorstores/classification/classification_index.faiss"
    },
    "classification_pkl": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/vectorstores/classification/classification_index.pkl",
        "local_path": "vectorstores/classification/classification_index.pkl"
    },
    # Main Data CSV
    "classification_csv": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/data/classification_code.csv",
        "local_path": "data/classification_code.csv"
    },
    # Asset Images - CateGOMe 폴더
    "logo_main": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/assets/CateGOMe/CateGOMe_kor.png",
        "local_path": "assets/CateGOMe/CateGOMe_kor.png"
    },
    "logo_eng": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/assets/CateGOMe/CateGOMe_eng.png",
        "local_path": "assets/CateGOMe/CateGOMe_eng.png"
    },
    "logo_icon": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/assets/CateGOMe/CateGOMe_logo.png",
        "local_path": "assets/CateGOMe/CateGOMe_logo.png"
    },
    # Asset Images - emoji 폴더
    "emoji_hi": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/assets/emoji/CateGOMe_emoji_hi.png",
        "local_path": "assets/emoji/CateGOMe_emoji_hi.png"
    },
    "emoji_categorizing": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/assets/emoji/CateGOMe_emoji_categorying.png",
        "local_path": "assets/emoji/CateGOMe_emoji_categorying.png"
    },
    "emoji_sorry": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/assets/emoji/CateGOMe_emoji_sorry.png",
        "local_path": "assets/emoji/CateGOMe_emoji_sorry.png"
    },
    "emoji_00": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/assets/emoji/CateGOMe_emoji_00.png",
        "local_path": "assets/emoji/CateGOMe_emoji_00.png"
    },
    "emoji_haha": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/assets/emoji/CateGOMe_emoji_haha.png",
        "local_path": "assets/emoji/CateGOMe_emoji_haha.png"
    },
    "emoji_heart": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/assets/emoji/CateGOMe_emoji_heart.png",
        "local_path": "assets/emoji/CateGOMe_emoji_heart.png"
    },
    "emoji_okay": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/assets/emoji/CateGOMe_emoji_okay.png",
        "local_path": "assets/emoji/CateGOMe_emoji_okay.png"
    }
}

# --- 핵심 기능 (Core Logic) ---

def download_file(url: str, local_path: str):
    """
    주어진 URL에서 파일을 다운로드하여 지정된 로컬 경로에 저장합니다.
    경로가 존재하지 않으면 자동으로 생성합니다.
    """
    if not os.path.exists(local_path):
        # 파일이 위치할 디렉토리 생성
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except requests.exceptions.RequestException as e:
            st.error(f"파일 다운로드 실패: {local_path} ({e})")
            return False
    return True

@st.cache_resource
def initialize_app_data():
    """
    앱 실행에 필요한 모든 데이터 파일을 다운로드합니다.
    Streamlit의 @st.cache_resource를 사용하여 앱 세션 간에 한번만 실행되도록 합니다.
    성공 시, 로컬 파일 경로가 담긴 딕셔너리를 반환합니다.
    """
    st.write("필수 데이터 파일을 다운로드하고 앱을 초기화하는 중입니다...")
    
    local_paths = {}
    all_successful = True
    
    for key, info in DATA_FILES.items():
        if download_file(info["url"], info["local_path"]):
            local_paths[key] = info["local_path"]
        else:
            all_successful = False
            local_paths[key] = None # 실패한 경우 None으로 표시
    
    if all_successful:
        st.write("초기화 완료!")
        return local_paths
    else:
        st.error("일부 필수 파일을 다운로드하지 못했습니다. 앱이 정상적으로 동작하지 않을 수 있습니다.")
        return None
