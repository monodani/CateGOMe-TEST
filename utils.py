# /utils.py (FINAL FIXED VERSION 2)

import streamlit as st
import requests
import os
from pathlib import Path
# 'Image'와 'io'는 이 파일에서 직접 사용되지 않으므로 제거합니다.

# --- 설정값 (Configuration) ---
# 아래 딕셔너리에 GitHub Raw URL과 로컬 저장 경로를 추가/수정하여 관리합니다.
DATA_FILES = {
    # Vectorstores - cases
    "cases_faiss": {
        # <<< 여기가 수정된 부분입니다: 'case_index.faiss' -> 'cases_index.faiss'
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/vectorstores/cases/cases_index.faiss",
        "local_path": "vectorstores/cases/cases_index.faiss"
    },
    "cases_pkl": {
        # <<< 여기가 수정된 부분입니다: 'case_index.pkl' -> 'cases_index.pkl'
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
    # Asset Images
    "logo_main": {
        "url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/assets/CateGOMe/CateGOMe_kor.png",
        "local_path": "assets/CateGOMe_kor.png"
    },
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
    }
}

# --- 핵심 기능 (Core Logic) ---
def download_file(url: str, local_path: str):
    """
    주어진 URL에서 파일을 다운로드하여 지정된 로컬 경로에 저장합니다.
    """
    if not os.path.exists(local_path):
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
    """
    # st.write는 로딩 중에 여러 번 호출될 수 있으므로 st.spinner 외부에서는 제거하는 것이 깔끔합니다.
    local_paths = {}
    all_successful = True
    
    for key, info in DATA_FILES.items():
        if download_file(info["url"], info["local_path"]):
            local_paths[key] = info["local_path"]
        else:
            all_successful = False
            local_paths[key] = None
    
    if not all_successful:
        st.error("일부 필수 파일을 다운로드하지 못했습니다. 앱이 정상적으로 동작하지 않을 수 있습니다.")
    
    return local_paths
