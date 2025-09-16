# ========================================
# 🔧 설정값 (Colab 코드 그대로)
# ========================================
import streamlit as st

# API Key 설정 (Streamlit secrets 사용)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GENAI_API_KEY = st.secrets["GENAI_API_KEY"]

# --- Global (1회 로드 캐시) ----------------------------------------------------
EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"  # 통합 모델명 변수 사용

VECTORSTORE_DIR_CASES = "vectorstores/cases"
INDEX_NAME_CASES = "cases_index"
VECTORSTORE_DIR_CLASSIFICATION = "vectorstores/classification"
INDEX_NAME_CLASSIFICATION = "classification_index"
CSV_PATH = "data/classification_code.csv"

REQUIRED_COLS = ["항목명", "입력코드", "처리코드", "항목분류내용", "포함항목", "제외항목"]

# ========================================
# 📦 라이브러리 임포트 (Colab 코드 그대로)
# ========================================
import os
import re
import json
import ast
from typing import List, Dict, Any
from operator import itemgetter

import pandas as pd
from PIL import Image
import google.generativeai as genai

from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Gemini 설정
genai.configure(api_key=GENAI_API_KEY)

# ========================================
# Streamlit 페이지 설정
# ========================================
st.set_page_config(
    page_title="카테고미 - 통계청 항목자동분류AI",
    page_icon="📊",
    layout="wide"
)

# ========================================
# Colab 초기화 코드 그대로 (캐싱 추가)
# ========================================
@st.cache_resource
def initialize_system():
    try:
        _embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)

        # === 수정된 부분: 두 개의 벡터스토어 로드 ===
        _vectorstore_cases = FAISS.load_local(
            folder_path=VECTORSTORE_DIR_CASES,
            embeddings=_embeddings,
            index_name=INDEX_NAME_CASES,
            allow_dangerous_deserialization=True
        )

        _vectorstore_classification = FAISS.load_local(
            folder_path=VECTORSTORE_DIR_CLASSIFICATION,
            embeddings=_embeddings,
            index_name=INDEX_NAME_CLASSIFICATION,
            allow_dangerous_deserialization=True
        )

        # 벡터스토어들을 리스트로 관리하여 확장성 확보
        _vectorstores = {
            "cases": _vectorstore_cases,
            "classification": _vectorstore_classification
        }
        # ============================================

        _df = pd.read_csv(CSV_PATH, encoding='utf-8')
        missing = [c for c in REQUIRED_COLS if c not in _df.columns]
        if missing:
            raise KeyError(f"ERROR[csv]: Missing required columns: {missing}")
        # 고유키(중복 제거용) 없으면 행 인덱스 사용
        _df = _df.reset_index(drop=False).rename(columns={"index": "_rowid"})

        # LLM 모델도 캐시
        _llm_model = ChatOpenAI(
            model_name=LLM_MODEL,
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY
        )

        return _embeddings, _vectorstores, _df, _llm_model

    except Exception as e:
        st.error(f"초기화 실패: {e}")
        return None, None, None, None

# 초기화
_embeddings, _vectorstores, _df, _llm_model = initialize_system()

# ========================================
# Colab 헬퍼 함수들 그대로
# ========================================

        
def _short_doc_from_row(row: pd.Series) -> Document:
    """
    page_content는 토큰 낭비를 줄이기 위해 핵심 필드만.
    나머지는 metadata에 담는다.
    '출처' 정보를 page_content 맨 앞에 추가하고, '입력코드'를 정수형으로 변환합니다.
    """
    source = row.get('출처', '항목분류집')
    source_info = f"출처: {source}\n"

    core_fields_order = [col for col in ["입력코드", "항목명", "항목분류내용", "처리코드", "포함항목", "제외항목"] if col in row.index]

    core_lines = []
    for col in core_fields_order:
        value = row[col]

        # '입력코드' 컬럼일 경우, 정수로 변환을 시도합니다.
        if col == "입력코드":
            try:
                # float으로 먼저 변환 후 int로 변환하여 "720.0" 같은 문자열도 처리
                value_str = str(int(float(value)))
            except (ValueError, TypeError):
                # 변환 실패 시 (예: NaN, 비숫자 문자열) 원본 값을 그대로 사용
                value_str = str(value)
        else:
            value_str = str(value)
        # ============================

        core_lines.append(f"{col}: {value_str}")

    page = source_info + "\n".join(core_lines)
    meta = row.to_dict()
    return Document(page_content=page, metadata=meta)


def _keyword_search(df: pd.DataFrame, term: str) -> List[Document]:
    """부분일치 contains, 대소문자 무시. 상한 없음(요청 반영)."""
    if df is None:  # 초기화 실패 시
        return []
    # NaN 안전 처리 및 타입 변환
    df_copy = df.copy()  # 원본 데이터프레임 변경 방지
    for c in REQUIRED_COLS:
        if c in df_copy.columns and df_copy[c].dtype != object:
            df_copy[c] = df_copy[c].astype(str)

    mask = (
        df_copy["항목분류내용"].str.contains(term, case=False, na=False) |
        df_copy["항목명"].str.contains(term, case=False, na=False) |
        df_copy["포함항목"].str.contains(term, case=False, na=False) |
        df_copy["제외항목"].str.contains(term, case=False, na=False)
    )
    sub = df_copy.loc[mask]
    # 중복 제거(행 인덱스 기반)
    sub = sub.drop_duplicates(subset=["_rowid"], keep="first")
    return [_short_doc_from_row(r) for _, r in sub.iterrows()]


def _keyword_search_on_docs(docs: List[Document], term: str) -> List[Document]:
    """메모리에 로드된 Document 객체 리스트에서 직접 키워드 검색을 수행합니다."""
    if not docs:
        return []

    # page_content에 term이 포함된 모든 문서를 반환 (대소문자 무시)
    return [doc for doc in docs if term.lower() in doc.page_content.lower()]


def _similarity_topk_for_term(vs: FAISS, embeddings: OpenAIEmbeddings, term: str, k: int = 2) -> List[Document]:
    if vs is None or embeddings is None:  # 초기화 실패 시
        return []
    retriever = vs.as_retriever(
        search_type="mmr",  # MMR 사용 유지
        search_kwargs={"k": k, "fetch_k": 30, "lambda_mult": 0.5}
    )
    return retriever.invoke(term)

def _get_term_info_via_llm(llm: ChatOpenAI, user_query: str, num_related_terms: int = 3) -> List[Dict[str, Any]]:
    """
    LLM을 호출하여 사용자 쿼리에서 핵심 품목명들을 추출하고, 각 품목명에 대한 설명과 관련 용어를 받습니다.
    안정적인 JSON 추출을 위해 프롬프트와 파싱 로직이 강화되었습니다.
    """
    if llm is None:
        return []

    # === 품목 설명 및 관련어 반환 프롬프트 (Colab 그대로) ===
    prompt = f"""
너는 사용자 쿼리에서 'product_name' 리스트에 포함된 모든 품목명을 분석하고, 오탈자를 교정한 뒤 검색에 유용한 정보를 추출하는 전문가 AI이다.

## 작업 절차 (반드시 순서대로 따를 것) ##
1. **품목명 추출:** `product_name = [...]` 리스트에 있는 모든 원본 품목명을 빠짐없이 추출한다.
2. **오탈자 교정:** 각 원본 품목명의 오탈자나 불분명한 표현을 가장 자연스럽고 일반적인 표현으로 수정한다. (예: "파플레시티" -> "퍼플렉시티") 만약 수정할 필요가 없다면 원본을 그대로 사용한다.
3. **설명 생성:** **수정된 품목명**에 대해, 그 품목의 본질과 목적을 2~3 문장으로 간결하게 설명한다.
4. **관련 용어 추출 (매우 중요):**
   - **수정된 품목명**과 **네가 작성한 설명**을 모두 참고하여, 검색에 가장 중요하다고 판단되는 핵심 관련 용어를 {num_related_terms}개 추출한다.
   - **단순 동의어를 넘어, 그 품목의 '목적'이나 '상위 카테고리'에 해당하는 개념적인 단어를 반드시 포함해야 한다.** (예: '전기차 충전'의 목적은 '연료'를 채우는 것이므로 '연료'나 '에너지'를 포함)

5. **JSON 출력:** 다른 어떤 설명도 없이, 아래 "출력 예시"와 **완벽하게 동일한 JSON 형식**으로만 응답한다.

## 입력 및 출력 예시 (매우 중요) ##
### 입력 쿼리 예시:
'''product_name = ['모둠쌈', '전기차 충전', '파플레시티'] ...'''

### 너의 JSON 출력 예시:
```json
{{
  "terms": [
    {{
      "original_term": "모둠쌈",
      "term": "모둠쌈",
      "description": 상추·깻잎 등 다양한 쌈채소를 한데 묶어 제공하는 구성으로, 밥과 고기, 쌈장을 곁들여 싸 먹는 한국식 식사 방식입니다. 채소 섭취를 늘리고 담백하게 즐기려는 목적의 메뉴로도 활용됩니다.",
      "related_terms": [한식", "쌈", "채소"]
    }},
    {{
      "original_term": "전기차 충전",
      "term": "전기차 충전",
      "description": "전기 자동차의 배터리에 전력을 공급하는 행위입니다. 자동차를 운행하기 위한 에너지를 채우는 과정입니다.",
      "related_terms": ["전기차", "충전소", "연료"]
    }},
    {{
      "original_term": "파플레시티",
      "term": "퍼플렉시티",
      "description": "실시간 웹검색과 생성형 AI 질의응답을 결합한 검색 서비스입니다. 질문에 대해 요약된 답변과 함께 관련 출처를 제시해 신속한 정보탐색·리서치를 돕습니다.",
      "related_terms": ["AI 검색엔진", "웹검색", "AI"]
    }}
  ]
}}
---
이제 아래 사용자 입력 쿼리를 처리해라. 다른 말은 절대 하지 말고 JSON만 출력해라.

사용자 입력 쿼리: {user_query}
"""

    try:
        res = llm.invoke(prompt)
        text_content = res.content.strip()

        json_match = re.search(r'\{.*\}', text_content, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("LLM 응답에서 유효한 JSON 객체를 찾지 못했습니다.", text_content, 0)

        json_str = json_match.group(0)
        data = json.loads(json_str)

        terms_info = data.get("terms", [])

        # 기본 정리
        cleaned_terms_info = []
        for item in terms_info:
            if item.get("term") and item.get("description"):
                cleaned_related = []
                for rt in item.get("related_terms", []):
                    rt_norm = re.sub(r"\s+", " ", str(rt)).strip().lower()
                    if rt_norm:
                        cleaned_related.append(re.sub(r"\s+", " ", str(rt)).strip())
                cleaned_terms_info.append({
                    "term": re.sub(r"\s+", " ", str(item["term"])).strip(),
                    "description": item["description"].strip(),
                    "related_terms": cleaned_related[:num_related_terms]
                })
        return cleaned_terms_info

    except Exception as e:
        # 폴백 로직
        return [{"term": user_query, "description": "", "related_terms": []}]

# search_classification_codes 함수 (Colab 그대로)
def search_classification_codes(
    user_query: str,
    all_docs_from_vs: Dict[str, List[Document]],  # 파라미터
    sim_topk_per_term: int = 3,  # 유사도 검색 결과 개수
    num_related_terms: int = 3  # LLM 관련 용어 개수
) -> Dict[str, Any]:
    """
    사용자 쿼리에 대해 분류 코드를 검색합니다.
    (Colab 코드 그대로)
    """
    # 초기화 상태 확인
    if _df is None or _embeddings is None or _vectorstores is None or _llm_model is None or OPENAI_API_KEY is None:
        return {
            "query": user_query,
            "extracted_terms_info": [],
            "results": {"keyword": [], "similarity": []},
            "context_docs": [],
            "error": "시스템 초기화 실패 (데이터, 벡터스토어, LLM 또는 API 키). 관리자에게 문의하세요."
        }

    if not isinstance(user_query, str) or not user_query.strip():
        return {
            "query": user_query,
            "extracted_terms_info": [],
            "results": {"keyword": [], "similarity": []},
            "context_docs": [],
            "error": "유효하지 않은 사용자 쿼리입니다."
        }

    # 1. LLM을 사용하여 쿼리에서 핵심 용어 추출 및 설명, 관련 용어 받기
    extracted_terms_info = _get_term_info_via_llm(_llm_model, user_query, num_related_terms=num_related_terms)

    all_relevant_docs: List[Document] = []  # 키워드 또는 유사도 검색 결과를 모두 담을 리스트
    seen_docs_page_content = set()  # 중복 제거용

    all_keyword_docs_raw: List[Document] = []  # 디버깅용: 중복 포함 키워드 결과
    all_similarity_docs_raw: List[Document] = []  # 디버깅용: 중복 포함 유사도 결과

    for item in extracted_terms_info:
        term = item["term"]
        description = item["description"]
        related_terms = item.get("related_terms", [])

        # 2. 각 핵심 용어(원어)와 관련 용어에 대해 키워드 검색 수행
        terms_to_keyword_search = [term] + related_terms
        for search_term in terms_to_keyword_search:
            kw_docs = _keyword_search(_df, search_term)
            if kw_docs:
                all_keyword_docs_raw.extend(kw_docs)  # 디버깅용 결과 추가
                for doc in kw_docs:
                    if doc.page_content not in seen_docs_page_content:
                        all_relevant_docs.append(doc)
                        seen_docs_page_content.add(doc.page_content)

        # === 검색 B: 벡터스토어에 대한 키워드 검색 ===
        for name, doc_list in all_docs_from_vs.items():
            for search_term in terms_to_keyword_search:
                kw_docs_vs = _keyword_search_on_docs(doc_list, search_term)
                if kw_docs_vs:
                    docs_to_add = [Document(page_content=f"출처: {name}\n{doc.page_content}", metadata=doc.metadata) for doc in kw_docs_vs]
                    all_keyword_docs_raw.extend(docs_to_add)
                    for doc in docs_to_add:
                        if doc.page_content not in seen_docs_page_content:
                            all_relevant_docs.append(doc)
                            seen_docs_page_content.add(doc.page_content)

        # === 여러 벡터스토어에서 유사도 검색 수행 ===
        if description:
            for name, vs in _vectorstores.items():
                sim_docs = _similarity_topk_for_term(vs, _embeddings, description, k=sim_topk_per_term)
                if sim_docs:
                    docs_to_add = []
                    for doc in sim_docs:
                        new_doc = Document(page_content=f"출처: {name}\n{doc.page_content}", metadata=doc.metadata)
                        docs_to_add.append(new_doc)

                    all_similarity_docs_raw.extend(docs_to_add)
                    for doc in docs_to_add:
                        if doc.page_content not in seen_docs_page_content:
                            all_relevant_docs.append(doc)
                            seen_docs_page_content.add(doc.page_content)

    # 4. 수집된 모든 문서를 합치고 중복 제거 (all_relevant_docs에 이미 중복 제거되어 수집됨)
    unique_docs_objects = all_relevant_docs  # 변수명 통일

    return {
        "query": user_query,
        "extracted_terms_info": extracted_terms_info,
        "results": {
            "keyword": all_keyword_docs_raw,  # 모든 키워드 검색 결과 (중복 포함 가능)
            "similarity": all_similarity_docs_raw  # 모든 유사도 검색 결과 (중복 포함 가능)
        },
        "context_docs": unique_docs_objects  # GPT에 전달할 최종 중복 제거된 Document 객체 목록
    }

# prompt_template_single (Colab 프롬프트 그대로 사용)
prompt_template_single = PromptTemplate.from_template("""
    SYSTEM: 당신은 주어진 데이터를 분석하여 가장 적합한 '입력코드'와 '항목명'을 추론하는, 극도로 꼼꼼하고 규칙을 엄수하는 데이터 분류 AI이며, 당신의 이름은 "카테고미(CateGOMe)"입니다. 당신의 답변은 반드시 지정된 JSON 형식이어야 합니다.

    ## 절대 규칙 (가장 중요! 반드시 따를 것) ##
    1. **수입/지출 규칙:** `question`의 `expense` 값이 0보다 크면, `input_code`는 **절대로 1000 미만이 될 수 없습니다.** 반대로 `income` 값이 0보다 크면, `input_code`는 **절대로 1000 이상이 될 수 없습니다.** 예외는 없습니다.
    2. **정보 우선순위 규칙:** `context`에서 `출처: 조사사례집`(또는 cases) 정보는 `출처: 항목분류집` 정보보다 **항상 우선**합니다. 만약 두 정보가 충돌하면, 무조건 '조사사례집'의 코드를 따라야 합니다.

    ## 작업 절차 ##
    1. **입력 분석:** `question`의 `품목명`, `income`, `expense` 값을 확인하고 [절대 규칙 1]을 기억합니다.
    2. **컨텍스트 분석:**
        - `품목명`과 가장 일치하는 **'조사사례집'** 내용이 있는지 먼저 찾습니다.
        - 만약 명확한 사례가 있다면, [절대 규칙 2]에 따라 해당 코드를 **최우선 후보**로 고려합니다.
        - 명확한 사례가 없다면, '항목분류집'에서 가장 적합한 정의를 찾습니다.
    3. **분류 타입 결정:**
        - **(DEFINITE 조건):** 위의 과정을 거쳐, 단 하나의 입력코드를 90% 이상의 신뢰도로 확신할 수 있는 경우에만 "DEFINITE"로 결정합니다. (예: '챗지피티' -> '챗지피티 구독료' 사례가 명확히 존재)
        - **(AMBIGUOUS 조건):** 다음 중 하나라도 해당하면 **반드시 "AMBIGUOUS"**로 결정해야 합니다.
            - 품목명이 너무 일반적이어서 여러 코드가 후보가 될 때 (예: '고등어' -> 간고등어? 바다어류? 수산동물통조림? 알 수 없음)
            - **품목명이 특정 회사 이름이고, 그 회사가 다양한 종류의 상품/서비스를 제공하는 경우 (예: '네이버'  -> 온라인쇼핑몰, 페이 결제, 웹툰 등 다양한 서비스 상품이 있어 하나로 특정 불가)**
            - 소득의 주체(가구주, 배우자, 기타가구원 등)가 불명확하여 여러 코드가 후보가 될 때 (예: '급여' -> 가구주급여? 배우자급여? 기타가구원급여?)
    4. **JSON 출력:**결정된 분류 타입에 맞는 JSON 형식으로만 응답합니다. 다른 설명은 절대 추가하지 마세요.

    ---
    ## 좋은 예시와 나쁜 예시 ##

    - **Question:** product_name = ['할리스커피조각케익'], expense = [10000]
    - **Context:** ... [항목분류집] 케이크: 1085(케이크) ... [조사사례집] 커피숍 구매 조각 케익: 7560(주점·커피숍) ...
    - **나쁜 판단:** '케이크'라는 일반 분류를 보고 `1085`를 선택하는 것.
    - **좋은 판단:** [정보 우선순위 규칙]에 따라 '조사사례집'의 `7560`을 선택하고 "DEFINITE"로 분류.

    ---
    ## 출력 형식 (아래 형식 중 하나로만 응답) ##

    ### A. 명확한 경우 (DEFINITE):
    ```json
    {{
      "classification_type": "DEFINITE",
      "result": {{
        "input_code": "추론한 숫자 입력코드",
        "confidence": "신뢰도 (예: 95%)",
        "reason": "절대 규칙과 정보 우선순위 규칙에 입각하여 이 코드를 선택한 명확한 이유.",
        "evidence": "근거로 사용한 가장 핵심적인 컨텍스트 내용(청크) 하나를 그대로 복사"
      }}
    }}
    ```

    ### B. 모호한 경우 (AMBIGUOUS)
    ```json
    {{
      "classification_type": "AMBIGUOUS",
      "reason_for_ambiguity": "왜 단일 코드로 확정할 수 없는지에 대한 핵심 이유 (예: '보험의 종류(화재, 건강, 운전, 자동차 등)가 명시되지 않아 여러 후보가 가능함')",
      "candidates": [
        {{
          "input_code": "후보 입력코드 1"("후보 입력코드 1의 항목명"),
          "confidence": "후보 1의 신뢰도 (예: 50%)",
          "reason": "이 코드가 후보인 이유"
        }},
        {{
          "input_code": "후보 입력코드 2"("후보 입력코드 2의 항목명"),,
          "confidence": "후보 2의 신뢰도 (예: 30%)",
          "reason": "이 코드가 후보인 이유"
        }}
      ],
      "evidence": "판단에 사용된 가장 관련성 높은 컨텍스트 내용(청크) 하나를 그대로 복사"
    }}
    ```
    ---
    HUMAN:
    #Question: {question}
    #Context: {context}
    Answer:
""")

# 단일 품목 처리 전용 체인
classification_chain_single = (
    {"question": itemgetter("question"), "context": itemgetter("context")}
    | prompt_template_single
    | _llm_model
    | StrOutputParser()
)

def fmt_won(x):
    try:
        return f"{int(x):,}원"
    except Exception:
        return "0원"
        
# ========================================
# Streamlit UI (심플하게)
# ========================================
# 헤더
# 전역 어느 곳(예: 페이지 설정 아래)에 CSS 주입
st.markdown("""
<style>
.categome-center { display:flex; justify-content:center; align-items:center; }
.categome-caption { text-align:center; color:#666; margin-bottom:30px; }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown('<div class="categome-center">', unsafe_allow_html=True)
if os.path.exists("assets/CateGOMe_kor.png"):
    st.image("assets/CateGOMe_kor.png", width=420)
else:
    st.title("🤖 CateGOMe")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="categome-caption">
가계동향조사 항목코드 자동분류 AI 서비스<br>
가계부 이미지를 업로드하면 자동으로 품목을 분류해드립니다
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# 세션 스토리지 기본값
# ----------------------------------------------------------
st.session_state.setdefault("results", None)        # 전체 결과 캐시
st.session_state.setdefault("last_file_name", None) # 업로드 파일 변경 감지

# === 업로더(유일) ===
uploaded_file = st.file_uploader(
    "가계부 이미지를 업로드하세요",
    type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'],
    help="드래그 앤 드롭 또는 클릭하여 파일 선택",
    key="main_uploader_v2",
)

# 파일 바뀌면 결과 초기화
if uploaded_file is not None and st.session_state["last_file_name"] != uploaded_file.name:
    st.session_state["results"] = None
    st.session_state["last_file_name"] = uploaded_file.name

# ----------------------------------------------------------
# 버튼 (중앙)
# ----------------------------------------------------------
if uploaded_file is not None:
    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        run = st.button("🚀 분류 시작", type="primary", use_container_width=True, key="run_btn_v2")

    # ======================================================
    # 1) 무거운 파이프라인: 버튼 눌렀을 때만 실행
    #    실행 결과는 session_state["results"]에 저장
    # ======================================================
    if run:
        progress = st.progress(0, "이미지 분석 준비 중...")

        # --- 여기는 기존 파이프라인 그대로 (OCR → 검색 → LLM) ---
        #     단, 마지막에 df_definite / ambiguous_results / failed_results만 저장해주면 됩니다.
        img = Image.open(uploaded_file).convert("RGB")
        progress.progress(20, "📸 이미지에서 텍스트 추출 중...")

        gemini_model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = """
가계부 사진에서 표를 인식해서 각 행의
1) 품목명(= '수입종류 및 지출의 품명과 용도' 열),
2) 수입 금액,
3) 지출 금액
을 추출하라.

규칙:
- 금액의 쉼표(,)는 제거하고 정수로.
- 값이 비어 있으면 0으로.
- 제목행·체크박스·빈줄은 제외.
- 반드시 아래 JSON 스키마로만 출력.

JSON 스키마:
{
  "items": [
    {"name": "품목명", "income": 0, "expense": 0},
    ...
  ]
}
"""
        img_bytes = uploaded_file.getvalue()
        resp = gemini_model.generate_content(
            [{"text": prompt}, {"inline_data": {"mime_type": uploaded_file.type, "data": img_bytes}}],
            generation_config={"response_mime_type": "application/json"}
        )
        raw = resp.text
        data = json.loads(raw)

        # 후처리
        items = []
        for it in data.get("items", []):
            name = str(it.get("name", "")).strip()
            def to_int(x):
                s = str(x).replace(",", "").strip()
                return int(re.sub(r"[^\d]", "", s)) if re.search(r"\d", s) else 0
            income = to_int(it.get("income", 0))
            expense = to_int(it.get("expense", 0))
            if name:
                items.append({"name": name, "income": income, "expense": expense})

        product_name_list = [it["name"] for it in items]
        income_list        = [it["income"] for it in items]
        expense_list       = [it["expense"] for it in items]

        progress.progress(30, f"✅ {len(items)}개 품목 발견")

        # 코드→항목명 맵
        _df['입력코드_str'] = _df['입력코드'].astype(str).str.replace(r'\.0$', '', regex=True)
        code_to_name_map = pd.Series(_df.항목명.values, index=_df.입력코드_str).to_dict()

        # 벡터스토어 문서 메모리 로드
        all_docs_from_vs = {name: list(vs.docstore._dict.values()) for name, vs in _vectorstores.items()}

        # 결과 컨테이너
        definite_results, ambiguous_results, failed_results = [], [], []

        total = max(len(product_name_list), 1)
        for i, pname_orig in enumerate(product_name_list):
            progress.progress(30 + int(60 * (i + 1) / total), f"🔍 분류 중... ({i+1}/{total}) - {pname_orig}")

            q_single = f"product_name = ['{pname_orig}'], income = [{income_list[i]}], expense = [{expense_list[i]}]"
            search_output = search_classification_codes(q_single, all_docs_from_vs, sim_topk_per_term=3, num_related_terms=3)
            pname = (search_output.get("extracted_terms_info") or [{"term": pname_orig}])[0]["term"]

            if "error" in search_output or not search_output["context_docs"]:
                failed_results.append({"품목명": pname, "수입": income_list[i], "지출": expense_list[i], "실패 이유": "검색 결과 없음"})
                continue

            context = "\n\n---\n\n".join([d.page_content for d in search_output["context_docs"]])
            context = context.replace("출처: cases", "출처: 조사사례집").replace("출처: classification", "출처: 항목분류집")

            final_question = f"product_name = ['{pname}'], income = [{income_list[i]}], expense = [{expense_list[i]}]"
            input_data = {"question": final_question, "context": context}

            try:
                out_str = classification_chain_single.invoke(input_data)
                m = re.search(r'\{.*\}', out_str, re.DOTALL)
                llm = json.loads(m.group(0)) if m else {}
                ctype = llm.get("classification_type")

                if ctype == "DEFINITE":
                    r = llm.get("result", {})
                    code = str(r.get("input_code", "")).strip()
                    item_name = code_to_name_map.get(code, "항목명 없음")
                    definite_results.append({
                        "품목명": pname, "입력코드": code, "항목명": item_name,
                        "수입": income_list[i], "지출": expense_list[i],
                        "신뢰도": r.get("confidence","N/A"),
                        "추론 이유": r.get("reason","N/A"), "근거정보": r.get("evidence","N/A")
                    })
                elif ctype == "AMBIGUOUS":
                    cands = llm.get("candidates", [])
                    for c in cands:
                        c["항목명"] = code_to_name_map.get(str(c.get("input_code","")).strip(), "항목명 없음")
                    ambiguous_results.append({
                        "품목명": pname, "수입": income_list[i], "지출": expense_list[i],
                        "모호성 이유": llm.get("reason_for_ambiguity","N/A"),
                        "후보": cands, "근거정보": llm.get("evidence","N/A")
                    })
                else:
                    failed_results.append({"품목명": pname, "수입": income_list[i], "지출": expense_list[i], "실패 이유": f"알 수 없는 타입: {ctype}"})
            except Exception as e:
                failed_results.append({"품목명": pname, "수입": income_list[i], "지출": expense_list[i], "실패 이유": str(e)})

        # ----- DataFrame 생성 및 숫자형으로 강제(⚠️ 제거 핵심) -----
        df_definite = pd.DataFrame(definite_results)
        if not df_definite.empty:
            for col in ["수입", "지출"]:
                df_definite[col] = pd.to_numeric(df_definite[col], errors="coerce").fillna(0).astype(int)

        # 캐시에 저장 (다음 rerun에서 재사용)
        st.session_state["results"] = {
            "df_definite": df_definite,
            "ambiguous_results": ambiguous_results,
            "failed_results": failed_results,
        }
        
        progress.progress(100, "✅ 분류 완료!")

# ======================================================
# 2) 렌더링: results가 있으면 재계산 없이 그대로 표시
#    (체크박스 눌러도 ‘다시 분류’ 안 돌아감)
# ======================================================
results = st.session_state.get("results")
if results is not None:
    df_definite        = results["df_definite"]
    ambiguous_results  = results["ambiguous_results"]
    failed_results     = results["failed_results"]
    
    st.markdown("---")
    st.markdown("## 📊 분류 결과")

    # --- (1) 명확하게 분류된 품목 ---
    if not df_definite.empty:
        st.markdown("### ✅ 명확하게 분류된 품목")
        view_def = df_definite.copy()
        view_def["수입(원)"] = view_def["수입"].apply(fmt_won)
        view_def["지출(원)"] = view_def["지출"].apply(fmt_won)
        view_def = view_def[["품목명", "입력코드", "항목명", "신뢰도", "수입(원)", "지출(원)"]]
        sty = (
            view_def
            .style
            .set_properties(subset=["수입(원)", "지출(원)"], **{"text-align": "right"})
        )
        # st.table은 Styler를 반영해 정렬이 먹음
        st.table(sty)
    
    # --- (2) 입력코드별 요약 보기 (재계산 없이 캐시로부터) ---
    if st.checkbox("입력코드별 요약 보기", key="show_summary"):
        if not df_definite.empty:
            numeric_codes_mask = pd.to_numeric(df_definite['입력코드'], errors='coerce').notna()
            df_summary = df_definite[numeric_codes_mask].copy()
            
            if not df_summary.empty:
                df_summary['입력코드'] = df_summary['입력코드'].astype(float).astype(int)
                df_summary_agg = df_summary.groupby('입력코드').agg(
                    항목명=('항목명', 'first'),
                    수입합계=('수입', 'sum'),
                    지출합계=('지출', 'sum'),
                    해당품목명=('품목명', lambda x: ', '.join(x))
                ).reset_index()
                
                view_sum = df_summary_agg.copy()
                view_sum["수입합계(원)"] = view_sum["수입합계"].apply(fmt_won)
                view_sum["지출합계(원)"] = view_sum["지출합계"].apply(fmt_won)
                view_sum = view_sum[['입력코드', '항목명', '수입합계(원)', '지출합계(원)', '해당품목명']]
                
                sty2 = (
                    view_sum
                    .style
                    .set_properties(subset=["수입합계(원)", "지출합계(원)"], **{"text-align": "right"})
                )
                # st.table은 Styler를 반영해 정렬이 먹음
                st.table(sty2)
            else:
                st.warning("숫자 코드가 있는 항목이 없습니다.")
        else:
            st.warning("명확하게 분류된 품목이 없습니다.")
    
    # --- (3) 명확한 분류에 대한 상세 근거 ---
    if not df_definite.empty:
        with st.expander("🔎 명확한 분류에 대한 상세 근거", expanded=False):
            for row in df_definite.to_dict(orient="records"):
                st.markdown(
                    f"**품목명: {row['품목명']} (선택된 코드: {row['입력코드']}, "
                    f"항목명: {row['항목명']}, 신뢰도: {row['신뢰도']})**"
                )
                if row.get("추론 이유"):
                    st.write(f"**- 추론 이유:** {row['추론 이유']}")
                if row.get("근거정보"):
                    st.write("**- 핵심 근거:**")
                    st.code(row["근거정보"])
                st.markdown("---")
    
    # --- (4) 사용자 검토가 필요한 품목 (컬럼명 한글화) ---
    if ambiguous_results:
        st.markdown("### ⚠️ 사용자 검토가 필요한 품목")
        st.info("아래 품목들은 정보가 부족하여 단일 코드를 확정하지 못했습니다.")
        for result in ambiguous_results:
            with st.expander(f"📌 {result['품목명']} (수입: {fmt_won(result['수입'])}, 지출: {fmt_won(result['지출'])})"):
                st.write(f"**검토 필요 이유:** {result['모호성 이유']}")
                candidates_df = pd.DataFrame(result['후보']).rename(columns={
                    "input_code": "입력코드",
                    "confidence": "신뢰도",
                    "reason": "근거정보",
                })
                # 표시 컬럼 순서 고정
                view_cols = [c for c in ["입력코드", "항목명", "신뢰도", "근거정보"] if c in candidates_df.columns]
                h3 = min(44 * (len(candidates_df) + 1), 400)
                st.dataframe(candidates_df[view_cols], use_container_width=True, height=h3, hide_index=True)
    
    # --- (5) 실패 항목 ---
    if failed_results:
        with st.expander("❌ 처리 실패 항목"):
            df_failed = pd.DataFrame(failed_results)
            h4 = min(44 * (len(df_failed) + 1), 400)
            st.dataframe(df_failed, use_container_width=True, height=h4, hide_index=True)
