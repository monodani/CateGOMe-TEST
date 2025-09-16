# /cate_gome_logic.py (FINAL VERSION - ALL-IN-ONE)

import streamlit as st
import os
import re
import json
import requests
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

# LangChain 및 AI 모델 관련 라이브러리
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import google.generativeai as genai

# --- 설정값 (Configuration) ---
EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-1.5-flash"
NUM_RELATED_TERMS = 3
SIMILARITY_TOP_K = 3


@st.cache_resource
def load_all_data_and_models(openai_api_key: str):
    """
    앱에 필요한 모든 것을 다운로드하고 메모리에 로드하는 단일 함수.
    @st.cache_resource 덕분에 앱 세션 동안 딱 한 번만 실행됩니다.
    """
    # 1. 경로 및 파일 URL 정의
    BASE_DIR = Path.cwd() 
    DATA_FILES = {
        # Core Data
        "cases_faiss": {"url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/vectorstores/cases/cases_index.faiss", "path": BASE_DIR / "vectorstores/cases/cases_index.faiss"},
        "cases_pkl": {"url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/vectorstores/cases/cases_index.pkl", "path": BASE_DIR / "vectorstores/cases/cases_index.pkl"},
        "classification_faiss": {"url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/vectorstores/classification/classification_index.faiss", "path": BASE_DIR / "vectorstores/classification/classification_index.faiss"},
        "classification_pkl": {"url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/vectorstores/classification/classification_index.pkl", "path": BASE_DIR / "vectorstores/classification/classification_index.pkl"},
        "classification_csv": {"url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/data/classification_code.csv", "path": BASE_DIR / "data/classification_code.csv"},
        
        # <<< 이 부분이 누락되었던 이미지 파일 다운로드 목록입니다.
        "logo_main": {"url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/assets/CateGOMe/CateGOMe_kor.png", "path": BASE_DIR / "assets/CateGOMe_kor.png"},
        "emoji_hi": {"url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/assets/emoji/CateGOMe_emoji_hi.png", "path": BASE_DIR / "assets/emoji/CateGOMe_emoji_hi.png"},
        "emoji_categorizing": {"url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/assets/emoji/CateGOMe_emoji_categorying.png", "path": BASE_DIR / "assets/emoji/CateGOMe_emoji_categorying.png"},
        "emoji_sorry": {"url": "https://raw.githubusercontent.com/monodani/CateGOMe-TEST/main/assets/emoji/CateGOMe_emoji_sorry.png", "path": BASE_DIR / "assets/emoji/CateGOMe_emoji_sorry.png"}
    }

    # 2. 파일 다운로드 로직
    for key, info in DATA_FILES.items():
        local_path = info["path"]
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with requests.get(info["url"], stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            except Exception as e:
                # 다운로드 실패 시 앱을 중지시키는 치명적 오류 발생
                st.error(f"필수 파일 다운로드 실패: {local_path}. 앱을 재부팅해주세요. 오류: {e}")
                st.stop()

    # 3. 데이터 및 모델 로딩
    try:
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=openai_api_key)

        vs_cases = FAISS.load_local(
            str(BASE_DIR / "vectorstores/cases"),
            embeddings,
            index_name="cases_index",
            allow_dangerous_deserialization=True,
        )
        vs_classification = FAISS.load_local(
            str(BASE_DIR / "vectorstores/classification"),
            embeddings,
            index_name="classification_index",
            allow_dangerous_deserialization=True,
        )
        vectorstores = {"cases": vs_cases, "classification": vs_classification}

        df = pd.read_csv(DATA_FILES["classification_csv"]["path"], encoding="utf-8")
        df = df.reset_index(drop=False).rename(columns={"index": "_rowid"})

        df["입력코드_str"] = df["입력코드"].astype(str).str.replace(r"\.0$", "", regex=True)
        code_to_name_map = pd.Series(df.항목명.values, index=df.입력코드_str).to_dict()

        # 최신 langchain_openai 시그니처 호환 (model=)
        llm_model = ChatOpenAI(model=LLM_MODEL, temperature=0.1, openai_api_key=openai_api_key)

        # 로드된 모든 객체를 딕셔너리로 반환
        return {
            "embeddings": embeddings,
            "vectorstores": vectorstores,
            "df": df,
            "llm_model": llm_model,
            "code_to_name_map": code_to_name_map,
        }
    except Exception as e:
        st.error(f"데이터 또는 모델 로딩 실패. 앱을 재부팅해주세요. 오류: {e}")
        st.stop()


# --- 헬퍼 함수들 ---
def _short_doc_from_row(row: pd.Series) -> Document:
    source = row.get("출처", "항목분류집")
    source_info = f"출처: {source}\n"
    core_fields_order = [
        col
        for col in ["입력코드", "항목명", "항목분류내용", "처리코드", "포함항목", "제외항목"]
        if col in row.index
    ]
    core_lines = [f"{col}: {str(row[col])}" for col in core_fields_order]
    page = source_info + "\n".join(core_lines)
    meta = row.to_dict()
    return Document(page_content=page, metadata=meta)


def _keyword_search(df: pd.DataFrame, term: str) -> List[Document]:
    if df is None:
        return []
    df_copy = df.copy()
    search_cols = ["항목분류내용", "항목명", "포함항목", "제외항목"]
    for c in search_cols:
        if c in df_copy.columns:
            df_copy[c] = df_copy[c].astype(str)

    # regex=False 추가하여 UserWarning 방지
    mask = (
        df_copy["항목분류내용"].str.contains(term, case=False, na=False, regex=False)
        | df_copy["항목명"].str.contains(term, case=False, na=False, regex=False)
        | df_copy["포함항목"].str.contains(term, case=False, na=False, regex=False)
        | df_copy["제외항목"].str.contains(term, case=False, na=False, regex=False)
    )
    sub = df_copy.loc[mask].drop_duplicates(subset=["_rowid"], keep="first")
    return [_short_doc_from_row(r) for _, r in sub.iterrows()]


def _keyword_search_on_docs(docs: List[Document], term: str) -> List[Document]:
    if not docs:
        return []
    return [doc for doc in docs if term.lower() in doc.page_content.lower()]


def _similarity_topk_for_term(vs: FAISS, embeddings: OpenAIEmbeddings, term: str, k: int) -> List[Document]:
    if vs is None or embeddings is None:
        return []
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 30, "lambda_mult": 0.5})
    return retriever.invoke(term)


def _get_term_info_via_llm(llm: ChatOpenAI, user_query: str, num_related_terms: int) -> List[Dict[str, Any]]:
    # ... (내용 변경 없음)
    prompt = f"""너는 사용자 쿼리에서 'product_name' 리스트에 포함된 모든 품목명을 분석하고, 오탈자를 교정한 뒤 검색에 유용한 정보를 추출하는 전문가 AI이다. 다른 어떤 설명도 없이, 아래 "출력 예시"와 완벽하게 동일한 JSON 형식으로만 응답한다.
## 입력 쿼리 예시:
'''product_name = ['스타벅스조각케익', '전기차 충전', '색지피티'] ...'''
### 너의 JSON 출력 예시:
```json
{{
  "terms": [
    {{
      "original_term": "스타벅스조각케익", "term": "스타벅스 조각케익", "description": "스타벅스 커피 전문점에서 판매하는 조각 형태의 케이크입니다. 커피와 함께 즐기는 대표적인 디저트 메뉴 중 하나입니다.", "related_terms": ["스타벅스", "케이크", "디저트"]
    }},
    {{
      "original_term": "전기차 충전", "term": "전기차 충전", "description": "전기 자동차의 배터리에 전력을 공급하는 행위입니다. 자동차를 운행하기 위한 에너지를 채우는 과정입니다.", "related_terms": ["전기차", "충전소", "연료"]
    }},
    {{
      "original_term": "색지피티", "term": "챗지피티", "description": "OpenAI가 개발한 대화형 인공지능 서비스입니다.", "related_terms": ["생성형AI", "AI", "LLM"]
    }}
  ]
}}
이제 아래 사용자 입력 쿼리를 처리해라. 다른 말은 절대 하지 말고 JSON만 출력해라. 관련 용어는 {num_related_terms}개 추출해라.
사용자 입력 쿼리: {user_query}"""

    try:
        res = llm.invoke(prompt)
        text_content = res.content.strip()
        json_match = re.search(r"\{.*\}", text_content, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No JSON found", text_content, 0)
        data = json.loads(json_match.group(0))
        return data.get("terms", [])
    except Exception as e:
        print(f"ERROR[llm]: Failed to get term info: {e}")
        match = re.search(r"product_name\s*=\s*\[\s*[\"']([^\"']+)[\"']\s*\]", user_query)
        term = match.group(1) if match else "unknown"
        return [{"term": term, "description": "", "related_terms": []}]


# --- 메인 로직 함수 ---
def get_classification_report(image_bytes: bytes, genai_api_key: str, loaded_data: dict) -> str:
    # 로드된 객체들을 인자로 받음
    _llm_model = loaded_data["llm_model"]
    _df = loaded_data["df"]
    _vectorstores = loaded_data["vectorstores"]
    _embeddings = loaded_data["embeddings"]
    _code_to_name_map = loaded_data["code_to_name_map"]

    # 1. 이미지 분석
    # ... (내용 변경 없음)
    try:
        genai.configure(api_key=genai_api_key)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = """가계부 사진에서 표를 인식해서 각 행의 1) 품목명(= '수입종류 및 지출의 품명과 용도' 열), 2) 수입 금액, 3) 지출 금액을 추출하라.

규칙: 금액의 쉼표(,)는 제거하고 정수로. 값이 비어 있으면 0으로. 제목행·체크박스·빈줄은 제외. 반드시 아래 JSON 스키마로만 출력.
JSON 스키마: {"items": [{"name": "품목명", "income": 0, "expense": 0}, ...]}"""

        resp = gemini_model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": image_bytes}],
            generation_config={"response_mime_type": "application/json"},
        )
        data = json.loads(resp.text)
        items = data.get("items", [])
        if not items:
            return "### 이미지 분석 결과\n\n이미지에서 가계부 내역을 찾을 수 없습니다."

        product_name_list, income_list, expense_list = [], [], []
        for item in items:
            name = str(item.get("name", "")).strip()
            if name:
                product_name_list.append(name)
                income_list.append(int(str(item.get("income", 0)).replace(",", "")))
                expense_list.append(int(str(item.get("expense", 0)).replace(",", "")))
    except Exception as e:
        return f"### 이미지 분석 오류\n\n이미지 처리 중 오류가 발생했습니다: {e}"

    # 2. 개별 품목 처리
    prompt_template_single = PromptTemplate.from_template(
        """SYSTEM: 당신은 주어진 데이터를 분석하여 가장 적합한 '입력코드'와 '항목명'을 추론하는, 극도로 꼼꼼하고 규칙을 엄수하는 데이터 분류 AI이며, 당신의 이름은 "카테고미(CateGOMe)"입니다. 당신의 답변은 반드시 지정된 JSON 형식이어야 합니다.

[절대 규칙 1] expense > 0 이면 input_code >= 1000, income > 0 이면 input_code < 1000 입니다.
[절대 규칙 2] '출처: 조사사례집' 정보는 '출처: 항목분류집' 정보보다 항상 우선합니다.
[분류 타입] 90% 이상 확신할 수 있으면 'DEFINITE', 후보가 여러 개이거나 정보가 부족하면 'AMBIGUOUS'로 결정하세요. (예: '네이버' -> 서비스가 다양해 모호함)
[출력 형식] 다른 설명 없이 'DEFINITE' 또는 'AMBIGUOUS' 형식의 JSON으로만 응답하세요.
HUMAN: #Question: {question}\n#Context: {context}\nAnswer:"""
    )

    classification_chain_single = (
        {"question": itemgetter("question"), "context": itemgetter("context")}
        | prompt_template_single
        | _llm_model
        | StrOutputParser()
    )

    definite_results, ambiguous_results, failed_results = [], [], []
    all_docs_from_vs = {name: list(vs.docstore._dict.values()) for name, vs in _vectorstores.items()}

    for i, product_name_original in enumerate(product_name_list):
        q_single = f"product_name = ['{product_name_original}'], income = [{income_list[i]}], expense = [{expense_list[i]}]"

        # 검색 로직
        extracted_terms_info = _get_term_info_via_llm(
            _llm_model, q_single, num_related_terms=NUM_RELATED_TERMS
        )
        search_context_docs = []
        seen_docs_page_content = set()
        for item in extracted_terms_info:
            # ... (이하 검색 로직은 search_classification_codes 함수 내용을 그대로 가져옴. 변경 없음)
            term = item.get("term", "")
            if not term:
                continue
            terms_to_search = [term] + item.get("related_terms", [])
            for search_term in terms_to_search:
                for doc in _keyword_search(_df, search_term):
                    if doc.page_content not in seen_docs_page_content:
                        search_context_docs.append(doc)
                        seen_docs_page_content.add(doc.page_content)
            for vs_name, doc_list in all_docs_from_vs.items():
                for search_term in terms_to_search:
                    for doc in _keyword_search_on_docs(doc_list, search_term):
                        new_doc = Document(
                            page_content=f"출처: {vs_name}\n{doc.page_content}",
                            metadata=doc.metadata,
                        )
                        if new_doc.page_content not in seen_docs_page_content:
                            search_context_docs.append(new_doc)
                            seen_docs_page_content.add(new_doc.page_content)
            description = item.get("description", "")
            if description:
                for vs_name, vs in _vectorstores.items():
                    for doc in _similarity_topk_for_term(
                        vs, _embeddings, description, k=SIMILARITY_TOP_K
                    ):
                        new_doc = Document(
                            page_content=f"출처: {vs_name}\n{doc.page_content}",
                            metadata=doc.metadata,
                        )
                        if new_doc.page_content not in seen_docs_page_content:
                            search_context_docs.append(new_doc)
                            seen_docs_page_content.add(new_doc.page_content)

        try:
            product_name_corrected = extracted_terms_info[0]["term"]
        except (IndexError, KeyError):
            product_name_corrected = product_name_original

        if not search_context_docs:
            failed_results.append(
                {
                    "품목명": product_name_corrected,
                    "수입": income_list[i],
                    "지출": expense_list[i],
                    "실패 이유": "관련 정보 검색 실패",
                }
            )
            continue

        context = (
            "\n\n---\n\n".join([doc.page_content for doc in search_context_docs])
            .replace("출처: cases", "출처: 조사사례집")
            .replace("출처: classification", "출처: 항목분류집")
        )
        final_question = (
            f"product_name = ['{product_name_corrected}'], income = [{income_list[i]}], expense = [{expense_list[i]}]"
        )

        try:
            # ... (이하 분류 및 결과 수집 로직 변경 없음)
            output_json_str = classification_chain_single.invoke(
                {"question": final_question, "context": context}
            )
            json_match = re.search(r"\{.*\}", output_json_str, re.DOTALL)
            if not json_match:
                raise ValueError("LLM 응답에서 JSON 객체를 찾지 못했습니다.")

            llm_result = json.loads(json_match.group(0))

            if llm_result.get("classification_type") == "DEFINITE":
                res = llm_result.get("result", {})
                code = str(res.get("input_code", "N/A")).strip()
                definite_results.append(
                    {
                        "품목명": product_name_corrected,
                        "입력코드": code,
                        "항목명": _code_to_name_map.get(code, "항목명 없음"),
                        "수입": income_list[i],
                        "지출": expense_list[i],
                        "신뢰도": res.get("confidence", "N/A"),
                        "추론 이유": res.get("reason", "N/A"),
                        "근거 정보": res.get("evidence", "N/A"),
                    }
                )
            elif llm_result.get("classification_type") == "AMBIGUOUS":
                candidates = llm_result.get("candidates", [])
                for cand in candidates:
                    cand_code = str(cand.get("input_code", "")).strip()
                    cand["항목명"] = _code_to_name_map.get(cand_code, "항목명 없음")
                ambiguous_results.append(
                    {
                        "품목명": product_name_corrected,
                        "수입": income_list[i],
                        "지출": expense_list[i],
                        "모호성 이유": llm_result.get("reason_for_ambiguity", "N/A"),
                        "후보": candidates,
                        "근거 정보": llm_result.get("evidence", "N/A"),
                    }
                )
            else:
                raise ValueError(f"알 수 없는 classification_type: {llm_result.get('classification_type')}")
        except Exception as e:
            failed_results.append(
                {
                    "품목명": product_name_corrected,
                    "수입": income_list[i],
                    "지출": expense_list[i],
                    "실패 이유": f"LLM 응답 처리 실패: {e}",
                }
            )

    # 3. 보고서 생성
    # ... (내용 변경 없음)
    report = ["## 📊 카테고미 분류 결과 보고서"]
    if definite_results:
        report.append("\n### 1. 명확하게 분류된 품목\n")
        df_definite = pd.DataFrame(definite_results)
        report.append("#### 품목별 분류 결과")
        report.append(df_definite[["품목명", "입력코드", "항목명", "신뢰도", "수입", "지출"]].to_markdown(index=False))
        df_summary = df_definite[pd.to_numeric(df_definite["입력코드"], errors="coerce").notna()].copy()
        if not df_summary.empty:
            df_summary["입력코드"] = df_summary["입력코드"].astype(int)
            df_summary_agg = (
                df_summary.groupby("입력코드")
                .agg(
                    항목명=("항목명", "first"),
                    수입합계=("수입", "sum"),
                    지출합계=("지출", "sum"),
                    해당품목명=("품목명", lambda x: ", ".join(x)),
                )
                .reset_index()
            )
            report.append("\n#### 입력코드별 요약 결과")
            report.append(df_summary_agg.to_markdown(index=False))
    else:
        report.append("\n명확하게 분류된 품목이 없습니다.\n")

    if ambiguous_results:
        report.append("\n\n### 2. 사용자의 검토가 필요한 품목\n")
        report.append("> 아래 품목들은 정보가 부족하여 단일 코드를 확정하지 못했습니다.\n")
        for res in ambiguous_results:
            report.append(
                f"---\n**품목명: {res['품목명']}** (수입: {res['수입']:,}원, 지출: {res['지출']:,}원)"
            )
            report.append(f"**- 검토 필요 이유:** {res['모호성 이유']}")
            if res.get("후보"):
                df_cand = pd.DataFrame(res["후보"]).rename(
                    columns={"input_code": "입력코드", "confidence": "신뢰도", "reason": "이유"}
                )
                report.append(df_cand[["입력코드", "항목명", "신뢰도", "이유"]].to_markdown(index=False))

    if definite_results:
        report.append("\n\n### 3. 명확한 분류에 대한 상세 근거\n")
        for res in definite_results:
            report.append(f"---\n**품목명: {res['품목명']} (선택된 코드: {res['입력코드']})**")
            report.append(f"**- 추론 이유:** {res['추론 이유']}")
            report.append(f"**- 핵심 근거:**\n```\n{res['근거 정보']}\n```")

    if failed_results:
        report.append("\n\n### 4. 처리 실패 항목\n")
        report.append(pd.DataFrame(failed_results).to_markdown(index=False))

    return "\n".join(report)
