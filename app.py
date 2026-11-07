"""PDF 분석기 - Streamlit 메인 앱."""

import hashlib
from datetime import datetime, timedelta, timezone

import streamlit as st

from llm import (
    ANALYSIS_PROMPTS,
    COMPARE_TYPE,
    CUSTOM_PROMPT_MAX_CHARS,
    CUSTOM_TYPE,
    LLMError,
    analyze,
    answer_question,
    get_client,
    get_config,
    get_model,
    is_mock,
)
from pdf_utils import Document, extract_pdf

st.set_page_config(page_title="PDF 분석기", page_icon="📄", layout="wide")

# Streamlit Cloud 서버는 UTC라서 한국 시간으로 고정한다 (Windows는 zoneinfo DB가 없어 고정 오프셋 사용).
KST = timezone(timedelta(hours=9), "KST")

TAB_RESULT = "📊 분석 결과"
TAB_QA = "💬 Q&A"
TAB_SOURCE = "📄 원문"

ANALYSIS_HELP = {
    "요약": "문서 전체를 3~5문단으로 요약합니다.",
    "핵심 포인트": "중요한 내용을 bullet 목록으로 정리합니다.",
    "키워드 추출": "핵심 키워드와 의미를 표로 정리합니다.",
    "문서 비교": "여러 문서의 공통점과 차이점을 비교합니다.",
    "사용자 정의": "직접 입력한 지시문대로 분석합니다.",
}
EXAMPLE_QUESTIONS = [
    "이 문서의 핵심 내용은 무엇인가요?",
    "주요 수치와 날짜를 정리해 주세요.",
    "결론이나 시사점은 무엇인가요?",
]

# ── 세션 상태 ─────────────────────────────────────────────
for key, default in {
    "files_sig": None,
    "messages": [],  # Q&A 대화 [{role, content}]
    "analysis": None,  # 마지막 분석 결과 {type, files, text, at}
    "analysis_error": None,
    "run_requested": False,
    "pending_question": None,
    "tabs_nonce": 0,
}.items():
    st.session_state.setdefault(key, default)


def request_run() -> None:
    st.session_state.run_requested = True
    # 탭 key를 바꿔 새 탭 위젯으로 그리게 하면 default(분석 결과) 탭이 선택된다.
    st.session_state.tabs_nonce += 1


def ask_example(q: str) -> None:
    st.session_state.pending_question = q


def reset_conversation() -> None:
    st.session_state.messages = []


# 사이드바 영역을 순서대로 미리 잡아 두고, 내용은 필요한 시점에 채운다.
sb_settings = st.sidebar.container()
sb_options = st.sidebar.container()
sb_footer = st.sidebar.container()

# ── 사이드바: API 설정 ────────────────────────────────────
with sb_settings:
    st.header("⚙️ 설정")
    api_key = get_config("OPENAI_API_KEY")
    if is_mock():
        st.badge("MOCK 모드", icon="🧪", color="blue")
        st.caption("OpenAI를 호출하지 않고 테스트 응답을 사용합니다.")
    elif api_key:
        st.badge("API 연결됨", icon="✅", color="green")
    else:
        st.badge("API 키 미설정", icon="⚠️", color="red")
        api_key = st.text_input(
            "OpenAI API 키",
            type="password",
            help=".env 또는 Streamlit Secrets에 OPENAI_API_KEY를 설정하면 입력하지 않아도 됩니다.",
        )
    st.caption(f"모델: `{get_model()}`")

llm_ready = is_mock() or bool(api_key)

with sb_footer:
    st.divider()
    st.button("🗑️ 대화 초기화", on_click=reset_conversation, width="stretch")
    with st.expander("사용 방법"):
        st.markdown(
            "1. PDF 파일을 업로드합니다 (여러 개 가능).\n"
            "2. 분석 유형을 고르고 **분석 실행**을 누릅니다.\n"
            "3. **Q&A** 탭에서 문서 내용을 질문할 수 있습니다.\n\n"
            "업로드한 파일은 서버에 저장되지 않습니다. 스캔한 이미지 PDF는 텍스트를 읽을 수 없습니다."
        )

# ── 헤더 & 업로드 ─────────────────────────────────────────
st.title("📄 PDF 분석기")
st.caption(f"PDF를 올리면 AI(`{get_model()}`)가 요약·분석하고 질문에 답합니다.")

uploaded_files = st.file_uploader(
    "PDF 파일 업로드",
    type=["pdf"],
    accept_multiple_files=True,
    help="여러 파일을 한 번에 올릴 수 있습니다. 파일당 최대 200MB.",
)

if not uploaded_files:
    st.session_state.files_sig = None
    with sb_options:
        st.caption("PDF를 올리면 분석 옵션이 나타납니다.")
    st.write("")
    steps = [
        ("1️⃣", "PDF 업로드", "분석할 PDF를 위 영역에 끌어다 놓으세요. 여러 개도 가능합니다."),
        ("2️⃣", "분석 실행", "요약, 핵심 포인트, 키워드, 문서 비교 중에서 고르거나 직접 지시할 수 있습니다."),
        ("3️⃣", "질문하기", "Q&A 탭에서 문서 내용을 물어보면 근거 페이지와 함께 답합니다."),
    ]
    for col, (icon, title, desc) in zip(st.columns(3), steps):
        with col.container(border=True):
            st.markdown(f"### {icon} {title}")
            st.write(desc)
    if not llm_ready:
        st.warning("AI 기능을 쓰려면 사이드바에 OpenAI API 키를 입력하세요.")
    st.stop()

# ── 텍스트 추출 ───────────────────────────────────────────
file_bytes: list[tuple[str, bytes]] = []
seen_names: dict[str, int] = {}
for f in uploaded_files:
    # 이름이 같은 파일이 여러 개면 출처 표시와 원문 탭 선택이 섞이므로 (2), (3)을 붙인다.
    n = seen_names[f.name] = seen_names.get(f.name, 0) + 1
    name = f.name if n == 1 else f"{f.name.removesuffix('.pdf')} ({n}).pdf"
    file_bytes.append((name, f.getvalue()))
files_sig = tuple(sorted((name, hashlib.md5(data).hexdigest()) for name, data in file_bytes))
if st.session_state.files_sig != files_sig:
    # 업로드 파일 구성이 바뀌면 이전 대화와 분석 결과를 지운다.
    st.session_state.files_sig = files_sig
    st.session_state.messages = []
    st.session_state.analysis = None
    st.session_state.analysis_error = None

docs: list[Document] = []
rows = []
problems: list[tuple[str, str, str]] = []  # (level, name, message)
for name, data in file_bytes:
    r = extract_pdf(data)
    if r.error:
        status = "❌ 오류"
        problems.append(("error", name, r.error))
    elif r.char_count == 0:
        status = "⚠️ 텍스트 없음"
        problems.append(("warning", name, "텍스트를 추출하지 못했습니다. 스캔한 이미지 PDF로 보이며 분석에서 제외됩니다."))
    elif r.likely_scanned:
        status = "⚠️ 스캔본 의심"
        problems.append(("warning", name, "추출된 텍스트가 매우 적습니다. 스캔한 이미지 PDF일 수 있습니다."))
    else:
        status = "✅ 정상"
    if not r.error and r.char_count > 0:
        docs.append(Document(name=name, pages=r.pages))
    rows.append({"파일명": name, "페이지": r.page_count, "글자 수": r.char_count, "상태": status})

# 표 대신 목록 형태: 좁은 화면에서도 긴 파일명 때문에 상태가 잘리지 않는다.
with st.container(border=True):
    st.markdown(f"**업로드한 파일 {len(rows)}개** · 분석 가능 {len(docs)}개")
    for row in rows:
        st.markdown(
            f"{row['상태'].split()[0]} **{row['파일명']}**  \n"
            f":gray[{row['페이지']}쪽 · {row['글자 수']:,}자 · {row['상태'].split(maxsplit=1)[1]}]"
        )
for level, name, msg in problems:
    (st.error if level == "error" else st.warning)(f"**{name}**: {msg}")

if not docs:
    with sb_options:
        st.caption("분석할 수 있는 문서가 없습니다.")
    st.error("분석할 수 있는 텍스트가 있는 PDF가 없습니다. 다른 파일을 올려 주세요.")
    st.stop()

# ── 사이드바: 분석 옵션 (문서 수에 따라 선택지가 달라져 추출 뒤에 그린다) ──
with sb_options:
    st.divider()
    st.subheader("🔍 분석 옵션")
    type_options = [t for t in ANALYSIS_PROMPTS if t != COMPARE_TYPE or len(docs) >= 2]
    analysis_type = st.radio(
        "분석 유형",
        type_options,
        captions=[ANALYSIS_HELP[t] for t in type_options],
    )
    if len(docs) < 2:
        st.caption("💡 문서 비교는 텍스트가 있는 PDF를 2개 이상 올리면 선택할 수 있습니다.")
    custom_prompt = ""
    if analysis_type == CUSTOM_TYPE:
        custom_prompt = st.text_area(
            "분석 지시문",
            placeholder="예: 이 문서에서 일정과 마감일만 표로 정리해 주세요.",
            max_chars=CUSTOM_PROMPT_MAX_CHARS,
        )

# ── 분석 실행 버튼 ────────────────────────────────────────
running = st.session_state.run_requested
need_custom = analysis_type == CUSTOM_TYPE and not custom_prompt.strip()
if not llm_ready:
    disabled_reason = "API 키를 먼저 설정하세요."
elif need_custom:
    disabled_reason = "사이드바에 분석 지시문을 입력하세요."
else:
    disabled_reason = None

st.button(
    "분석 중..." if running else f"✨ '{analysis_type}' 분석 실행",
    type="primary",
    on_click=request_run,
    disabled=running or disabled_reason is not None,
    help=disabled_reason,
)
if disabled_reason and not running:
    st.caption(f"ℹ️ {disabled_reason}")

# on_change는 기본값("ignore")으로 둔다. "rerun"이면 분석 스트리밍 도중 탭을 누를 때
# 스크립트가 다시 실행되어 진행 중인 분석이 버려진다.
tab_result, tab_qa, tab_source = st.tabs(
    [TAB_RESULT, TAB_QA, TAB_SOURCE], key=f"main_tabs_{st.session_state.tabs_nonce}", default=TAB_RESULT
)

# ── 탭: 분석 결과 ─────────────────────────────────────────
with tab_result:
    if running:
        # 실행 플래그를 먼저 내려서, 도중에 화면이 다시 그려져도 분석이 반복되지 않게 한다.
        st.session_state.run_requested = False
        st.session_state.analysis_error = None
        st.caption(f"**{analysis_type}** · 대상: {', '.join(d.name for d in docs)}")
        status = st.status("문서를 준비하는 중...", expanded=True)

        def on_progress(ratio: float, msg: str) -> None:
            status.update(label=msg)
            status.write(f"{msg}")

        try:
            stream = analyze(get_client(api_key), docs, analysis_type, custom_prompt, on_progress)
            status.update(label="AI가 결과를 작성하는 중...")
            with st.container(border=True):
                text = st.write_stream(stream)
            st.session_state.analysis = {
                "type": analysis_type,
                "files": [d.name for d in docs],
                "text": text if isinstance(text, str) else "".join(map(str, text)),
                "at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
            }
        except LLMError as e:
            st.session_state.analysis_error = str(e)
        st.rerun()  # 버튼 활성화 상태와 결과 화면을 정리해서 다시 그린다

    if st.session_state.analysis_error:
        st.error(st.session_state.analysis_error)

    result = st.session_state.analysis
    if result:
        st.caption(f"**{result['type']}** · {', '.join(result['files'])} · {result['at']}")
        with st.container(border=True):
            st.markdown(result["text"])
        md = (
            f"# PDF 분석 결과: {result['type']}\n\n"
            f"- 분석 일시: {result['at']}\n"
            f"- 대상 파일: {', '.join(result['files'])}\n"
            f"- 모델: {get_model()}\n\n---\n\n{result['text']}\n"
        )
        st.download_button(
            "⬇️ 결과 다운로드 (.md)",
            data=md.encode("utf-8"),
            file_name=f"pdf_analysis_{result['type']}.md",
            mime="text/markdown",
        )
    elif not st.session_state.analysis_error:
        st.info(
            "사이드바에서 분석 유형을 고르고 위의 **분석 실행** 버튼을 누르세요. "
            "모바일에서는 왼쪽 위 **»** 버튼으로 사이드바를 엽니다."
        )

# ── 탭: Q&A ──────────────────────────────────────────────
with tab_qa:
    if not llm_ready:
        st.info("Q&A를 사용하려면 사이드바에 OpenAI API 키를 입력하세요.")
    else:
        st.caption("업로드한 문서 내용만 근거로 답하며, 근거가 된 파일명과 페이지를 함께 표시합니다.")
        if not st.session_state.messages:
            st.write("**예시 질문**")
            with st.container(horizontal=True):
                for q in EXAMPLE_QUESTIONS:
                    st.button(q, on_click=ask_example, args=(q,))

        chat_box = st.container(height=460, border=True)
        with chat_box:
            if not st.session_state.messages:
                st.caption("아직 대화가 없습니다. 아래에 질문을 입력하세요.")
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        question = st.chat_input("문서에 대해 질문하세요")
        if not question and st.session_state.pending_question:
            question = st.session_state.pending_question
        st.session_state.pending_question = None

        if question:
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    try:
                        stream = answer_question(get_client(api_key), docs, st.session_state.messages, question)
                        answer = st.write_stream(stream)
                        answer = answer if isinstance(answer, str) else "".join(map(str, answer))
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except LLMError as e:
                        st.error(str(e))

# ── 탭: 원문 ─────────────────────────────────────────────
with tab_source:
    names = [d.name for d in docs]
    left, right = st.columns([3, 1])
    doc_name = left.selectbox("파일", names)
    doc = docs[names.index(doc_name)]
    page_no = right.number_input("페이지", min_value=1, max_value=len(doc.pages), value=1, step=1)
    st.caption(f"p.{page_no} / {len(doc.pages)} · {len(doc.pages[page_no - 1]):,}자")
    st.code(doc.pages[page_no - 1] or "(이 페이지에는 추출된 텍스트가 없습니다)", language=None, wrap_lines=True, height=480)
