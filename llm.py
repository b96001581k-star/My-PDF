"""설정 로딩, OpenAI 호출, 분석/Q&A 로직."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Iterator

import streamlit as st
from dotenv import load_dotenv

from pdf_utils import Chunk, Document, chunk_text, document_text

load_dotenv()

DEFAULT_MODEL = "gpt-6.1-terra"

# 한 번의 요청에 넣는 문서 본문 최대 글자 수. 넘으면 map-reduce로 처리한다.
MAX_CONTEXT_CHARS = 60_000
# map 단계 청크 크기
MAP_CHUNK_CHARS = 15_000
# Q&A에서 컨텍스트로 넣을 최대 글자 수와 청크 크기
QA_CONTEXT_CHARS = 30_000
QA_CHUNK_CHARS = 3_000
# Q&A에서 모델에 다시 보내는 이전 대화 메시지 수
QA_HISTORY_MESSAGES = 6
CUSTOM_PROMPT_MAX_CHARS = 2_000

COMMON_RULES = (
    "당신은 문서 분석 전문가입니다. 반드시 한국어로, 마크다운 형식으로 답하세요. "
    "문서 본문에는 [파일명 p.페이지] 형식의 출처 태그가 붙어 있습니다. "
    "중요한 내용에는 (파일명 p.N) 형식으로 출처를 표시하세요. "
    "문서에 없는 내용을 지어내지 마세요."
)

ANALYSIS_PROMPTS: dict[str, str] = {
    "요약": "문서 전체 내용을 3~5개 문단으로 요약하세요. 문서가 여러 개면 문서별로 소제목을 붙여 요약하세요.",
    "핵심 포인트": "문서의 핵심 내용을 중요도 순으로 7~12개 bullet 목록으로 정리하세요. 각 항목은 한두 문장으로 쓰세요.",
    "키워드 추출": (
        "문서의 핵심 키워드 10개 내외를 뽑아 표로 정리하세요. "
        "열은 '키워드 | 설명 | 출처'로 하고, 설명은 문서에서 그 키워드가 어떤 의미로 쓰였는지 한 문장으로 쓰세요."
    ),
    "문서 비교": (
        "업로드된 문서들을 비교하세요. '## 문서별 개요', '## 공통점', '## 차이점', '## 종합 의견' 순서로 작성하고, "
        "차이점은 비교 항목을 행으로 하는 표로 정리하세요."
    ),
    "사용자 정의": "",  # 사용자가 입력한 지시문을 사용
}
COMPARE_TYPE = "문서 비교"
CUSTOM_TYPE = "사용자 정의"

MAP_PROMPT = (
    "아래는 긴 문서의 일부입니다. 이후 종합 분석에 쓸 수 있도록 이 부분의 핵심 내용, 주요 수치, 결론을 "
    "한국어 bullet로 빠짐없이 정리하세요. 각 bullet 끝에 원문의 출처 태그 (파일명 p.N)를 그대로 남기세요."
)

QA_PROMPT = (
    "당신은 업로드된 문서를 근거로 질문에 답하는 도우미입니다. 반드시 한국어로 답하세요. "
    "아래 제공된 문서 내용만 근거로 답하고, 답의 근거가 된 위치를 (파일명 p.N) 형식으로 표시하세요. "
    "문서에서 답을 찾을 수 없으면 추측하지 말고 '문서에서 해당 내용을 찾을 수 없습니다'라고 답하세요."
)


# ── 설정 ─────────────────────────────────────────────────


def get_config(key: str, default: str | None = None) -> str | None:
    """st.secrets → 환경변수(.env) 순으로 설정값을 읽는다. 공백뿐인 값은 없는 것으로 본다."""
    try:
        if key in st.secrets:
            value = str(st.secrets[key]).strip()
            if value:
                return value
    except Exception:
        pass  # 로컬에 secrets.toml이 없으면 st.secrets 접근 시 예외가 난다
    value = (os.getenv(key) or "").strip()
    return value or default


def get_model() -> str:
    return get_config("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


def is_mock() -> bool:
    return (get_config("MOCK_LLM", "0") or "0").strip() == "1"


def get_client(api_key: str):
    """OpenAI 클라이언트를 만든다. MOCK 모드에서는 None."""
    if is_mock():
        return None
    from openai import OpenAI

    return OpenAI(api_key=api_key, max_retries=2, timeout=120)


# ── 오류 처리 ─────────────────────────────────────────────


class LLMError(Exception):
    """사용자에게 그대로 보여줄 한국어 메시지를 담은 오류."""


def _to_llm_error(e: Exception) -> LLMError:
    import openai

    if isinstance(e, openai.AuthenticationError):
        return LLMError("API 키가 올바르지 않습니다. OPENAI_API_KEY 값을 확인하세요.")
    if isinstance(e, openai.PermissionDeniedError):
        return LLMError("이 API 키로는 해당 모델을 사용할 권한이 없습니다.")
    if isinstance(e, openai.NotFoundError):
        return LLMError(f"모델 '{get_model()}'을(를) 찾을 수 없습니다. OPENAI_MODEL 값을 확인하세요.")
    if isinstance(e, openai.RateLimitError):
        return LLMError("요청 한도를 초과했거나 크레딧이 부족합니다. 잠시 후 다시 시도하거나 결제 상태를 확인하세요.")
    if isinstance(e, openai.APITimeoutError):
        return LLMError("OpenAI 응답 시간이 초과되었습니다. 잠시 후 다시 시도하세요.")
    if isinstance(e, openai.APIConnectionError):
        return LLMError("OpenAI 서버에 연결할 수 없습니다. 네트워크 연결을 확인하세요.")
    if isinstance(e, openai.BadRequestError):
        return LLMError("요청이 거부되었습니다. 문서가 너무 길거나 요청 형식에 문제가 있을 수 있습니다.")
    if isinstance(e, openai.APIStatusError):
        return LLMError(f"OpenAI API 오류가 발생했습니다 (HTTP {e.status_code}). 잠시 후 다시 시도하세요.")
    return LLMError(f"AI 호출 중 알 수 없는 오류가 발생했습니다: {type(e).__name__}")


# ── 저수준 호출 ───────────────────────────────────────────


def _mock_text(instructions: str, user_input: str) -> str:
    head = user_input[:80].replace("\n", " ")
    return (
        "**[MOCK 응답]** OpenAI를 호출하지 않은 테스트용 응답입니다.\n\n"
        f"- 지시문 길이: {len(instructions)}자\n"
        f"- 입력 길이: {len(user_input)}자\n"
        f"- 입력 앞부분: {head}"
    )


def mock_stream(text: str, delay: float | None = None) -> Iterator[str]:
    """MOCK 모드용 가짜 스트리밍 응답. MOCK_DELAY(초/단어)로 속도를 늦출 수 있다 (테스트용)."""
    if delay is None:
        try:
            delay = float(get_config("MOCK_DELAY", "0.005") or 0.005)
        except ValueError:
            delay = 0.005
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


def _complete(client, instructions: str, user_input: str) -> str:
    """스트리밍 없이 전체 응답 텍스트를 받는다 (map 단계용)."""
    if client is None:
        return _mock_text(instructions, user_input)
    try:
        resp = client.responses.create(model=get_model(), instructions=instructions, input=user_input)
        return resp.output_text
    except Exception as e:
        raise _to_llm_error(e) from e


def _stream(client, instructions: str, user_input: str | list) -> Iterator[str]:
    """응답을 텍스트 조각 단위로 스트리밍한다. 오류는 LLMError로 바꿔 던진다."""
    if client is None:
        text = user_input if isinstance(user_input, str) else str(user_input[-1].get("content", ""))
        yield from mock_stream(_mock_text(instructions, text))
        return
    try:
        stream = client.responses.create(
            model=get_model(), instructions=instructions, input=user_input, stream=True
        )
        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta
            elif event.type in ("response.failed", "error"):
                raise LLMError("모델이 응답을 생성하지 못했습니다. 잠시 후 다시 시도하세요.")
    except LLMError:
        raise
    except Exception as e:
        raise _to_llm_error(e) from e


# ── 분석 ─────────────────────────────────────────────────

ProgressFn = Callable[[float, str], None]


def build_context(client, docs: list[Document], on_progress: ProgressFn | None = None) -> str:
    """분석용 문서 컨텍스트를 만든다.

    전체 길이가 한도 이하면 원문을 그대로 쓰고, 넘으면 청크별 부분 요약(map)을 만들어 이어 붙인다.
    """
    full = "\n\n".join(f"# 문서: {d.name}\n\n{document_text(d)}" for d in docs)
    if len(full) <= MAX_CONTEXT_CHARS:
        return full

    chunks: list[Chunk] = [c for d in docs for c in chunk_text(d, MAP_CHUNK_CHARS)]
    notes: dict[str, list[str]] = {d.name: [] for d in docs}
    for i, chunk in enumerate(chunks, 1):
        if on_progress:
            on_progress((i - 1) / len(chunks), f"긴 문서 부분 요약 중 ({i}/{len(chunks)}): {chunk.label}")
        notes[chunk.source].append(_complete(client, MAP_PROMPT, chunk.text))
    if on_progress:
        on_progress(1.0, "부분 요약 완료, 종합 분석 중")

    return "\n\n".join(
        f"# 문서: {name} (긴 문서라 부분 요약본으로 제공됨)\n\n" + "\n\n".join(parts)
        for name, parts in notes.items()
    )


def analyze(
    client,
    docs: list[Document],
    analysis_type: str,
    custom_prompt: str = "",
    on_progress: ProgressFn | None = None,
) -> Iterator[str]:
    """선택한 분석 유형으로 문서를 분석해 스트리밍한다."""
    if analysis_type == CUSTOM_TYPE:
        task = custom_prompt.strip()[:CUSTOM_PROMPT_MAX_CHARS]
        if not task:
            raise LLMError("사용자 정의 지시문을 입력하세요.")
    else:
        task = ANALYSIS_PROMPTS[analysis_type]
    if analysis_type == COMPARE_TYPE and len(docs) < 2:
        raise LLMError("문서 비교는 텍스트가 있는 PDF가 2개 이상 필요합니다.")

    context = build_context(client, docs, on_progress)
    instructions = f"{COMMON_RULES}\n\n[작업]\n{task}"
    return _stream(client, instructions, context)


# ── Q&A ─────────────────────────────────────────────────

_WORD = re.compile(r"[0-9A-Za-z가-힣]+")


def _bigrams(text: str) -> set[str]:
    grams: set[str] = set()
    for word in _WORD.findall(text.lower()):
        if len(word) == 1:
            continue
        grams.update(word[i : i + 2] for i in range(len(word) - 1))
    return grams


def select_chunks(docs: list[Document], question: str, budget: int = QA_CONTEXT_CHARS) -> list[Chunk]:
    """질문과 겹치는 글자 bigram이 많은 청크를 골라 budget 안에서 반환한다.

    한국어는 조사가 붙어 단어 일치가 잘 안 되므로 단어 대신 2글자 조각으로 비교한다.
    문서 전체가 budget 이하면 전부 반환한다.
    """
    chunks = [c for d in docs for c in chunk_text(d, QA_CHUNK_CHARS)]
    if sum(len(c.text) for c in chunks) <= budget:
        return chunks

    q = _bigrams(question)
    scored = sorted(
        enumerate(chunks), key=lambda ic: (len(q & _bigrams(ic[1].text)), -ic[0]), reverse=True
    )
    picked: list[tuple[int, Chunk]] = []
    used = 0
    for idx, chunk in scored:
        if used + len(chunk.text) > budget:
            continue
        picked.append((idx, chunk))
        used += len(chunk.text)
    picked.sort()  # 원래 문서 순서로 정렬
    return [c for _, c in picked]


def answer_question(client, docs: list[Document], history: list[dict], question: str) -> Iterator[str]:
    """문서 기반으로 질문에 답한다. history는 이전 대화 [{role, content}, ...]."""
    chunks = select_chunks(docs, question)
    context = "\n\n".join(c.text for c in chunks)
    instructions = f"{QA_PROMPT}\n\n[문서 내용]\n{context}"
    messages = [{"role": m["role"], "content": m["content"]} for m in history[-QA_HISTORY_MESSAGES:]]
    messages.append({"role": "user", "content": question})
    return _stream(client, instructions, messages)
