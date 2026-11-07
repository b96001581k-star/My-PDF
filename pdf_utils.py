"""PDF 텍스트 추출 유틸리티."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import streamlit as st
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# 페이지당 평균 글자 수가 이 값보다 적으면 스캔본(이미지 PDF)으로 의심한다.
SCANNED_CHARS_PER_PAGE = 20

# 일부 PDF는 공백을 NUL(\x00) 등 제어문자로 추출한다. 줄바꿈/탭은 유지.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACES = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    text = _CONTROL_CHARS.sub(" ", text)
    text = _MULTI_SPACES.sub(" ", text)
    return text.strip()


@dataclass
class PdfResult:
    pages: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p) for p in self.pages)

    @property
    def likely_scanned(self) -> bool:
        if self.error or not self.pages:
            return False
        return self.char_count / self.page_count < SCANNED_CHARS_PER_PAGE


@dataclass
class Document:
    """분석에 쓰는 문서 (파일명 + 페이지별 텍스트)."""

    name: str
    pages: list[str]

    @property
    def char_count(self) -> int:
        return sum(len(p) for p in self.pages)


@dataclass
class Chunk:
    source: str  # 파일명
    start_page: int
    end_page: int
    text: str  # 출처 태그가 붙은 본문

    @property
    def label(self) -> str:
        if self.start_page == self.end_page:
            return f"{self.source} p.{self.start_page}"
        return f"{self.source} p.{self.start_page}-{self.end_page}"


def tag_page(name: str, page_no: int, text: str) -> str:
    return f"[{name} p.{page_no}]\n{text}"


def document_text(doc: Document) -> str:
    """문서 전체를 페이지 출처 태그와 함께 하나의 문자열로 만든다."""
    return "\n\n".join(tag_page(doc.name, i, t) for i, t in enumerate(doc.pages, 1) if t)


def chunk_text(doc: Document, max_chars: int) -> list[Chunk]:
    """페이지 경계를 최대한 지키며 문서를 max_chars 이하 청크로 나눈다.

    한 페이지가 max_chars보다 길면 그 페이지만 글자 수 기준으로 자른다.
    """
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    start = 0
    last_page = 0  # 버퍼에 마지막으로 넣은 페이지 (빈 페이지는 건너뛰므로 page_no-1과 다를 수 있음)

    def flush(end_page: int) -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append(Chunk(doc.name, start, end_page, "\n\n".join(buf)))
        buf, buf_len = [], 0

    for page_no, text in enumerate(doc.pages, 1):
        if not text:
            continue
        piece = tag_page(doc.name, page_no, text)
        if len(piece) > max_chars:
            flush(last_page)
            body_max = max_chars - len(tag_page(doc.name, page_no, ""))
            for i in range(0, len(text), body_max):
                chunks.append(
                    Chunk(doc.name, page_no, page_no, tag_page(doc.name, page_no, text[i : i + body_max]))
                )
            continue
        if buf_len + len(piece) + 2 > max_chars:
            flush(last_page)
        if not buf:
            start = page_no
        buf.append(piece)
        buf_len += len(piece) + 2
        last_page = page_no
    flush(last_page)
    return chunks


@st.cache_data(show_spinner=False)
def extract_pdf(file_bytes: bytes) -> PdfResult:
    """PDF 바이트에서 페이지별 텍스트를 추출한다.

    UploadedFile 대신 bytes를 받아 캐시 키가 파일 내용 기준으로 안정적이게 한다.
    오류는 예외 대신 PdfResult.error에 담아 반환한다.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    return PdfResult(error="암호가 걸린 PDF라서 읽을 수 없습니다.")
            except Exception:
                return PdfResult(error="암호가 걸린 PDF라서 읽을 수 없습니다.")

        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(clean_text(page.extract_text() or ""))
            except Exception:
                pages.append("")  # 특정 페이지만 깨진 경우 해당 페이지는 빈 텍스트로 처리
        if not pages:
            return PdfResult(error="페이지가 없는 PDF입니다.")
        return PdfResult(pages=pages)
    except PdfReadError:
        return PdfResult(error="PDF 파일이 손상되었거나 올바른 PDF 형식이 아닙니다.")
    except Exception as e:  # pypdf가 던지는 기타 예외
        return PdfResult(error=f"PDF를 읽는 중 오류가 발생했습니다: {type(e).__name__}")
