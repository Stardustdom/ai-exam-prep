# app/ingestion/parsers.py
#
# Extensible-by-design (spec section 5): add a new file type by adding an
# entry to _PARSERS and a function with this signature — nothing else in the
# pipeline needs to change.
from typing import Tuple, Callable, Dict
from pathlib import Path
import io


def _parse_pdf(content: bytes) -> Tuple[str, int]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages), len(reader.pages)


def _parse_docx(content: bytes) -> Tuple[str, int]:
    import docx
    document = docx.Document(io.BytesIO(content))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs), 1


def _parse_txt(content: bytes) -> Tuple[str, int]:
    return content.decode("utf-8", errors="replace"), 1


_PARSERS: Dict[str, Callable[[bytes], Tuple[str, int]]] = {
    ".pdf": _parse_pdf,
    ".doc": _parse_docx,
    ".docx": _parse_docx,
    ".txt": _parse_txt,
}


class UnsupportedFileType(Exception):
    pass


def parse_document(file_path: str) -> Tuple[str, int]:
    """
    Parse a document into (plain_text, page_count). Raises UnsupportedFileType
    for extensions with no registered parser.
    """
    ext = Path(file_path).suffix.lower()
    parser = _PARSERS.get(ext)
    if not parser:
        raise UnsupportedFileType(f"No parser registered for extension '{ext}'")

    with open(file_path, "rb") as f:
        content = f.read()
    text, page_count = parser(content)

    text = clean_text(text)
    return text, page_count


def clean_text(text: str) -> str:
    """Normalize whitespace and drop null bytes/control characters that break downstream storage"""
    import re
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
