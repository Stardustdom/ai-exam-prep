# app/ingestion/chunking.py
#
# Token-based sliding-window chunking. We never send a whole document to an
# LLM in one shot (spec section 19) — everything downstream operates on
# these bounded chunks, retrieved by relevance instead.
from typing import List, NamedTuple
import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


class Chunk(NamedTuple):
    index: int
    content: str
    token_count: int


def chunk_text(text: str, chunk_tokens: int = 500, overlap_tokens: int = 50) -> List[Chunk]:
    """
    Splits text into overlapping token windows. Overlap keeps concepts that
    straddle a chunk boundary retrievable from either neighboring chunk.
    """
    if not text or not text.strip():
        return []

    tokens = _ENCODING.encode(text)
    if not tokens:
        return []

    chunks: List[Chunk] = []
    start = 0
    index = 0
    step = max(chunk_tokens - overlap_tokens, 1)

    while start < len(tokens):
        window = tokens[start:start + chunk_tokens]
        content = _ENCODING.decode(window).strip()
        if content:
            chunks.append(Chunk(index=index, content=content, token_count=len(window)))
            index += 1
        start += step

    return chunks
