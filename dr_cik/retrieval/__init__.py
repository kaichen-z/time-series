"""Chunking, the Retriever interface, and the factory agents use to build one."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from ..models import AgentDocument

RETRIEVERS = ("bm25", "dense")


@dataclass(frozen=True)
class Chunk:
    """One retrievable span of text from a document."""

    document_id: str
    chunk_id: str
    text: str


class Retriever(Protocol):
    """What an agent needs from a retriever: ranked (chunk, score) pairs for a query."""

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]: ...


def chunk_document(document: AgentDocument, max_chars: int = 1400, overlap: int = 150) -> list[Chunk]:
    """Split a document into overlapping chunks, or return it whole if short enough."""
    text = document.text
    if len(text) <= max_chars:
        return [Chunk(document_id=document.document_id, chunk_id=f"{document.document_id}:0", text=text)]
    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = max(1, max_chars - overlap)
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(Chunk(document_id=document.document_id, chunk_id=f"{document.document_id}:{index}", text=text[start:end]))
        if end >= len(text):
            break
        start += step
        index += 1
    return chunks


def chunk_corpus(documents: Sequence[AgentDocument], max_chars: int = 1400, overlap: int = 150) -> list[Chunk]:
    """Chunk every document in a corpus into one flat list."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars=max_chars, overlap=overlap))
    return chunks


def build_index(documents: Sequence[AgentDocument], retriever: str = "bm25", **kwargs) -> Retriever:
    """Chunk a corpus and build the requested retriever over it: 'bm25' (lexical, dependency-free) or 'dense' (see retrieval/dense.py)."""
    if retriever not in RETRIEVERS:
        raise ValueError(f"Unknown retriever {retriever!r}, expected one of {RETRIEVERS}")
    chunks = chunk_corpus(documents)
    if retriever == "dense":
        from .dense import DenseIndex

        return DenseIndex(chunks, **kwargs)
    from .bm25 import BM25Index

    return BM25Index(chunks, **kwargs)
