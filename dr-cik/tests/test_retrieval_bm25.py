"""BM25 ranking and chunking sanity checks against a small toy corpus."""

from __future__ import annotations

from dr_cik.models import AgentDocument
from dr_cik.retrieval import BM25Index, build_index, chunk_document, tokenize


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("Solar Irradiance, 2022!") == ["solar", "irradiance", "2022"]


def test_bm25_ranks_the_matching_document_first() -> None:
    documents = (
        AgentDocument("doc_a", "Solar irradiance measurements at the research annex."),
        AgentDocument("doc_b", "Quarterly sales figures for the cosmetic lab."),
        AgentDocument("doc_c", "CPU usage patterns for the technical site."),
    )
    index = build_index(documents)
    results = index.search("solar irradiance annex", top_k=2)
    assert results[0][0].document_id == "doc_a"
    assert results[0][1] > 0


def test_bm25_returns_empty_for_empty_corpus() -> None:
    index = BM25Index([])
    assert index.search("anything") == []


def test_chunk_document_splits_long_text_with_overlap() -> None:
    document = AgentDocument("doc_long", "x" * 3000)
    chunks = chunk_document(document, max_chars=1000, overlap=100)
    assert len(chunks) > 1
    assert all(chunk.document_id == "doc_long" for chunk in chunks)
    assert all(len(chunk.text) <= 1000 for chunk in chunks)


def test_chunk_document_keeps_short_text_whole() -> None:
    document = AgentDocument("doc_short", "short text")
    chunks = chunk_document(document, max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"
