"""Chunking, BM25 ranking, and dense ranking against small toy corpora."""

from __future__ import annotations

import sys

import pytest

from dr_cik.models import AgentDocument
from dr_cik.retrieval import Chunk, build_index, chunk_document
from dr_cik.retrieval.bm25 import BM25Index, tokenize
from dr_cik.retrieval.dense import DenseConfig, DenseIndex, DenseRetrieverUnavailableError

TOY_CORPUS = (
    AgentDocument("doc_a", "Solar irradiance measurements at the research annex."),
    AgentDocument("doc_b", "Quarterly sales figures for the cosmetic lab."),
    AgentDocument("doc_c", "CPU usage patterns for the technical site."),
)


# --- chunking ---------------------------------------------------------------------------


def test_chunk_document_splits_long_text_with_overlap() -> None:
    chunks = chunk_document(AgentDocument("doc_long", "x" * 3000), max_chars=1000, overlap=100)
    assert len(chunks) > 1
    assert all(chunk.document_id == "doc_long" for chunk in chunks)
    assert all(len(chunk.text) <= 1000 for chunk in chunks)


def test_chunk_document_keeps_short_text_whole() -> None:
    chunks = chunk_document(AgentDocument("doc_short", "short text"), max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


# --- BM25 -------------------------------------------------------------------------------


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("Solar Irradiance, 2022!") == ["solar", "irradiance", "2022"]


def test_bm25_ranks_the_matching_document_first() -> None:
    index = build_index(TOY_CORPUS)
    results = index.search("solar irradiance annex", top_k=2)
    assert results[0][0].document_id == "doc_a"
    assert results[0][1] > 0


def test_bm25_returns_empty_for_empty_corpus() -> None:
    assert BM25Index([]).search("anything") == []


# --- dense ------------------------------------------------------------------------------


class _FakeEncoder:
    """Maps text to a 3-d vector by keyword presence, so cosine ranking is predictable."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "solar" in lowered or "irradiance" in lowered else 0.0,
                    1.0 if "sales" in lowered or "cosmetic" in lowered else 0.0,
                    1.0 if "cpu" in lowered or "technical" in lowered else 0.0,
                ]
            )
        return vectors


def _dense_index(**kwargs) -> DenseIndex:
    chunks = [chunk_document(document)[0] for document in TOY_CORPUS]
    return DenseIndex(chunks, config=DenseConfig(**kwargs), encoder=_FakeEncoder())


def test_dense_ranks_the_semantically_matching_document_first() -> None:
    results = _dense_index().search("solar irradiance", top_k=2)
    assert results[0][0].document_id == "doc_a"
    assert results[0][1] == pytest.approx(1.0)


def test_dense_encodes_the_corpus_only_once() -> None:
    index = _dense_index()
    index.search("solar", top_k=1)
    first = index._embeddings
    index.search("cpu", top_k=1)
    assert index._embeddings is first


def test_dense_threshold_filters_low_similarity_hits() -> None:
    """DRBench applies a similarity floor; ours defaults to 0.0 but must work when set."""
    unfiltered = _dense_index().search("solar irradiance", top_k=3)
    filtered = _dense_index(threshold=0.99).search("solar irradiance", top_k=3)

    assert len(unfiltered) == 3  # top_k as ranked, including the orthogonal docs scoring 0.0
    assert [chunk.document_id for chunk, _ in filtered] == ["doc_a"]


def test_dense_returns_empty_for_empty_corpus_or_query() -> None:
    assert DenseIndex([], encoder=_FakeEncoder()).search("anything") == []
    assert _dense_index().search("   ") == []


def test_dense_without_sentence_transformers_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # force ImportError regardless of install state
    index = DenseIndex([Chunk("d", "d:0", "text")])
    with pytest.raises(DenseRetrieverUnavailableError):
        index._load_encoder()


# --- factory ----------------------------------------------------------------------------


def test_build_index_selects_the_requested_retriever() -> None:
    assert isinstance(build_index(TOY_CORPUS, retriever="bm25"), BM25Index)
    assert isinstance(build_index(TOY_CORPUS, retriever="dense", encoder=_FakeEncoder()), DenseIndex)


def test_build_index_rejects_an_unknown_retriever() -> None:
    with pytest.raises(ValueError, match="Unknown retriever"):
        build_index(TOY_CORPUS, retriever="nope")
