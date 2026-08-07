"""Dense embedding retrieval, mirroring the method DRBench's own agent uses.

Ported from ServiceNow/drbench (Apache-2.0), `drbench/agents/drbench_agent/vector_store.py`:
its `_get_embeddings_local` encodes with a SentenceTransformer, and `semantic_search`
ranks by cosine similarity and takes `argsort(...)[::-1][:top_k]`. Both are reproduced below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from . import Chunk

# DRBench's local-embedding option. Chosen over its text-embedding-ada-002 default so
# retrieval stays offline and key-free, matching how the rest of this package runs.
DEFAULT_MODEL_ID = "all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = "/raid/home/air/khoutaibi/models"


class DenseRetrieverUnavailableError(RuntimeError):
    """Raised when sentence-transformers or the embedding checkpoint can't be loaded."""


@dataclass(frozen=True)
class DenseConfig:
    """Embedding model and search configuration."""

    model_id: str = DEFAULT_MODEL_ID
    cache_dir: str | None = DEFAULT_CACHE_DIR
    device: str | None = None  # None -> sentence-transformers picks (GPU if free)
    max_chars: int = 8192  # DRBench truncates the query at max_length before embedding
    threshold: float = 0.0
    """Minimum cosine similarity to keep.

    DRBench defaults to 0.7, tuned for text-embedding-ada-002, whose similarities sit high.
    MiniLM similarities are spread much lower, so 0.7 would discard nearly every hit; we
    default to 0.0 (keep the top_k as ranked) and leave the knob exposed.
    """


class DenseIndex:
    """Embeds chunks once, then ranks them against a query by cosine similarity."""

    def __init__(self, chunks: Sequence[Chunk], config: DenseConfig | None = None, encoder: Any | None = None) -> None:
        self.config = config or DenseConfig()
        self.chunks = list(chunks)
        self._encoder = encoder
        self._embeddings: Any | None = None

    def _load_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DenseRetrieverUnavailableError(
                "sentence-transformers is not installed; pip install 'dr-cik[dense]'"
            ) from exc
        try:
            self._encoder = SentenceTransformer(
                self.config.model_id, cache_folder=self.config.cache_dir, device=self.config.device
            )
        except Exception as exc:
            raise DenseRetrieverUnavailableError(f"Failed to load embedding model {self.config.model_id}: {exc}") from exc
        return self._encoder

    def _ensure_embeddings(self) -> Any:
        """Encode the corpus once, on first search."""
        import numpy as np

        if self._embeddings is None:
            texts = [chunk.text[: self.config.max_chars] for chunk in self.chunks]
            self._embeddings = np.asarray(self._load_encoder().encode(texts), dtype=float)
        return self._embeddings

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the top_k chunks ranked by cosine similarity against the query."""
        import numpy as np

        if not self.chunks or not query.strip():
            return []
        embeddings = self._ensure_embeddings()
        query_vector = np.asarray(self._load_encoder().encode([query[: self.config.max_chars]])[0], dtype=float)

        # Cosine similarity, exactly as DRBench's semantic_search computes it.
        denominator = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vector)
        similarities = np.divide(
            embeddings @ query_vector, denominator, out=np.zeros(len(embeddings)), where=denominator != 0
        )
        ranked = np.argsort(similarities)[::-1][:top_k]
        return [(self.chunks[i], float(similarities[i])) for i in ranked if similarities[i] >= self.config.threshold]
