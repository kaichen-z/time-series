"""Lexical BM25 retrieval: dependency-free, and the default."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from . import Chunk

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric tokens."""
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class BM25Config:
    """BM25 scoring hyperparameters."""

    k1: float = 1.5
    b: float = 0.75


class BM25Index:
    """A BM25 index over a fixed set of chunks."""

    def __init__(self, chunks: Sequence[Chunk], config: BM25Config | None = None) -> None:
        self.config = config or BM25Config()
        self.chunks = list(chunks)
        self._tokenized = [tokenize(chunk.text) for chunk in self.chunks]
        self._doc_frequency: Counter[str] = Counter()
        for tokens in self._tokenized:
            self._doc_frequency.update(set(tokens))
        total_length = sum(len(tokens) for tokens in self._tokenized)
        self._avg_length = (total_length / len(self._tokenized)) if total_length else 1.0
        self._total = len(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the top_k chunks ranked by BM25 score against the query."""
        query_counts = Counter(tokenize(query))
        if not query_counts or not self.chunks:
            return []
        scored: list[tuple[float, int]] = []
        for index, tokens in enumerate(self._tokenized):
            counts = Counter(tokens)
            length_norm = self.config.k1 * (1 - self.config.b + self.config.b * len(tokens) / self._avg_length)
            score = 0.0
            for term, query_weight in query_counts.items():
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                doc_frequency = self._doc_frequency[term]
                idf = math.log(1 + (self._total - doc_frequency + 0.5) / (doc_frequency + 0.5))
                term_score = idf * frequency * (self.config.k1 + 1) / (frequency + length_norm)
                score += min(query_weight, 3) * term_score
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], self.chunks[item[1]].chunk_id))
        return [(self.chunks[index], score) for score, index in scored[:top_k]]
