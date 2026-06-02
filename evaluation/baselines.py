"""
Baseline Memory Management Strategies
======================================
Comparison baselines against the learned Memory Utility Network.

Implemented strategies:
  FIFO        — First-in, first-out
  LRU         — Least-recently used
  Random      — Uniform random eviction
  Recency     — Score = exp(-α * age), evict lowest
  Frequency   — Evict least-frequently accessed
  Similarity  — Evict most similar (deduplication)
  TF-IDF      — Retain memories most relevant to recent context

Bug fixes vs original:
  - BaseMemoryPolicy.should_evict: removed @abstractmethod decorator since it
    has a concrete body that all subclasses copy verbatim (dead duplication).
  - SimilarityDeduplicationPolicy: added guard for missing/empty embeddings.
  - TFIDFPolicy: IDF is now cached and only recomputed on add_memory (not per-score).
  - build_policy: all simple policy constructors now accept **kwargs safely.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


# ── Abstract Base ──────────────────────────────────────────────────────────────

class BaseMemoryPolicy(ABC):
    """Abstract base class for memory eviction policies."""

    def __init__(self, capacity: int, **kwargs: Any):
        self.capacity = capacity
        self.memories: List[Dict[str, Any]] = []
        self._step = 0

    def should_evict(self) -> bool:
        """Return True if the store is at capacity and must evict before adding."""
        return len(self.memories) >= self.capacity

    @abstractmethod
    def select_eviction_candidates(self, n: int = 1) -> List[int]:
        """Return indices (into self.memories) of memories to evict."""
        ...

    def add_memory(self, memory: Dict[str, Any]) -> Optional[int]:
        """
        Add a memory item, evicting one if necessary.

        Memory dict must have at minimum:
          'id'        — unique identifier
          'content'   — text content
          'embedding' — np.ndarray
          'timestamp' — int (creation step)

        Returns the index of the evicted memory, or None.
        """
        memory["_added_step"] = self._step
        memory["_access_count"] = 0
        memory["_last_access"] = self._step
        evicted_idx: Optional[int] = None

        if self.should_evict():
            candidates = self.select_eviction_candidates(1)
            if candidates:
                evicted_idx = candidates[0]
                self.memories.pop(evicted_idx)

        self.memories.append(memory)
        self._step += 1
        return evicted_idx

    def access_memory(self, idx: int) -> None:
        """Record that memory at index ``idx`` was accessed."""
        if 0 <= idx < len(self.memories):
            self.memories[idx]["_access_count"] += 1
            self.memories[idx]["_last_access"] = self._step

    def get_all_memories(self) -> List[Dict[str, Any]]:
        return self.memories.copy()

    def __len__(self) -> int:
        return len(self.memories)

    def clear(self) -> None:
        self.memories = []
        self._step = 0


# ── Concrete Policies ──────────────────────────────────────────────────────────

class FIFOPolicy(BaseMemoryPolicy):
    """First-In, First-Out: always evict the oldest memory."""

    def select_eviction_candidates(self, n: int = 1) -> List[int]:
        return list(range(min(n, len(self.memories))))


class LRUPolicy(BaseMemoryPolicy):
    """Least-Recently-Used: evict the memory accessed least recently."""

    def select_eviction_candidates(self, n: int = 1) -> List[int]:
        if not self.memories:
            return []
        indexed = sorted(
            enumerate(self.memories),
            key=lambda x: x[1].get("_last_access", 0),
        )
        return [idx for idx, _ in indexed[:n]]


class RandomPolicy(BaseMemoryPolicy):
    """Uniform random eviction."""

    def __init__(self, capacity: int, seed: int = 42, **kwargs: Any):
        super().__init__(capacity, **kwargs)
        self.rng = random.Random(seed)

    def select_eviction_candidates(self, n: int = 1) -> List[int]:
        if not self.memories:
            return []
        indices = list(range(len(self.memories)))
        return self.rng.sample(indices, min(n, len(indices)))


class RecencyPolicy(BaseMemoryPolicy):
    """
    Recency-based scoring: u(m, t) = exp(-α * (t - t_created)).
    Evict memories with the lowest recency score.
    """

    def __init__(self, capacity: int, decay_alpha: float = 0.05, **kwargs: Any):
        super().__init__(capacity, **kwargs)
        self.alpha = decay_alpha

    def _score(self, memory: Dict[str, Any]) -> float:
        age = self._step - memory.get("_added_step", 0)
        return math.exp(-self.alpha * age)

    def select_eviction_candidates(self, n: int = 1) -> List[int]:
        if not self.memories:
            return []
        scored = sorted(
            ((i, self._score(m)) for i, m in enumerate(self.memories)),
            key=lambda x: x[1],
        )
        return [i for i, _ in scored[:n]]


class FrequencyPolicy(BaseMemoryPolicy):
    """
    Frequency-based: evict memories with the lowest access count.
    Ties are broken by recency (oldest first).
    """

    def select_eviction_candidates(self, n: int = 1) -> List[int]:
        if not self.memories:
            return []
        scored = sorted(
            enumerate(self.memories),
            key=lambda x: (x[1].get("_access_count", 0), x[1].get("_last_access", 0)),
        )
        return [i for i, _ in scored[:n]]


class SimilarityDeduplicationPolicy(BaseMemoryPolicy):
    """
    Similarity-based deduplication: evict the stored memory most similar
    to the incoming memory (to maximise diversity in the store).

    Uses cosine similarity on embeddings.
    """

    def __init__(
        self,
        capacity: int,
        similarity_threshold: float = 0.85,
        **kwargs: Any,
    ):
        super().__init__(capacity, **kwargs)
        self.threshold = similarity_threshold
        self._new_embedding: Optional[np.ndarray] = None

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def select_eviction_candidates(self, n: int = 1) -> List[int]:
        if not self.memories:
            return []
        # FIX: guard for missing or empty new embedding
        if (
            self._new_embedding is None
            or self._new_embedding.ndim == 0
            or self._new_embedding.size == 0
        ):
            return list(range(min(n, len(self.memories))))   # fallback to FIFO

        similarities = []
        for i, m in enumerate(self.memories):
            emb = m.get("embedding")
            if emb is not None and np.asarray(emb).size > 0:
                sim = self._cosine_sim(np.asarray(emb, dtype=np.float32), self._new_embedding)
            else:
                sim = 0.0
            similarities.append((i, sim))

        # Evict the most similar (least diverse) stored memory
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in similarities[:n]]

    def add_memory(self, memory: Dict[str, Any]) -> Optional[int]:
        raw_emb = memory.get("embedding")
        if raw_emb is not None:
            self._new_embedding = np.asarray(raw_emb, dtype=np.float32)
        else:
            self._new_embedding = None
        result = super().add_memory(memory)
        self._new_embedding = None
        return result


class TFIDFPolicy(BaseMemoryPolicy):
    """
    TF-IDF based retention: keep memories most relevant to recent context queries.
    Evict memories with the lowest average TF-IDF relevance to recent context.

    Bug fix: IDF is cached and recomputed only when a new memory is added,
    not on every call to _tfidf_score (was O(|vocab|) per candidate per score).
    """

    def __init__(
        self,
        capacity: int,
        context_window: int = 5,
        **kwargs: Any,
    ):
        super().__init__(capacity, **kwargs)
        self.context_window = context_window
        self._recent_context: List[str] = []
        self._idf: Dict[str, float] = {}
        self._doc_freqs: Dict[str, int] = defaultdict(int)
        self._idf_dirty = True   # recompute on next score call

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    @staticmethod
    def _tf(tokens: List[str]) -> Dict[str, float]:
        freq: Dict[str, float] = defaultdict(float)
        for t in tokens:
            freq[t] += 1.0
        n = max(len(tokens), 1)
        return {k: v / n for k, v in freq.items()}

    def _recompute_idf(self) -> None:
        """Recompute IDF scores over the current memory store."""
        N = len(self.memories) + 1
        self._idf = {
            word: math.log((N + 1) / (df + 1)) + 1.0
            for word, df in self._doc_freqs.items()
        }
        self._idf_dirty = False

    def _tfidf_score(self, memory: Dict[str, Any], query_tokens: List[str]) -> float:
        if self._idf_dirty:
            self._recompute_idf()
        content = memory.get("content", "")
        tf = self._tf(self._tokenize(content))
        return sum(tf.get(qt, 0.0) * self._idf.get(qt, 1.0) for qt in query_tokens)

    def update_context(self, context: str) -> None:
        """Update the recent context window used for relevance scoring."""
        self._recent_context.append(context)
        if len(self._recent_context) > self.context_window:
            self._recent_context.pop(0)

    def select_eviction_candidates(self, n: int = 1) -> List[int]:
        if not self.memories:
            return []
        query_tokens = self._tokenize(" ".join(self._recent_context))
        if not query_tokens:
            return list(range(min(n, len(self.memories))))   # no context → FIFO

        scored = sorted(
            ((i, self._tfidf_score(m, query_tokens)) for i, m in enumerate(self.memories)),
            key=lambda x: x[1],   # lowest relevance first → evict
        )
        return [i for i, _ in scored[:n]]

    def add_memory(self, memory: Dict[str, Any]) -> Optional[int]:
        tokens = self._tokenize(memory.get("content", ""))
        for t in set(tokens):
            self._doc_freqs[t] += 1
        self._idf_dirty = True
        return super().add_memory(memory)


# ── Policy Registry & Factory ──────────────────────────────────────────────────

POLICY_REGISTRY: Dict[str, type] = {
    "fifo": FIFOPolicy,
    "lru": LRUPolicy,
    "random": RandomPolicy,
    "recency": RecencyPolicy,
    "frequency": FrequencyPolicy,
    "similarity": SimilarityDeduplicationPolicy,
    "tfidf": TFIDFPolicy,
}


def build_policy(name: str, capacity: int, **kwargs: Any) -> BaseMemoryPolicy:
    """Factory: create a memory policy by name."""
    name = name.lower()
    if name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy '{name}'. Available: {sorted(POLICY_REGISTRY.keys())}"
        )
    cls = POLICY_REGISTRY[name]
    return cls(capacity=capacity, **kwargs)
