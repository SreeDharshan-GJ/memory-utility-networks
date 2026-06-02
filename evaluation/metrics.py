"""
Evaluation Metrics
==================
Publication-quality evaluation metrics for memory retention systems.

Metrics:
  Recall@K              — fraction of truly useful memories in top-K predictions
  Precision@K           — fraction of top-K predicted memories that are truly useful
  Average Precision (AP) — area under precision-recall curve (per query)
  Mean Average Precision (MAP) — AP averaged over queries
  NDCG@K                — normalised discounted cumulative gain (soft labels)
  Utility AUC           — AUROC for binary utility prediction
  Utility Accuracy      — binary classification accuracy at threshold 0.5
  Long-Horizon Recall   — Recall@K for memories older than a threshold
  Retrieval Success Rate — fraction of needed memories still in store

Bug fixes vs original:
  - ndcg_at_k: when idcg==0, return float('nan') rather than 1.0 (undefined case)
  - recall_at_k: when n_pos==0, return float('nan') rather than 1.0 (vacuously true)
  - MetricsEvaluator.compute: filter nan values before averaging
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# ── Utility Helpers ────────────────────────────────────────────────────────────

def _rank_scores(scores: np.ndarray) -> np.ndarray:
    """Return indices that sort scores descending."""
    return np.argsort(-scores)


def _dcg(relevances: np.ndarray, k: int) -> float:
    """Discounted cumulative gain at K."""
    relevances = relevances[:k].astype(float)
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(relevances / discounts))


def _idcg(relevances: np.ndarray, k: int) -> float:
    """Ideal DCG: sorted relevances descending."""
    ideal = np.sort(relevances)[::-1]
    return _dcg(ideal, k)


# ── Per-Query Metrics ──────────────────────────────────────────────────────────

def recall_at_k(
    scores: np.ndarray,
    binary_labels: np.ndarray,
    k: int,
) -> float:
    """
    Recall@K: fraction of positive memories that appear in the top-K.

    Returns float('nan') when there are no positive memories (undefined).
    Callers should filter NaN values before aggregation.
    """
    n_pos = int(binary_labels.sum())
    if n_pos == 0:
        return float("nan")   # FIX: was 1.0 (inflated MAP when no positives)
    top_k_idx = _rank_scores(scores)[:k]
    hits = int(binary_labels[top_k_idx].sum())
    return hits / n_pos


def precision_at_k(
    scores: np.ndarray,
    binary_labels: np.ndarray,
    k: int,
) -> float:
    """Precision@K: fraction of top-K predicted memories that are truly useful."""
    top_k_idx = _rank_scores(scores)[:k]
    if len(top_k_idx) == 0:
        return 0.0
    return float(binary_labels[top_k_idx].mean())


def average_precision(
    scores: np.ndarray,
    binary_labels: np.ndarray,
) -> float:
    """
    Average Precision (AP): area under the Precision-Recall curve.
    Returns 0.0 when there are no positive memories.
    """
    n_pos = int(binary_labels.sum())
    if n_pos == 0:
        return 0.0
    ranked_idx = _rank_scores(scores)
    hits = 0
    ap = 0.0
    for rank, idx in enumerate(ranked_idx, start=1):
        if binary_labels[idx]:
            hits += 1
            ap += hits / rank
    return ap / n_pos


def ndcg_at_k(
    scores: np.ndarray,
    soft_labels: np.ndarray,
    k: int,
) -> float:
    """
    NDCG@K using soft utility labels as relevance grades.

    Returns float('nan') when all soft labels are zero (undefined ideal ranking).
    """
    ranked_idx = _rank_scores(scores)
    ranked_relevances = soft_labels[ranked_idx]
    dcg = _dcg(ranked_relevances, k)
    idcg = _idcg(soft_labels, k)
    if idcg == 0.0:
        return float("nan")   # FIX: was 1.0 (undefined when all relevances are 0)
    return dcg / idcg


def utility_auc(
    scores: np.ndarray,
    binary_labels: np.ndarray,
) -> float:
    """
    AUROC for binary utility classification (trapezoidal approximation).
    Returns 0.5 when only one class is present (uninformative).
    """
    n_pos = int(binary_labels.sum())
    n_neg = len(binary_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranked_idx = _rank_scores(scores)
    ranked_labels = binary_labels[ranked_idx]

    tps = np.cumsum(ranked_labels)
    fps = np.cumsum(1 - ranked_labels)
    tpr = tps / max(n_pos, 1)
    fpr = fps / max(n_neg, 1)

    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    # np.trapezoid added in NumPy 2.0; fall back to np.trapz for 1.x
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))


def utility_accuracy(
    scores: np.ndarray,
    binary_labels: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Binary classification accuracy at a fixed decision threshold."""
    predicted = (scores >= threshold).astype(int)
    return float((predicted == binary_labels).mean())


# ── System-Level Metrics ───────────────────────────────────────────────────────

def memory_retention_efficiency(
    retained_labels: np.ndarray,
    evicted_labels: np.ndarray,
) -> float:
    """
    Memory Retention Efficiency (MRE):
    Fraction of useful memories retained (not evicted).

    MRE = TP_retained / (TP_retained + FN_evicted)
    """
    retained_pos = int(retained_labels.sum())
    evicted_pos = int(evicted_labels.sum())
    total_pos = retained_pos + evicted_pos
    if total_pos == 0:
        return 1.0
    return retained_pos / total_pos


def storage_utilisation(retained_labels: np.ndarray) -> float:
    """
    Storage Utilisation: fraction of memory slots holding genuinely useful items.
    Penalises storing many useless memories.
    """
    if len(retained_labels) == 0:
        return 0.0
    return float(retained_labels.mean())


def long_horizon_recall(
    scores: np.ndarray,
    binary_labels: np.ndarray,
    ages: np.ndarray,
    k: int,
    age_threshold: float = 20.0,
) -> float:
    """
    Long-Horizon Recall@K computed only on memories older than ``age_threshold``.
    Tests whether the model retains distantly-created but still-useful memories.
    Returns float('nan') if no memories are older than the threshold.
    """
    old_mask = ages >= age_threshold
    if old_mask.sum() == 0:
        return float("nan")
    return recall_at_k(scores[old_mask], binary_labels[old_mask], k)


def retrieval_success_rate(
    retrieved_memory_ids: List[int],
    future_needed_ids: List[int],
) -> float:
    """
    Retrieval Success Rate: fraction of future-needed memories that were
    successfully retrieved (i.e. were still present in the memory store).
    """
    if len(future_needed_ids) == 0:
        return 1.0
    store_set = set(retrieved_memory_ids)
    found = sum(1 for mid in future_needed_ids if mid in store_set)
    return found / len(future_needed_ids)


# ── Aggregate Evaluator ────────────────────────────────────────────────────────

class MetricsEvaluator:
    """
    Computes and aggregates all metrics over a collection of evaluation records.

    Usage::

        evaluator = MetricsEvaluator(k_values=[5, 10, 20])
        for record in eval_records:
            evaluator.update(
                scores=record['predicted_scores'],
                binary_labels=record['binary_labels'],
                soft_labels=record['soft_labels'],
                memory_ages=record['memory_ages'],
            )
        results = evaluator.compute()
    """

    def __init__(
        self,
        k_values: Optional[List[int]] = None,
        age_threshold: float = 20.0,
        threshold: float = 0.5,
    ):
        self.k_values = k_values or [5, 10, 20, 50]
        self.age_threshold = age_threshold
        self.threshold = threshold
        self._records: List[Dict] = []
        self._latencies: List[float] = []

    def update(
        self,
        scores: np.ndarray,
        binary_labels: np.ndarray,
        soft_labels: np.ndarray,
        memory_ages: Optional[np.ndarray] = None,
        latency_s: Optional[float] = None,
    ) -> None:
        """Register one evaluation instance (one memory snapshot)."""
        self._records.append({
            "scores": scores.astype(np.float32),
            "binary_labels": binary_labels.astype(np.int32),
            "soft_labels": soft_labels.astype(np.float32),
            "ages": memory_ages.astype(np.float32) if memory_ages is not None else None,
        })
        if latency_s is not None:
            self._latencies.append(float(latency_s))

    def compute(self) -> Dict[str, float]:
        """Compute and aggregate all metrics. NaN values are excluded from means."""
        if not self._records:
            return {}

        accumulators: Dict[str, List[float]] = {
            "auc": [],
            "accuracy": [],
            "ap": [],
        }
        for k in self.k_values:
            accumulators[f"recall@{k}"] = []
            accumulators[f"precision@{k}"] = []
            accumulators[f"ndcg@{k}"] = []
            accumulators[f"long_horizon_recall@{k}"] = []

        for rec in self._records:
            s = rec["scores"]
            b = rec["binary_labels"]
            soft = rec["soft_labels"]
            ages = rec["ages"]

            accumulators["auc"].append(utility_auc(s, b))
            accumulators["accuracy"].append(utility_accuracy(s, b, self.threshold))
            accumulators["ap"].append(average_precision(s, b))

            for k in self.k_values:
                r = recall_at_k(s, b, k)
                p = precision_at_k(s, b, k)
                n = ndcg_at_k(s, soft, k)
                accumulators[f"recall@{k}"].append(r)
                accumulators[f"precision@{k}"].append(p)
                accumulators[f"ndcg@{k}"].append(n)

                if ages is not None:
                    lhr = long_horizon_recall(s, b, ages, k, self.age_threshold)
                    accumulators[f"long_horizon_recall@{k}"].append(lhr)

        aggregated: Dict[str, float] = {}
        for key, vals in accumulators.items():
            # FIX: filter NaN before computing mean/std
            finite = [v for v in vals if not np.isnan(v)]
            if finite:
                aggregated[f"mean_{key}"] = float(np.mean(finite))
                aggregated[f"std_{key}"] = float(np.std(finite))

        # Rename mean_ap → map for conventional reporting
        if "mean_ap" in aggregated:
            aggregated["map"] = aggregated.pop("mean_ap")
        if "std_ap" in aggregated:
            aggregated["std_map"] = aggregated.pop("std_ap")

        if self._latencies:
            aggregated["mean_latency_ms"] = float(np.mean(self._latencies)) * 1000.0
            aggregated["p95_latency_ms"] = float(np.percentile(self._latencies, 95)) * 1000.0

        return aggregated

    def reset(self) -> None:
        self._records.clear()
        self._latencies.clear()
