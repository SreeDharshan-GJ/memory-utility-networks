"""
Benchmark Runner
================
Runs comprehensive comparison experiments for all memory policies
and generates publication-quality plots and LaTeX tables.

Experiments:
  1. Overall metric comparison (all policies × all metrics)
  2. Task-type breakdown (qa / reasoning / planning / coding / summarization)
  3. Scalability: recall vs memory capacity
  4. Forgetting curves: recall vs memory age
  5. Distractor robustness: needle-in-haystack
  6. Long-horizon: recall at increasing episode horizons

Outputs:
  benchmark/
    tables/
      overall_comparison.tex
      task_breakdown.tex
    plots/
      overall_comparison.pdf
      forgetting_curves.pdf
      recall_vs_capacity.pdf
      long_horizon.pdf
    data/
      benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import build_model
from training.synthetic_memory_generator import SyntheticEpisodeGenerator, TaskPattern
from evaluation import MetricsEvaluator
from evaluate import BaselineScorer, MUNScorer, evaluate_policy


# ── Config ────────────────────────────────────────────────────────────────────

POLICIES = ["fifo", "lru", "random", "recency", "frequency", "similarity", "tfidf"]
TASK_TYPES = TaskPattern.TASK_TYPES
K_VALUES = [5, 10, 20, 50]


def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = Path(path).parent / cfg.pop("_base_")
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        return base
    return cfg


# ── Experiment Runners ────────────────────────────────────────────────────────

def run_overall_comparison(
    episodes: List[Any],
    scorers: Dict[str, Any],
    embed_dim: int,
    k_values: List[int],
) -> Dict[str, Dict[str, float]]:
    """Run all policies on all episodes, return full metric dicts."""
    results = {}
    for name, scorer in scorers.items():
        is_mun = name == "mun"
        print(f"  [{name}] ...", end=" ", flush=True)
        t0 = time.perf_counter()
        res = evaluate_policy(scorer, episodes, embed_dim, k_values, is_mun=is_mun)
        elapsed = time.perf_counter() - t0
        results[name] = res
        print(f"MAP={res.get('map', 0):.4f} ({elapsed:.1f}s)")
    return results


def run_task_breakdown(
    all_episodes: List[Any],
    scorers: Dict[str, Any],
    embed_dim: int,
    k_values: List[int],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Per task-type breakdown: {task_type: {policy: metrics}}."""
    by_task: Dict[str, List[Any]] = {t: [] for t in TASK_TYPES}
    for ep in all_episodes:
        t = ep.task_type
        if t in by_task:
            by_task[t].append(ep)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for task_type, episodes in by_task.items():
        if not episodes:
            continue
        print(f"\n  Task: {task_type} ({len(episodes)} episodes)")
        results[task_type] = {}
        for name, scorer in scorers.items():
            is_mun = name == "mun"
            res = evaluate_policy(scorer, episodes, embed_dim, k_values, is_mun=is_mun)
            results[task_type][name] = res
            print(f"    {name}: MAP={res.get('map', 0):.4f}")
    return results


def run_capacity_scaling(
    scorers: Dict[str, Any],
    capacities: List[int],
    embed_dim: int,
    num_episodes: int,
    seed: int,
) -> Dict[int, Dict[str, float]]:
    """Recall@10 vs memory capacity for all policies."""
    results: Dict[int, Dict[str, float]] = {}
    for cap in capacities:
        print(f"  Capacity={cap}")
        gen = SyntheticEpisodeGenerator(
            embed_dim=embed_dim,
            memory_capacity=cap,
            min_episode_length=20,
            max_episode_length=50,
            seed=seed + cap,
        )
        episodes = gen.generate_episodes(num_episodes)
        cap_results: Dict[str, float] = {}
        for name, scorer in scorers.items():
            is_mun = name == "mun"
            res = evaluate_policy(scorer, episodes, embed_dim, [10], is_mun=is_mun)
            cap_results[name] = res.get("mean_recall@10", 0.0)
        results[cap] = cap_results
        print({k: f"{v:.4f}" for k, v in cap_results.items()})
    return results


def run_forgetting_curves(
    scorers: Dict[str, Any],
    age_thresholds: List[float],
    embed_dim: int,
    num_episodes: int,
    seed: int,
) -> Dict[float, Dict[str, float]]:
    """Recall@10 of old memories (age ≥ threshold) for all policies."""
    gen = SyntheticEpisodeGenerator(
        embed_dim=embed_dim,
        memory_capacity=200,
        min_episode_length=40,
        max_episode_length=100,
        seed=seed + 555,
    )
    episodes = gen.generate_episodes(num_episodes)

    results: Dict[float, Dict[str, float]] = {}
    for thresh in age_thresholds:
        print(f"  Age threshold={thresh}")
        thresh_results: Dict[str, float] = {}
        for name, scorer in scorers.items():
            evaluator = MetricsEvaluator(k_values=[10], age_threshold=thresh)
            for ep in episodes:
                recs = gen.episode_to_training_records(ep)
                for rec in recs:
                    mem_emb = rec["memory_embeddings"]
                    ctx_emb = rec["context_embeddings"]
                    ages = rec["memory_ages"]
                    binary = rec["binary_labels"].astype(np.int32)
                    soft = rec["soft_labels"].astype(np.float32)
                    if mem_emb.shape[0] == 0:
                        continue
                    if name == "mun":
                        scores = scorer.score_memories(mem_emb, ctx_emb, ages)
                    else:
                        scores = scorer.score_memories(mem_emb, binary, ages)
                    evaluator.update(scores, binary, soft, ages)

            res = evaluator.compute()
            lhr = res.get("mean_long_horizon_recall@10", float("nan"))
            thresh_results[name] = lhr
        results[thresh] = thresh_results
        print({k: f"{v:.4f}" if not np.isnan(v) else "nan" for k, v in thresh_results.items()})
    return results


def run_long_horizon(
    scorers: Dict[str, Any],
    horizon_lengths: List[int],
    embed_dim: int,
    num_episodes: int,
    seed: int,
) -> Dict[int, Dict[str, float]]:
    """Recall@10 vs episode horizon length."""
    results: Dict[int, Dict[str, float]] = {}
    for horizon in horizon_lengths:
        print(f"  Horizon={horizon}")
        gen = SyntheticEpisodeGenerator(
            embed_dim=embed_dim,
            memory_capacity=200,
            min_episode_length=max(5, horizon // 2),
            max_episode_length=horizon,
            seed=seed + horizon * 7,
        )
        episodes = gen.generate_episodes(num_episodes)
        h_results: Dict[str, float] = {}
        for name, scorer in scorers.items():
            is_mun = name == "mun"
            res = evaluate_policy(scorer, episodes, embed_dim, [10], is_mun=is_mun)
            h_results[name] = res.get("mean_recall@10", 0.0)
        results[horizon] = h_results
        print({k: f"{v:.4f}" for k, v in h_results.items()})
    return results


# ── Reporting ──────────────────────────────────────────────────────────────────

def _format_latex_table(
    data: Dict[str, Dict[str, float]],
    metric_keys: List[str],
    caption: str,
    label: str,
) -> str:
    """
    Produce a LaTeX booktabs table from a {policy: {metric: value}} dict.
    Best value in each column is bolded.
    """
    col_fmt = "l" + "c" * len(metric_keys)
    header = " & ".join(["Policy"] + [k.replace("_", r"\_").replace("@", "@") for k in metric_keys])

    # Find best value per metric (higher is better for all reported metrics)
    best: Dict[str, float] = {}
    for mk in metric_keys:
        vals = [v.get(mk, float("nan")) for v in data.values() if not np.isnan(v.get(mk, float("nan")))]
        best[mk] = max(vals) if vals else float("nan")

    rows = []
    for policy, metrics in data.items():
        cells = [policy.upper()]
        for mk in metric_keys:
            val = metrics.get(mk, float("nan"))
            if np.isnan(val):
                cells.append("—")
            elif abs(val - best.get(mk, float("nan"))) < 1e-6:
                cells.append(f"\\textbf{{{val:.4f}}}")
            else:
                cells.append(f"{val:.4f}")
        rows.append(" & ".join(cells) + r" \\")

    body = "\n        ".join(rows)
    return f"""\\begin{{table}}[t]
  \\centering
  \\caption{{{caption}}}
  \\label{{{label}}}
  \\begin{{tabular}}{{{col_fmt}}}
    \\toprule
    {header} \\\\
    \\midrule
    {body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""


def generate_plots(
    overall: Dict[str, Dict[str, float]],
    capacity_results: Dict[int, Dict[str, float]],
    forgetting_results: Dict[float, Dict[str, float]],
    horizon_results: Dict[int, Dict[str, float]],
    output_dir: Path,
    plot_format: str = "pdf",
    dpi: int = 150,
):
    """Generate matplotlib figures for all experiments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    POLICY_COLORS = {
        "fifo": "#e41a1c", "lru": "#377eb8", "random": "#4daf4a",
        "recency": "#984ea3", "frequency": "#ff7f00",
        "similarity": "#a65628", "tfidf": "#f781bf", "mun": "#000000",
    }
    POLICY_MARKERS = {
        "fifo": "o", "lru": "s", "random": "^", "recency": "D",
        "frequency": "v", "similarity": "P", "tfidf": "X", "mun": "*",
    }

    # 1. Overall bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metric_display = {"map": "MAP", "mean_auc": "AUC", "mean_recall@10": "Recall@10"}
    for ax, (mkey, mlabel) in zip(axes, metric_display.items()):
        policies = list(overall.keys())
        vals = [overall[p].get(mkey, 0.0) for p in policies]
        colors = [POLICY_COLORS.get(p, "#888888") for p in policies]
        bars = ax.bar(range(len(policies)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels([p.upper() for p in policies], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(mlabel)
        ax.set_ylim(0, 1.05)
        ax.set_title(mlabel)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.suptitle("Memory Policy Comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / f"overall_comparison.{plot_format}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 2. Forgetting curves
    if forgetting_results:
        fig, ax = plt.subplots(figsize=(7, 4))
        thresholds = sorted(forgetting_results.keys())
        all_policy_names = list(next(iter(forgetting_results.values())).keys())
        for p in all_policy_names:
            vals = [forgetting_results[t].get(p, float("nan")) for t in thresholds]
            ax.plot(
                thresholds, vals,
                label=p.upper(),
                color=POLICY_COLORS.get(p, "#888"),
                marker=POLICY_MARKERS.get(p, "o"),
                linewidth=1.5,
                markersize=5,
            )
        ax.set_xlabel("Memory Age Threshold (steps)")
        ax.set_ylabel("Long-Horizon Recall@10")
        ax.set_title("Forgetting Curves")
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(plots_dir / f"forgetting_curves.{plot_format}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # 3. Recall vs capacity
    if capacity_results:
        fig, ax = plt.subplots(figsize=(7, 4))
        caps = sorted(capacity_results.keys())
        all_policy_names = list(next(iter(capacity_results.values())).keys())
        for p in all_policy_names:
            vals = [capacity_results[c].get(p, 0.0) for c in caps]
            ax.plot(
                caps, vals,
                label=p.upper(),
                color=POLICY_COLORS.get(p, "#888"),
                marker=POLICY_MARKERS.get(p, "o"),
                linewidth=1.5,
                markersize=5,
            )
        ax.set_xlabel("Memory Capacity")
        ax.set_ylabel("Recall@10")
        ax.set_title("Recall@10 vs Memory Capacity")
        ax.legend(fontsize=7, ncol=2)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(plots_dir / f"recall_vs_capacity.{plot_format}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # 4. Long-horizon
    if horizon_results:
        fig, ax = plt.subplots(figsize=(7, 4))
        horizons = sorted(horizon_results.keys())
        all_policy_names = list(next(iter(horizon_results.values())).keys())
        for p in all_policy_names:
            vals = [horizon_results[h].get(p, 0.0) for h in horizons]
            ax.plot(
                horizons, vals,
                label=p.upper(),
                color=POLICY_COLORS.get(p, "#888"),
                marker=POLICY_MARKERS.get(p, "o"),
                linewidth=1.5,
                markersize=5,
            )
        ax.set_xlabel("Episode Horizon Length")
        ax.set_ylabel("Recall@10")
        ax.set_title("Long-Horizon Recall Performance")
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(plots_dir / f"long_horizon.{plot_format}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Plots saved to {plots_dir}/")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace):
    cfg = load_config(args.config)
    eval_cfg = cfg.get("eval", cfg)
    output_dir = Path(args.output_dir or "benchmark")
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    seed = eval_cfg.get("seed", 42)
    num_episodes = min(eval_cfg.get("num_episodes", 500), args.max_episodes)
    embed_dim = cfg.get("model", {}).get("embedding_dim", 384)
    capacity = 200

    # ── Generate shared test episodes ──
    print(f"\nGenerating {num_episodes} benchmark episodes ...")
    gen = SyntheticEpisodeGenerator(
        embed_dim=embed_dim,
        memory_capacity=capacity,
        min_episode_length=10,
        max_episode_length=50,
        seed=seed + 424242,
    )
    all_episodes = gen.generate_episodes(num_episodes)

    # ── Build scorers ──
    scorers: Dict[str, Any] = {}
    for p in POLICIES:
        scorers[p] = BaselineScorer(p, capacity)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or eval_cfg.get("checkpoint_path", "checkpoints/best_model.pt")
    if Path(checkpoint).exists():
        print(f"Loading MUN from {checkpoint}")
        model = build_model(cfg).to(device)
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        scorers["mun"] = MUNScorer(model, device, embed_dim)
    else:
        print(f"MUN checkpoint not found at {checkpoint}; running baselines only.")

    benchmark_data: Dict[str, Any] = {}

    # ── Experiment 1: Overall comparison ──
    print("\n[1/4] Overall comparison ...")
    overall = run_overall_comparison(all_episodes, scorers, embed_dim, K_VALUES)
    benchmark_data["overall"] = overall

    # ── Experiment 2: Task breakdown ──
    print("\n[2/4] Task-type breakdown ...")
    task_breakdown = run_task_breakdown(all_episodes, scorers, embed_dim, K_VALUES)
    benchmark_data["task_breakdown"] = task_breakdown

    # ── Experiment 3: Capacity scaling ──
    print("\n[3/4] Capacity scaling ...")
    capacities = [10, 50, 100, 200, 500, 1000]
    cap_episodes = min(num_episodes // 2, 50)
    capacity_results = run_capacity_scaling(scorers, capacities, embed_dim, cap_episodes, seed)
    benchmark_data["capacity_scaling"] = {str(k): v for k, v in capacity_results.items()}

    # ── Experiment 4: Forgetting curves ──
    print("\n[4/4] Forgetting curves + long-horizon ...")
    age_thresholds = [1.0, 5.0, 10.0, 20.0, 35.0, 50.0]
    forgetting_results = run_forgetting_curves(scorers, age_thresholds, embed_dim, cap_episodes, seed)
    benchmark_data["forgetting_curves"] = {str(k): v for k, v in forgetting_results.items()}

    horizon_lengths = [10, 20, 30, 50, 75, 100]
    horizon_results = run_long_horizon(scorers, horizon_lengths, embed_dim, cap_episodes, seed)
    benchmark_data["long_horizon"] = {str(k): v for k, v in horizon_results.items()}

    # ── Save raw data ──
    data_path = output_dir / "data" / "benchmark_results.json"
    with open(data_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\nRaw results saved to {data_path}")

    # ── LaTeX tables ──
    primary_metrics = ["map", "mean_auc", "mean_recall@10", "mean_precision@10", "mean_ndcg@10"]
    overall_tex = _format_latex_table(
        overall, primary_metrics,
        caption="Memory policy comparison across all task types.",
        label="tab:overall_comparison",
    )
    (output_dir / "tables" / "overall_comparison.tex").write_text(overall_tex)

    # ── Plots ──
    report_cfg = eval_cfg.get("report", {})
    if report_cfg.get("generate_plots", True):
        print("\nGenerating plots ...")
        generate_plots(
            overall, capacity_results, forgetting_results, horizon_results,
            output_dir,
            plot_format=report_cfg.get("plot_format", "pdf"),
            dpi=report_cfg.get("dpi", 150),
        )

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Policy':<15} {'MAP':>8} {'AUC':>8} {'R@10':>8} {'R@20':>8}")
    print("-" * 70)
    for pol, res in overall.items():
        print(
            f"{pol:<15} "
            f"{res.get('map', 0):>8.4f} "
            f"{res.get('mean_auc', 0):>8.4f} "
            f"{res.get('mean_recall@10', 0):>8.4f} "
            f"{res.get('mean_recall@20', 0):>8.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Memory Utility Network")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="benchmark")
    parser.add_argument("--max-episodes", type=int, default=500,
                        help="Cap on number of test episodes (for fast runs)")
    args = parser.parse_args()
    main(args)
