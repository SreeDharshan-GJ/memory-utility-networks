# Memory Utility Networks (MUN)

<div align="center">

### An Empirical Investigation of Utility-Based Memory Retrieval for In-Context Learning and Decision Making

[![Status](https://img.shields.io/badge/Status-Research%20Complete-success.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)]()
[![Research](https://img.shields.io/badge/Type-Research-red.svg)]()

</div>

---

## Overview

Memory Utility Networks (MUN) is a research framework that investigates whether intelligent systems can learn to retrieve memories based on estimated future usefulness rather than relying solely on traditional retrieval strategies such as similarity, recency, or random selection.

The central research question explored in this project is:

> Can utility-based memory retrieval outperform heuristic memory selection strategies in downstream prediction tasks?

To answer this question, MUN was developed as a complete experimental framework including memory retrieval algorithms, baseline comparisons, ablation studies, leakage audits, reproducibility analysis, and utility-based selection mechanisms.

---

## Research Motivation

Most memory retrieval systems assume that the most similar or most recent memory is also the most useful.

However, in many real-world scenarios:

* Similar memories may not be the most informative.
* Recent memories may not be the most relevant.
* Useful memories may depend on context, diversity, or historical value.

MUN explores whether a learned notion of utility can identify memories that contribute more effectively to downstream reasoning and prediction.

---

## Architecture

```text
Query
  │
  ▼
Memory Store
  │
  ▼
Utility Scoring Module
  │
  ▼
Utility Scores
  │
  ▼
Top-K Memory Selection
  │
  ▼
Prediction / Decision
```

The framework assigns utility scores to candidate memories and selects the highest-scoring subset for inference.

---

## Research Contributions

This project includes:

* Design and implementation of a utility-based memory retrieval framework.
* Comparison against Random, Recency, and Similarity retrieval baselines.
* Utility scoring and memory ranking mechanisms.
* Controlled evaluation pipelines.
* Label leakage investigations.
* Ablation studies.
* Reproducibility audits.
* Analysis of utility-based retrieval under label-aware and label-free settings.

---

## Experimental Investigation

The research was conducted in two stages.

### MUN v1

The initial version introduced utility-based retrieval with label-aware scoring mechanisms.

Early experiments showed very strong performance improvements over baseline retrieval methods.

However, extensive ablation studies revealed that a significant portion of the observed improvement originated from label-aware retrieval rather than from learned utility estimation.

---

### MUN v2

A second generation framework was developed to eliminate label dependence completely.

MUN v2 introduced:

* Label-free utility estimation
* Feature-based utility scoring
* Information-theoretic retrieval features
* Memory diversity and uniqueness measures
* Reproducible evaluation procedures

This version was used to investigate whether utility-based retrieval could outperform similarity-based retrieval without access to label information.

---

## Key Findings

The project produced several important findings:

### Finding 1

Utility-based retrieval can appear highly effective when label information influences memory selection.

### Finding 2

Removing label information significantly reduces performance gains.

### Finding 3

Strong similarity-based retrieval remains a highly competitive baseline.

### Finding 4

Label-free utility estimation remains an open research challenge.

### Finding 5

Rigorous ablation studies and reproducibility audits are essential when evaluating memory retrieval systems.

---

## Repository Structure

```text
memory-utility-networks/
│
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── train.py
├── evaluate.py
├── benchmark.py
│
├── models/
│   ├── __init__.py
│   └── utility_network.py
│
├── evaluation/
│   ├── __init__.py
│   ├── baselines.py
│   └── metrics.py
│
├── configs/
│   ├── default.yaml
│   ├── train.yaml
│   └── eval.yaml
│
└── research/
    ├── ablations/
    ├── audits/
    └── reports/
```

---

## Installation

```bash
git clone https://github.com/SreeDharshan-GJ/memory-utility-networks.git

cd memory-utility-networks

pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python train.py
```

### Evaluation

```bash
python evaluate.py
```

### Benchmarking

```bash
python benchmark.py
```

---

## Research Questions

This project investigates:

1. Can utility-based retrieval outperform similarity-based retrieval?
2. How should memory utility be defined?
3. What information predicts future memory usefulness?
4. Can utility be estimated without access to labels?
5. What are the limitations of utility-based retrieval systems?

---

## Lessons Learned

One of the most valuable outcomes of this project was the discovery that apparent improvements in memory retrieval systems can arise from subtle sources of information leakage.

Through extensive experimentation, the project demonstrates the importance of:

* Controlled evaluation
* Strong baselines
* Reproducibility analysis
* Ablation studies
* Scientific transparency

These findings provide useful guidance for future research on memory-augmented learning systems.

---

## Future Directions

Potential future work includes:

* Multi-agent memory systems
* Reinforcement learning based memory management
* Transformer-based utility estimators
* Retrieval-Augmented Generation (RAG)
* Long-context language models
* Continual learning systems
* Hierarchical memory architectures

---

## Citation

```bibtex
@software{memoryutilitynetworks,
  title={Memory Utility Networks: An Empirical Investigation of Utility-Based Memory Retrieval},
  author={Sree Dharshan G J},
  year={2026},
  url={https://github.com/SreeDharshan-GJ/memory-utility-networks}
}
```

---

## Intellectual Property

Copyright © 2026 Sree Dharshan G J

All Rights Reserved.

This repository is provided for academic review and research discussion purposes only.

For collaboration, licensing, or research inquiries, please contact the author directly.

---

## Author

**Sree Dharshan G J**

Electronics and Communication Engineering
SRM Institute of Science and Technology

### Research Interests

* Machine Learning
* Memory-Augmented Systems
* In-Context Learning
* Multi-Agent AI
* Autonomous Agents
* Intelligent Decision Systems

---

## Project Status

**Research Complete**

This repository represents a completed research investigation into utility-based memory retrieval and serves as a platform for future work on memory-augmented intelligent systems.
