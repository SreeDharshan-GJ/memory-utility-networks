# Memory Utility Networks (MUN) 

<div align="center">

### Utility-Based Memory Retrieval for In-Context Learning and Long-Horizon Decision Making

[![Status](https://img.shields.io/badge/Status-Active%20Research-orange.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)]()
[![Research](https://img.shields.io/badge/Type-Research-red.svg)]()

</div>

---

## Overview

Memory Utility Networks (MUN) is a research-oriented framework that investigates how intelligent systems can learn to retrieve and prioritize memories based on estimated future utility.

Traditional memory systems rely on handcrafted retrieval strategies such as:

* Random selection
* Similarity-based retrieval
* Recency-based retrieval

MUN introduces a utility-driven retrieval mechanism that estimates the future usefulness of memories and prioritizes them accordingly.

The framework is designed for studying:

* Memory-augmented learning systems
* In-context learning
* Long-horizon decision making
* Example selection strategies
* Adaptive memory retrieval

---

## Research Status

**Current Status:** Active Research

* Framework implementation complete
* Baseline evaluation complete
* Utility scoring module implemented
* Ongoing investigation of adaptive retrieval strategies

This repository represents an active research project exploring utility-based memory selection.

---

## Motivation

Most existing memory retrieval methods assume that the most similar or most recent memory is also the most useful.

However, useful memories are not always:

* the newest
* the closest
* the most frequently accessed

MUN explores whether a learned utility function can identify memories that contribute more effectively to downstream task performance.

The central hypothesis is that retrieval systems should optimize for future utility rather than similarity alone.

---

## Architecture

```text
Query
  │
  ▼
Memory Store
  │
  ▼
Utility Network
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

The Utility Network learns to assign a utility score to candidate memories and selects the most promising subset for downstream inference.

---

## Repository Structure

```text
memory-utility-networks/
│
├── README.md
├── .gitignore
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
└── configs/
    ├── default.yaml
    ├── train.yaml
    └── eval.yaml
```

---

## Installation

```bash
git clone https://github.com/SreeDharshan-GJ/memory-utility-networks.git
cd memory-utility-networks
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Evaluation

```bash
python evaluate.py
```

---

## Benchmarking

```bash
python benchmark.py
```

Supported baselines include:

* Random Retrieval
* Similarity Retrieval
* Recency Retrieval
* Utility-Based Retrieval (MUN)

---

## Experimental Findings

Memory Utility Networks were evaluated against standard retrieval baselines.

| Method     | Description                    |
| ---------- | ------------------------------ |
| Random     | Random memory retrieval        |
| Recency    | Most recent memories           |
| Similarity | Embedding similarity retrieval |
| MUN        | Utility-based retrieval        |

The framework enables systematic comparison of retrieval strategies and supports controlled evaluation of memory utility estimation.

---

## Research Questions

1. Can utility-based retrieval outperform heuristic memory selection?
2. How does memory utility evolve over time?
3. Can utility estimation improve long-horizon reasoning?
4. What retrieval strategies remain effective under constrained memory budgets?
5. How does adaptive retrieval influence downstream prediction quality?

---

## Future Work

* Transformer-based utility estimators
* Multi-agent memory systems
* Reinforcement learning for memory management
* Continual learning benchmarks
* Retrieval-Augmented Generation (RAG)
* Large Language Model integration
* Hierarchical memory architectures
* Dynamic memory compression

---

## Citation

```bibtex
@software{memoryutilitynetworks,
  title={Memory Utility Networks},
  author={Sree Dharshan G J},
  year={2026},
  url={https://github.com/SreeDharshan-GJ/memory-utility-networks}
}
```

---

## Usage Restrictions

This repository is provided for academic review and research discussion.

All rights are reserved by the author.

For collaboration, licensing inquiries, or research partnerships, please contact the author directly.

---

## Intellectual Property Notice

Copyright (c) 2026 Sree Dharshan G J

All rights reserved.

The concepts, experimental methodology, architecture design, and implementation details presented in this repository constitute original research work.

No permission is granted to copy, reproduce, redistribute, modify, commercialize, or create derivative works from this repository without explicit written permission from the author.

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
* Intelligent Decision Systems
* Autonomous Agents
* Adaptive Retrieval Systems

---

## Disclaimer

This repository is intended for research and academic review purposes. The framework is actively under development and serves as a platform for studying utility-based memory retrieval strategies in intelligent systems.

