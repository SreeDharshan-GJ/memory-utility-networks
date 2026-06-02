# Memory Utility Networks (MUN)

> A utility-based memory retrieval framework for adaptive memory selection, in-context learning, and long-horizon decision making.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()
[![Research](https://img.shields.io/badge/Type-Research-red.svg)]()

---

## Overview

Memory Utility Networks (MUN) is a research-oriented framework that investigates how intelligent systems can learn to retrieve and prioritize memories based on estimated future utility.

Traditional memory systems rely on handcrafted retrieval strategies such as:

- Random selection
- Similarity-based retrieval
- Recency-based retrieval

MUN introduces a utility-driven retrieval mechanism that estimates the future usefulness of memories and prioritizes them accordingly.

The framework is designed for studying:

- Memory-augmented learning systems
- In-context learning
- Long-horizon decision making
- Example selection strategies
- Adaptive memory retrieval

---

## Motivation

Most existing memory retrieval methods assume that the most similar or most recent memory is also the most useful.

However, useful memories are not always:

- the newest,
- the closest,
- or the most frequently accessed.

MUN explores whether a learned utility function can identify memories that contribute more effectively to downstream task performance.

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
├── LICENSE
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

Clone the repository:

```bash
git clone https://github.com/SreeDharshan-GJ/memory-utility-networks.git
cd memory-utility-networks
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Train the utility network:

```bash
python train.py
```

---

## Evaluation

Evaluate retrieval performance:

```bash
python evaluate.py
```

---

## Benchmarking

Compare MUN against baseline retrieval policies:

```bash
python benchmark.py
```

Supported baselines include:

- Random Retrieval
- Similarity Retrieval
- Recency Retrieval
- Utility-Based Retrieval (MUN)

---

## Research Goals

This project investigates the following questions:

1. Can utility-based retrieval outperform heuristic memory selection?
2. How does memory utility evolve over time?
3. Can utility estimation improve long-horizon reasoning?
4. What retrieval strategies are most effective under limited memory budgets?
5. How does adaptive retrieval affect downstream prediction quality?

---

## Future Work

- Transformer-based utility estimators
- Multi-agent memory systems
- Reinforcement learning for memory management
- Continual learning benchmarks
- Large language model integration

---

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@software{memoryutilitynetworks,
  title={Memory Utility Networks},
  author={Sree Dharshan G J},
  year={2026},
  url={https://github.com/SreeDharshan-GJ/memory-utility-networks}
}
```

---

## License

This project is released under the Apache 2.0 License.

See [LICENSE](LICENSE) for details.

---

## Author

**Sree Dharshan G J**

Electronics and Communication Engineering  
SRM Institute of Science and Technology

Research Interests:

- Machine Learning
- Memory-Augmented Systems
- In-Context Learning
- Multi-Agent AI
- Intelligent Decision Systems
