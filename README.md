# RAG Benchmark Kit

A lightweight Python toolkit for auditing and benchmarking Retrieval-Augmented Generation (RAG) evaluation datasets.

`rag-benchmark-kit` helps developers validate evaluation datasets, identify data-quality issues, and calculate retrieval performance metrics before evaluating their RAG systems.

## Features

- **Recall@k:** Measures how many relevant documents appear in the top-k retrieved results.
- **Reciprocal Rank:** Measures the rank of the first relevant retrieved document.
- **Mean Reciprocal Rank (MRR):** Calculates the average reciprocal rank across multiple queries.
- **Dataset Auditing:** Detects duplicate questions and invalid ground-truth document references.
- **Query Normalization:** Detects duplicates despite differences in capitalization or whitespace.
- **JSON Export:** Exports structured audit reports in JSON format.
- **CSV Export:** Exports audit issues in CSV format.
- **Input Validation:** Uses typed models and validation to catch invalid inputs.

## Installation

Requires Python 3.10 or later.

Install using pip:

```bash
pip install rag-benchmark-kit
```

## Quick Start

### 1. Create evaluation examples

```python
from rag_benchmark_kit import EvaluationExample

examples = [
    EvaluationExample(
        query="What is Redis?",
        relevant_doc_ids=frozenset({"doc_1", "doc_2"}),
    ),
    EvaluationExample(
        query="Explain PostgreSQL",
        relevant_doc_ids=frozenset({"doc_3"}),
    ),
]
```

Each evaluation example contains:
- A query.
- A set of relevant document IDs representing the ground truth.

### 2. Calculate retrieval metrics

```python
from rag_benchmark_kit import (
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)

retrieved = ["doc_1", "doc_3", "doc_2"]
relevant = {"doc_1", "doc_2"}

# Calculate Recall@2
recall = recall_at_k(retrieved, relevant, k=2)
print("Recall@2:", recall)

# Calculate Reciprocal Rank
rr = reciprocal_rank(retrieved, relevant)
print("Reciprocal Rank:", rr)

# Calculate Mean Reciprocal Rank
mrr = mean_reciprocal_rank(
    retrieved_results=[
        ["doc_1", "doc_3"],
        ["doc_4", "doc_3"],
    ],
    relevant_doc_sets=[
        {"doc_1", "doc_2"},
        {"doc_3"},
    ],
)
print("MRR:", mrr)
```

### 3. Audit an evaluation dataset

```python
from rag_benchmark_kit import audit_dataset

available_doc_ids = {
    "doc_1",
    "doc_2",
    "doc_3",
}

report = audit_dataset(
    examples=examples,
    available_doc_ids=available_doc_ids,
)

print("Total examples:", report.total_examples)
print("Dataset valid:", report.is_valid)

for issue in report.issues:
    print(issue.code, issue.message, issue.example_index)
```

The auditor checks:

| Check | Description |
|---|---|
| Duplicate questions | Identifies questions that match after case and whitespace normalization. |
| Unknown document IDs | Identifies ground-truth document IDs missing from the supplied document manifest. |

A dataset is considered valid when no audit issues are detected.

### 4. Export audit reports

Export the audit report as JSON:

```python
from rag_benchmark_kit import to_json

json_report = to_json(report)

print(json_report)
```

Export the audit report as CSV:

```python
from rag_benchmark_kit import to_csv

csv_report = to_csv(report)

print(csv_report)
```

Both functions return strings. You can write those strings to files using Python's standard file-handling functionality.

## Project Structure

```text
rag-benchmark-kit/
│
├── src/
│   └── rag_benchmark_kit/
│       ├── __init__.py
│       ├── dataset.py
│       ├── metrics.py
│       ├── audit.py
│       └── reports.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_metrics.py
│   ├── test_audit.py
│   └── test_reports.py
│
├── pyproject.toml
├── uv.lock
├── README.md
└── LICENSE
```

## Development Setup

Clone the repository:

```bash
git clone https://github.com/kanishkambk43/Rag-benchmark-kit.git
cd rag-benchmark-kit
```

Install development dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run pytest -v
```

Run lint checks:

```bash
uv run ruff check .
```

Check code formatting:

```bash
uv run ruff format --check .
```

## Technologies

- Python
- pytest
- Ruff
- uv

## Project Status

Version: `0.1.0`

This project is under active development.

