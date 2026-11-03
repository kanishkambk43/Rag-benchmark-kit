"""A lightweight toolkit for benchmarking and auditing RAG evaluation datasets."""

from rag_benchmark_kit.audit import (
    AuditIssue,
    AuditReport,
    audit_dataset,
)
from rag_benchmark_kit.dataset import EvaluationExample
from rag_benchmark_kit.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from rag_benchmark_kit.reports import (
    to_csv,
    to_json,
)

__all__ = [
    "AuditIssue",
    "AuditReport",
    "EvaluationExample",
    "audit_dataset",
    "mean_reciprocal_rank",
    "recall_at_k",
    "reciprocal_rank",
    "to_csv",
    "to_json",
]
