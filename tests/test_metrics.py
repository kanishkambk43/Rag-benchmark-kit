
import pytest

from rag_benchmark_kit.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)

# -------------------------
# Tests for Recall@k
# -------------------------

def test_recall_at_k_all_relevant_documents_found() -> None:
    retrieved = ["doc_1", "doc_2", "doc_3"]
    relevant = {"doc_1", "doc_2"}

    result = recall_at_k(retrieved, relevant, k=3)

    assert result == 1.0


def test_recall_at_k_some_relevant_documents_found() -> None:
    retrieved = ["doc_1", "doc_3", "doc_2"]
    relevant = {"doc_1", "doc_2"}

    result = recall_at_k(retrieved, relevant, k=2)

    assert result == 0.5


def test_recall_at_k_no_relevant_documents_found() -> None:
    retrieved = ["doc_3", "doc_4"]
    relevant = {"doc_1", "doc_2"}

    result = recall_at_k(retrieved, relevant, k=2)

    assert result == 0.0


def test_recall_at_k_respects_k_limit() -> None:
    retrieved = ["doc_3", "doc_1"]
    relevant = {"doc_1"}

    result = recall_at_k(retrieved, relevant, k=1)

    assert result == 0.0


def test_recall_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k(["doc_1"], {"doc_1"}, k=0)


def test_recall_at_k_rejects_empty_relevant_documents() -> None:
    with pytest.raises(ValueError, match="At least one relevant"):
        recall_at_k(["doc_1"], set(), k=1)


def test_recall_at_k_rejects_boolean_k() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k(["doc_1"], {"doc_1"}, k=True)


# -------------------------
# Tests for Reciprocal Rank
# -------------------------

def test_reciprocal_rank_first_result_is_relevant() -> None:
    result = reciprocal_rank(
        ["doc_1", "doc_2", "doc_3"],
        {"doc_1"},
    )

    assert result == 1.0


def test_reciprocal_rank_second_result_is_relevant() -> None:
    result = reciprocal_rank(
        ["doc_3", "doc_1", "doc_2"],
        {"doc_1"},
    )

    assert result == 0.5


def test_reciprocal_rank_no_relevant_result() -> None:
    result = reciprocal_rank(
        ["doc_3", "doc_4"],
        {"doc_1"},
    )

    assert result == 0.0


def test_reciprocal_rank_rejects_empty_relevant_documents() -> None:
    with pytest.raises(ValueError, match="At least one relevant"):
        reciprocal_rank(["doc_1"], set())


# -------------------------
# Tests for Mean Reciprocal Rank
# -------------------------

def test_mean_reciprocal_rank() -> None:
    retrieved_results = [
        ["doc_1", "doc_2"],  # Relevant document at rank 1
        ["doc_3", "doc_2"],  # Relevant document at rank 2
        ["doc_4", "doc_5"],  # No relevant document
    ]

    relevant_doc_sets = [
        {"doc_1"},
        {"doc_2"},
        {"doc_6"},
    ]

    result = mean_reciprocal_rank(
        retrieved_results,
        relevant_doc_sets,
    )

    # MRR = (1/1 + 1/2 + 0) / 3 = 0.5
    assert result == pytest.approx(0.5)


def test_mean_reciprocal_rank_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="At least one query"):
        mean_reciprocal_rank([], [])


def test_mean_reciprocal_rank_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="matching relevant"):
        mean_reciprocal_rank(
            [["doc_1"], ["doc_2"]],
            [{"doc_1"}],
        )


# -------------------------
# Input validation tests
# -------------------------

def test_recall_at_k_rejects_empty_retrieved_document_id() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        recall_at_k([""], {"doc_1"}, k=1)


def test_reciprocal_rank_rejects_empty_retrieved_document_id() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        reciprocal_rank([""], {"doc_1"})
