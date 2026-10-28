
from collections.abc import Sequence, Set


def _validate_relevant_doc_ids(
    relevant_doc_ids: Set[str],
) -> None:
    """Validate the ground-truth relevant document IDs."""
    if not relevant_doc_ids:
        raise ValueError(
            "At least one relevant document ID is required."
        )

    if any(not doc_id.strip() for doc_id in relevant_doc_ids):
        raise ValueError(
            "Relevant document IDs must not be empty."
        )


def recall_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Set[str],
    k: int,
) -> float:
    """Calculate Recall@k for one query."""
    _validate_relevant_doc_ids(relevant_doc_ids)

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    if any(not doc_id.strip() for doc_id in retrieved_doc_ids):
        raise ValueError("Retrieved document IDs must not be empty.")

    top_k = retrieved_doc_ids[:k]
    relevant_found = set(top_k) & relevant_doc_ids

    return len(relevant_found) / len(relevant_doc_ids)


def reciprocal_rank(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Set[str],
) -> float:
    """Calculate reciprocal rank for one query."""
    _validate_relevant_doc_ids(relevant_doc_ids)

    if any(not doc_id.strip() for doc_id in retrieved_doc_ids):
        raise ValueError("Retrieved document IDs must not be empty.")

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    retrieved_results: Sequence[Sequence[str]],
    relevant_doc_sets: Sequence[Set[str]],
) -> float:
    """Calculate MRR across multiple queries."""
    if not retrieved_results:
        raise ValueError("At least one query result is required.")

    if len(retrieved_results) != len(relevant_doc_sets):
        raise ValueError(
            "Each query result must have a matching relevant document set."
        )

    reciprocal_ranks = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(#zip() pairs the first result with the first relevant set, the second result with the second relevant set.
            retrieved_results, relevant_doc_sets, strict=True
        )
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)

# retrieved_results: Retrieved document IDs for multiple queries.
# relevant_doc_sets: Relevant document IDs for multiple queries.
# retrieved_doc_ids: Retrieved document IDs for a single query.
# relevant_doc_ids: Relevant document IDs for a single query.