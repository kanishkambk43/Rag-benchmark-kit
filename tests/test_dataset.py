import pytest

from rag_benchmark_kit.dataset import EvaluationExample


def test_create_valid_example() -> None:
    example = EvaluationExample(
        query="What is Redis caching?",
        relevant_doc_ids=frozenset({"doc_101", "doc_103"}),
    )

    assert example.query == "What is Redis caching?"
    assert example.relevant_doc_ids == {"doc_101", "doc_103"}


def test_reject_empty_query() -> None:
    with pytest.raises(ValueError, match="Query must not be empty"):
        EvaluationExample(
            query="   ",
            relevant_doc_ids=frozenset({"doc_101"}),
        )


def test_reject_empty_relevant_documents() -> None:
    with pytest.raises(ValueError, match="At least one relevant"):
        EvaluationExample(
            query="What is Redis caching?",
            relevant_doc_ids=frozenset(),
        )


def test_reject_empty_document_id() -> None:
    with pytest.raises(ValueError, match="document IDs must not be empty"):
        EvaluationExample(
            query="What is Redis caching?",
            relevant_doc_ids=frozenset({" "}),
        )


def test_rejects_non_string_query() -> None:
    with pytest.raises(TypeError, match="Query must be a string"):
        EvaluationExample(
            query=123,
            relevant_doc_ids=frozenset({"doc_1"}),
        )


def test_rejects_non_frozenset_document_ids() -> None:
    with pytest.raises(TypeError, match="must be a frozenset"):
        EvaluationExample(
            query="What is Redis?",
            relevant_doc_ids=["doc_1"],
        )


def test_rejects_non_string_document_id() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        EvaluationExample(
            query="What is Redis?",
            relevant_doc_ids=frozenset({"doc_1", 123}),
        )
