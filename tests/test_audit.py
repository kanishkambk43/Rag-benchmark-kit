
from rag_benchmark_kit.audit import audit_dataset
from rag_benchmark_kit.dataset import EvaluationExample


def make_example(
    query: str,
    *doc_ids: str,
) -> EvaluationExample:
    return EvaluationExample(
        query=query,
        relevant_doc_ids=frozenset(doc_ids),
    )


def test_valid_dataset_has_no_issues() -> None:
    examples = [
        make_example("What is Redis?", "doc_1"),
        make_example("What is caching?", "doc_2"),
    ]

    report = audit_dataset(examples, {"doc_1", "doc_2"})

    assert report.is_valid
    assert report.total_examples == 2
    assert report.issues == ()


def test_detects_duplicate_questions() -> None:
    examples = [
        make_example("What is Redis?", "doc_1"),
        make_example(" what IS redis? ", "doc_1"),
    ]

    report = audit_dataset(examples, {"doc_1"})

    assert not report.is_valid
    assert any(
        issue.code == "duplicate_query"
        for issue in report.issues
    )


def test_detects_unknown_document_reference() -> None:
    examples = [
        make_example("What is Redis?", "doc_999"),
    ]

    report = audit_dataset(examples, {"doc_1"})

    assert not report.is_valid
    assert any(
        issue.code == "unknown_document_id"
        for issue in report.issues
    )


def test_empty_dataset_is_valid() -> None:
    report = audit_dataset([], set())

    assert report.is_valid
    assert report.total_examples == 0


def test_reports_multiple_missing_document_ids() -> None:
    examples = [
        make_example("What is Redis?", "doc_10", "doc_20"),
    ]

    report = audit_dataset(examples, {"doc_1"})

    missing_issues = [
        issue
        for issue in report.issues
        if issue.code == "unknown_document_id"
    ]

    assert len(missing_issues) == 2
