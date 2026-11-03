import csv
import io
import json

import pytest

from rag_benchmark_kit.audit import AuditIssue, AuditReport
from rag_benchmark_kit.reports import to_csv, to_json


def make_report() -> AuditReport:
    return AuditReport(
        total_examples=3,
        issues=(
            AuditIssue(
                code="duplicate_query",
                message="Question duplicates example 0.",
                example_index=1,
            ),
            AuditIssue(
                code="unknown_document_id",
                message="Document ID 'doc_999' does not exist.",
                example_index=2,
            ),
        ),
    )


def test_to_json_returns_valid_json() -> None:
    report = make_report()

    result = to_json(report)
    data = json.loads(result)

    assert data["total_examples"] == 3
    assert data["is_valid"] is False
    assert len(data["issues"]) == 2


def test_to_json_includes_issue_details() -> None:
    report = make_report()

    data = json.loads(to_json(report))

    assert data["issues"][0]["code"] == "duplicate_query"
    assert data["issues"][0]["example_index"] == 1


def test_to_json_handles_valid_report() -> None:
    report = AuditReport(
        total_examples=2,
        issues=(),
    )

    data = json.loads(to_json(report))

    assert data["is_valid"] is True
    assert data["issues"] == []


def test_to_csv_returns_expected_headers() -> None:
    report = make_report()

    result = to_csv(report)
    reader = csv.DictReader(io.StringIO(result))

    assert reader.fieldnames == [
        "code",
        "message",
        "example_index",
    ]


def test_to_csv_includes_all_issues() -> None:
    report = make_report()

    result = to_csv(report)
    rows = list(csv.DictReader(io.StringIO(result)))

    assert len(rows) == 2
    assert rows[0]["code"] == "duplicate_query"
    assert rows[1]["code"] == "unknown_document_id"


def test_to_csv_handles_valid_report() -> None:
    report = AuditReport(
        total_examples=2,
        issues=(),
    )

    result = to_csv(report)
    rows = list(csv.DictReader(io.StringIO(result)))

    assert len(rows) == 0
    assert result.startswith("code,message,example_index")


def test_to_json_rejects_invalid_report() -> None:
    with pytest.raises(TypeError, match="AuditReport"):
        to_json("invalid")


def test_to_csv_rejects_invalid_report() -> None:
    with pytest.raises(TypeError, match="AuditReport"):
        to_csv("invalid")
