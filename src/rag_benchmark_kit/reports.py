import csv
import io
import json

from rag_benchmark_kit.audit import AuditReport


def _validate_report(report: AuditReport) -> None:
    """Validate the audit report before serialization."""

    # Ensure the provided object is an AuditReport instance.
    if not isinstance(report, AuditReport):
        raise TypeError("Report must be an AuditReport instance.")


def to_json(report: AuditReport) -> str:
    """Convert an audit report into a JSON string."""

    _validate_report(report)

    report_data = {
        "total_examples": report.total_examples,
        "is_valid": report.is_valid,
        "issues": [
            {
                "code": issue.code,
                "message": issue.message,
                "example_index": issue.example_index,
            }
            for issue in report.issues
        ],
    }

    return json.dumps(report_data, indent=4)


def to_csv(report: AuditReport) -> str:
    """Convert audit issues into a CSV string."""

    _validate_report(report)

    output = io.StringIO()  # This creates an empty text stream in memory.

    writer = (
        csv.DictWriter(  # It allows us to write dictionaries as rows in a CSV file.
            output,
            fieldnames=[
                "code",
                "message",
                "example_index",
            ],
        )
    )

    writer.writeheader()  # This writes the column names into the CSV stream.

    for issue in report.issues:
        writer.writerow(  # writerow() writes one dictionary as one CSV row.
            {
                "code": issue.code,
                "message": issue.message,
                "example_index": issue.example_index,
            }
        )

    return output.getvalue()
