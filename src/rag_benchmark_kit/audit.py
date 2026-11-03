"""
Audits the RAG evaluation dataset for duplicate questions and invalid ground-truth document IDs.
Generates a structured audit report containing all detected issues and their corresponding example indices.
"""

from collections.abc import Sequence, Set
from dataclasses import dataclass

from rag_benchmark_kit.dataset import EvaluationExample


@dataclass(frozen=True)
class AuditIssue:
    """A problem discovered during dataset auditing."""

    code: str
    message: str
    example_index: int


@dataclass(frozen=True)
class AuditReport:
    """Results produced by a dataset audit."""

    total_examples: int  # Total number of evaluation examples checked.
    issues: tuple[AuditIssue, ...]  # All the problems discovered during auditing.

    @property  # Allows us to access a method like an attribute, without using parentheses.
    def is_valid(self) -> bool:
        """Return True when no audit issues were found."""
        return not self.issues  # Returns True if no issues were found.


def _normalize_query(query: str) -> str:
    """Normalize a query for duplicate detection."""

    # casefold() converts text to lowercase for case-insensitive comparison.
    # split() removes extra spaces and handles leading/trailing whitespace.
    # join() combines the words using a single space.
    return " ".join(query.casefold().split())


def audit_dataset(
    examples: Sequence[EvaluationExample],
    available_doc_ids: Set[str],
) -> AuditReport:
    """Check duplicate questions and invalid document references."""

    # Validate that examples is a sequence and not a string or bytes.
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("Examples must be a sequence.")

    # Validate that available document IDs are provided as a set.
    if not isinstance(available_doc_ids, Set):
        raise TypeError("Available document IDs must be a set.")

    # Every document ID in the manifest must be a string.
    if any(not isinstance(doc_id, str) for doc_id in available_doc_ids):
        raise TypeError("Every available document ID must be a string.")

    # Document IDs must not contain only whitespace.
    if any(not doc_id.strip() for doc_id in available_doc_ids):
        raise ValueError("Available document IDs must not be empty.")

    # Ensure every item in examples is an EvaluationExample object.
    if any(not isinstance(example, EvaluationExample) for example in examples):
        raise TypeError("Every example must be an EvaluationExample instance.")

    issues: list[AuditIssue] = []  # Stores all problems discovered during auditing.

    seen_queries: dict[str, int] = {}  # Helps identify duplicate questions.
    # Dictionary format: {normalized_question: first_example_index}

    for index, example in enumerate(examples):
        # Normalize the question before checking for duplicates.
        normalized_query = _normalize_query(example.query)

        # Check whether this normalized question has already appeared.
        if normalized_query in seen_queries:
            # Retrieve the index of the first occurrence of this question.
            first_index = seen_queries[normalized_query]

            # Record the duplicate question as an audit issue.
            issues.append(
                AuditIssue(
                    code="duplicate_query",
                    message=f"Question duplicates example {first_index}.",
                    example_index=index,
                )
            )

        else:
            # Store the first occurrence of the question and its index.
            seen_queries[normalized_query] = index

        # Find relevant document IDs that are missing from the supplied manifest.
        missing_doc_ids = example.relevant_doc_ids - available_doc_ids

        # Check every missing document ID.
        for doc_id in sorted(missing_doc_ids):
            # Record each missing document ID as a separate audit issue.
            issues.append(
                AuditIssue(
                    code="unknown_document_id",
                    message=(
                        f"Relevant document ID {doc_id!r} "
                        "does not exist in the supplied manifest."
                    ),
                    example_index=index,
                )
            )

    # Create and return the final audit report.
    return AuditReport(
        total_examples=len(examples),  # Total number of examples checked.
        issues=tuple(issues),  # Convert the issue list into an immutable tuple.
    )
