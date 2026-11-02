"""
Audits the RAG evaluation dataset for duplicate questions and invalid ground-truth document IDs.
Generates a structured audit report containing all detected issues and their corresponding example indices.
"""
#A ground-truth document ID is the ID of a document that is already known to contain the correct information for a particular evaluation question.
from collections.abc import AbstractSet, Sequence
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

    total_examples: int    #Total number of evaluation examples checked.
    issues: tuple[AuditIssue, ...]  #All the problems discovered during auditing.

    @property #It allows us to access a method like an attribute, without using parentheses.
    def is_valid(self) -> bool:
        """Return True when no audit issues were found."""
        return not self.issues #returns true if no issues were found.


def _normalize_query(query: str) -> str:
    """Normalize a query for duplicate detection."""
    return " ".join(query.casefold().split())


def audit_dataset(
    examples: Sequence[EvaluationExample],
    available_doc_ids: AbstractSet[str],
) -> AuditReport:
    """Check duplicate questions and invalid document references."""

    issues: list[AuditIssue] = [] #Stores all problems discovered during auditing.
    seen_queries: dict[str, int] = {} #This dictionary helps us identify duplicate questions.

    for index, example in enumerate(examples):

        normalized_query = _normalize_query(example.query)

        if normalized_query in seen_queries:
            first_index = seen_queries[normalized_query]

            issues.append(
                AuditIssue(
                    code="duplicate_query",
                    message=f"Question duplicates example {first_index}.",
                    example_index=index,
                )
            )

        else:
            seen_queries[normalized_query] = index

        missing_doc_ids = example.relevant_doc_ids - available_doc_ids

        for doc_id in sorted(missing_doc_ids):
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

    return AuditReport(
        total_examples=len(examples),
        issues=tuple(issues),
    )
