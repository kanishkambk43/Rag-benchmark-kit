'''In simple terms, it stores:
1. A question we want to test.
2. The IDs of the documents that are known to be relevant to that question.'''

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationExample:
    """One question and its known relevant document IDs."""

    query: str
    relevant_doc_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("Query must not be empty.")

        if not self.relevant_doc_ids:
            raise ValueError(
                "At least one relevant document ID is required."
            )

        if any(not doc_id.strip() for doc_id in self.relevant_doc_ids):#Is there at least one document ID that is empty or contains only spaces?
            raise ValueError(
                "Relevant document IDs must not be empty."
            )
