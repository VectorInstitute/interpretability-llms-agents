"""Unified dataset-agnostic sample representation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

UNANSWERABLE_TOKEN = "UNANSWERABLE"

# Answer strings from the dataset that map to UNANSWERABLE
UNANSWERABLE_ANSWERS = {
    "unanswerable",
    "cannot be determined",
    "not determinable",
    "not answerable",
    "n/a",
    "none",
    "no answer",
}


class QuestionType(str, Enum):
    STANDARD = "standard"
    MCQ = "mcq"
    CONVERSATIONAL = "conversational"
    HYPOTHETICAL = "hypothetical"
    UNANSWERABLE = "unanswerable"


@dataclass
class PerceivedSample:
    sample_id: str
    image_path: str                              # local path to saved image
    question: str
    expected_output: str                         # canonical; UNANSWERABLE_TOKEN for unanswerable
    question_type: QuestionType
    choices: Optional[List[str]] = None          # MCQ option texts
    context: Optional[List[dict]] = None         # [{"role": "user"|"assistant", "content": "..."}]
    metadata: dict = field(default_factory=dict)

    def is_unanswerable(self) -> bool:
        return self.expected_output.strip().upper() == UNANSWERABLE_TOKEN

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "question": self.question,
            "expected_output": self.expected_output,
            "question_type": self.question_type.value,
            "choices": self.choices,
            "context": self.context,
            "metadata": self.metadata,
        }
