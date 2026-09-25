"""
Schema validation for Laya question bundles.
Supports 'choice', 'score', and 'noul' question types matching Laya and TypeSafe wire formats.
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, model_validator


class ChoiceQuestion(BaseModel):
    type: str = "choice"
    instructions: str
    criteria: Dict[str, str]

    @model_validator(mode="after")
    def check_criteria(self):
        if not self.criteria or len(self.criteria) < 2:
            raise ValueError("Choice questions require at least 2 criteria options.")
        return self


class ScoreQuestion(BaseModel):
    type: str = "score"
    instructions: str
    criteria: List[str]

    @model_validator(mode="after")
    def check_criteria(self):
        if not self.criteria or len(self.criteria) < 2:
            raise ValueError("Score questions require at least 2 levels.")
        return self


class NoulQuestion(BaseModel):
    type: str = "noul"
    instructions: str
    criteria: Optional[Dict[str, str]] = None


QuestionDefinition = Union[ChoiceQuestion, ScoreQuestion, NoulQuestion]


def validate_question_bundle(questions: Dict[str, Any]) -> Dict[str, QuestionDefinition]:
    """Parse and validate a dictionary of Laya questions."""
    parsed = {}
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Questions must be a non-empty dictionary.")

    for qid, qdef in questions.items():
        if not isinstance(qdef, dict):
            raise ValueError(f"Question '{qid}' must be a dictionary definition.")
        qtype = qdef.get("type", "choice").lower()
        if qtype == "choice":
            parsed[qid] = ChoiceQuestion(**qdef)
        elif qtype == "score":
            parsed[qid] = ScoreQuestion(**qdef)
        elif qtype == "noul":
            parsed[qid] = NoulQuestion(**qdef)
        else:
            raise ValueError(f"Unknown question type '{qtype}' for question '{qid}'. Supported: choice, score, noul.")

    return parsed
