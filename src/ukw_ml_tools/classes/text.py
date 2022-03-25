from pydantic import BaseModel
from typing import Any, Optional
from .terminology import TerminologyResult


class Text(BaseModel):
    text: str
    terminology_result: Optional[TerminologyResult]

    def __hash__(self):
        return hash(repr(self))


class Token(BaseModel):
    value: Any
