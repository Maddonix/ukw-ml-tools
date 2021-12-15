from pydantic import BaseModel
from typing import Any

class Text(BaseModel):
    text: str

class Token(BaseModel):############################
    value: Any