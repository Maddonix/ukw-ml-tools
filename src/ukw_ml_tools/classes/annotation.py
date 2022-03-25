from pydantic import (
    BaseModel,
    NonNegativeInt,
    validator
)
from typing import List, Any, Optional
from datetime import datetime
from .base import Flank

class Annotation(BaseModel):
    source: str #constr(regex=r"")
    annotator_id: NonNegativeInt
    date: datetime
    name: str
    instance_id: Optional[int]

    def __hash__(self):
        return hash(repr(self))

    @validator("date")
    def rm_ms(cls, v):
        v = v.replace(microsecond=0)
        return v

class BinaryAnnotation(Annotation):
    value: bool
    choices: List[bool]
    label_type = "binary"

class MultilabelAnnotation(Annotation):
    value: NonNegativeInt
    choices: List[str]
    label_type = "multilabel"

class MultichoiceAnnotation(Annotation):
    value: List[NonNegativeInt]
    choices: List[str]
    label_type = "multichoice"

class VideoSegmentationAnnotation(Annotation):
    value: List[Flank]
    label_type = "video_segmentation"
