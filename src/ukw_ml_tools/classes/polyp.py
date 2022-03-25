from bson import ObjectId
from pydantic import (
    BaseModel,
    Field,
    validator
)
from typing import Union, Dict, Optional
from .base import PyObjectId
from .report import ReportPolypAnnotationResult

class Polyp(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    intervention_id: PyObjectId
    video_key: Optional[str]
    images: Dict[int, PyObjectId]
    report: Optional[ReportPolypAnnotationResult]
    intervention_instance: int
    instance_type: str = "polyp"
    

    def __hash__(self):
        return hash(repr(self))
        
    def to_dict(self):
        r = self.dict(by_alias=True, exclude_none = True)
        r["images"] = {str(n): _id for n, _id in r["images"].items()}

        return r

