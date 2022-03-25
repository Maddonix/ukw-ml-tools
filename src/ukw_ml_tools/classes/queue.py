from typing import Dict, Union, Optional, List, Any
from bson import ObjectId
from pydantic import BaseModel, Field, NonNegativeInt, validator
from pathlib import Path
from .base import PyObjectId
import numpy as np


class QueueElement(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    image_id: PyObjectId
    intervention_id: PyObjectId
    video_key: Optional[str]
    queue_type: str
    labels: List[str]
    annotation: Optional[List[str]]
    path: Path
    report_text: Optional[str]
    prediction: Optional[List[Any]]
    ai_version: Optional[int]
    archive: bool

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {}
        }

    def get_prediction_value(self):
        indices = np.argwhere(np.array(self.prediction) > 0.5).squeeze().tolist()
        if isinstance(indices, int):
            indices = [indices]
        return indices

    def get_predicted_labels(self):
        indices = self.get_prediction_value()
        labels = [self.labels[i] for i in indices]
        return labels

    def to_dict(self):
        r = self.dict(by_alias=True, exclude_none = True)
        if "path" in r:
            r["path"] = str(r["path"])
        return r
