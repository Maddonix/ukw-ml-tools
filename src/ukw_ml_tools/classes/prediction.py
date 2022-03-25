from pydantic import (
    BaseModel,
    NonNegativeInt,
    Field
)
from typing import List, Optional, Union
from datetime import datetime

from .base import Flank, SettingsSmoothing, PyObjectId

class Prediction(BaseModel):
    name: str
    version: float
    date: Optional[datetime]

    def __hash__(self):
        return hash(repr(self))

class BinaryPrediction(Prediction):
    value: bool
    raw: float
    choices: List[Union[bool, int]]
    label_type = "binary"

class MultilabelPrediction(Prediction):
    value: NonNegativeInt
    raw: List[float]
    choices: List[str]
    label_type = "multilabel"

    def to_binary_predictions(self):
        binary_predictions = []
        for i, choice in enumerate(self.choices):
            prediction_dict = self.dict(include = {
                "version",
                "date"
            })
            prediction_dict["name"] = choice
            prediction_dict["value"] = self.value == i
            prediction_dict["raw"] = self.raw[i]
            prediction_dict["choices"] = [False, True]
            binary_predictions.append(BinaryPrediction(**prediction_dict))
        return binary_predictions

class MultichoicePrediction(Prediction):
    value: List[NonNegativeInt]
    raw: List[float]
    choices: List[str]
    label_type = "multichoice"


class VideoSegmentationPrediction(Prediction):
    value: List[Flank]
    name: str
    settings_smoothing: SettingsSmoothing
    label_type = "video_segmentation"

class BoxPrediction(Prediction):
    id: Optional[PyObjectId] = Field(alias="_id")
    x: int
    y: int
    width: int
    height: int
    score: float
    value: Optional[bool]
    image_id: PyObjectId
    intervention_id: PyObjectId
    instance_id: Optional[PyObjectId]
    frame_number: Optional[int]

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {}
        }

    