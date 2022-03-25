from typing import Dict, Union, Optional, List
from bson import ObjectId
from pydantic import BaseModel, Field, NonNegativeInt, validator

from .annotation import BinaryAnnotation, MultilabelAnnotation, MultichoiceAnnotation
from .base import PyObjectId
from .prediction import BinaryPrediction, MultilabelPrediction, MultichoicePrediction, BoxPrediction
from .metadata import ImageMetadata
from .text import Text

class Image(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    intervention_id: PyObjectId = Field(default_factory=PyObjectId)
    origin: str
    video_key: Optional[str]
    predictions: Dict[str, Union[BinaryPrediction, MultilabelPrediction, MultichoicePrediction]]
    predictions_smooth: Optional[Dict[str, Union[BinaryPrediction, MultilabelPrediction, MultichoicePrediction]]]
    annotations: Dict[str, Union[BinaryAnnotation, MultilabelAnnotation, MultichoiceAnnotation]]
    image_caption: Optional[Text]
    metadata: ImageMetadata
    box_detections: Optional[List[PyObjectId]] = []

    def __hash__(self):
        return hash(repr(self))

    @validator("metadata")
    def validate_metadata(cls, v, values):
        if v.is_frame:
            assert "video_key" in values
        if v.is_extracted:
            assert v.path != None
        return v

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str, ImageMetadata: dict}
        schema_extra = {
            "example": {}
        }

    def to_dict(self):
        r = self.dict(by_alias=True, exclude_none = True)
        if "path" in r["metadata"]:
            r["metadata"]["path"] = str(r["metadata"]["path"])
        return r
