from typing import Dict, List, Optional, Any
from bson import ObjectId
from pydantic import BaseModel, Field, NonNegativeInt, validator
import pandas as pd

from .annotation import VideoSegmentationAnnotation
from .polyp import ReportPolypAnnotationResult
from .base import PyObjectId
from .prediction import BoxPrediction, VideoSegmentationPrediction
from .text import Text, Token
from .metadata import InterventionMetadata, InterventionMetadataWü
from pathlib import Path
from .report import ReportAnnotationResult


class Intervention(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")
    video_key: Optional[str]
    origin: str
    id_extern: Optional[int]
    examiners: Optional[List[str]]
    age: Optional[int]
    gender: Optional[int]
    frames: Dict[int, PyObjectId]
    freezes: Dict[int, PyObjectId]
    image_annotations_sum: Dict[str, Dict[str, NonNegativeInt]]
    image_predictions_sum: Dict[str, Dict[str, NonNegativeInt]]
    video_segments_annotation: Optional[Dict[str, VideoSegmentationAnnotation]]
    video_segments_prediction: Optional[Dict[str, VideoSegmentationPrediction]]
    intervention_report_text: Optional[Text]
    intervention_histo_text: Optional[Text]
    intervention_report_structured: Optional[Dict[str, str]]
    intervention_report_id: Optional[int]
    intervention_histology_structured: Optional[Dict[str, str]]
    text_tokens: Optional[Dict[str, List[Token]]]
    polyp_reports: Optional[List[ReportPolypAnnotationResult]]
    metadata: InterventionMetadata
    metadata_wü: Optional[InterventionMetadataWü]
    report_classification: Optional[ReportAnnotationResult]

    def __hash__(self):
        return hash(repr(self))

    class Config:
        allow_population_by_field_name = True
        # arbitrary_types_allowed = True
        json_encoders = {ObjectId: str, InterventionMetadata: dict, Path: str}
        schema_extra = {"example": {}}

    # @validator("age")
    # def age_is_valid(cls, v):
    #     assert v >= 0
    #     return v

    @validator("metadata")
    def validate_fields_by_metadata(cls, v, values):
        if v.is_video:
            assert "video_segments_annotation" in values
            assert "video_segments_prediction" in values
            assert "video_key" in values

        return v

    def to_dict(self):
        r = self.dict(by_alias=True, exclude_none = True)
        if "path" in r["metadata"]:
            r["metadata"]["path"] = str(r["metadata"]["path"])
        r["frames"] = {str(n): _id for n, _id in r["frames"].items()}
        r["freezes"] = {str(n): _id for n, _id in r["freezes"].items()}

        return r

    def create_in_db(self, db_interventions):
        if self.video_key:
            _ = db_interventions.find_one({"video_key": self.video_key})
            assert not _

    def merge_id_extern(self, db_interventions):
        r = db_interventions.update_one({"video_key": self.video_key}, {"$set": {"id_extern": self.id_extern}})
        return r

    def summary(self):
        summary = self.dict(exclude={"frames", "freezes", "text_tokens"})
        summary["n_db_frames"] = len(self.frames)
        summary["n_db_freezes"] = len(self.freezes)
        return summary

    def frame_df(self):
        df = pd.DataFrame.from_records([self.frames]).transpose().reset_index()
        df.columns = ["frame_number", "id"]
        df = df.sort_values("frame_number")
        return df

    def get_cwd_hex(self):
        return hex(int(self.metadata_wü.cwd_intervention_id))[2:].upper()