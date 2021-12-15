from typing import Dict, List, Optional, Union
from bson import ObjectId
from pydantic import BaseModel, Field, validator
from pathlib import Path
from ..media.metadata import get_video_meta
from ..extern.base import path_to_timestamp,video_extern_exists_intern, video_extern_get_intern_id
from .base import Flank
from datetime import datetime, tzinfo

INTERVENTION_TYPE_MAPPING = {
    "Coloscopy": "Koloskopie",
    "Unknown": "Unknown",
    "Gastroscopy": "Gastroskopie"
}

class VideoExtern(BaseModel):
    id_extern: int = Field(alias="videoId")
    path: Path = Field(alias= "videoPath")
    images_path: Optional[Path] = Field(alias= "imagesPath")
    annotation_paths: Optional[Path]
    video_remark: Optional[str] #= Field(alias="")
    intervention_histo_text: Optional[str] =  Field(alias = "patho")
    intervention_report_text: Optional[str] = Field(alias = "report")
    origin: Optional[str] = Field(alias = "center")
    intervention_type: str = Field(alias="videoType")
    annotated: bool


    class Config:
        # allow_population_by_field_name = True
        json_encoders = {Path: str}
        schema_extra = {"example": {}}

    @validator('intervention_histo_text')
    def rm_whitespace_histo(cls, v) -> str:
        if v:
            v = v.replace("\n", " ")
            v = v.replace("  ", " ")
            v.strip()
            return v

    @validator('intervention_report_text')
    def rm_whitespace_report(cls, v) -> str:
        if v:
            v = v.replace("\n", " ")
            v = v.replace("  ", " ")
            v.strip()
            return v

    def get_video_meta(self):
        return get_video_meta(self.path)

    def map_intervention_type(self):
        _type=self.intervention_type
        assert _type in INTERVENTION_TYPE_MAPPING
        _type = INTERVENTION_TYPE_MAPPING[_type]
        return _type

    def exists_intern(self, db_interventions):
        return video_extern_exists_intern(self.id_extern, db_interventions)

    def get_id_intern(self, db_interventions):
        _id = video_extern_get_intern_id(self.id_extern, db_interventions)
        if _id: return _id

    def to_intervention_dict(self, db_interventions, time_format = "%Y-%m-%d_%H-%M-%S"):
        intervention_dict = self.dict()
        id_intern = self.get_id_intern(db_interventions)
        if id_intern:
            intervention_dict["id"] = id_intern
        intervention_dict["frames"] = {}
        intervention_dict["freezes"] = {}
        intervention_dict["image_annotations_sum"] = {}
        intervention_dict["image_predictions_sum"] = {}
        intervention_dict["video_key"] = self.path.name
        if self.intervention_histo_text: 
            intervention_dict["intervention_histo_text"] = {"text": self.intervention_histo_text}
        if self.intervention_report_text:
            intervention_dict["intervention_report_text"] = {"text": self.intervention_report_text}


        metadata_dict = self.get_video_meta()
        metadata_dict["path"] = self.path
        metadata_dict["is_video"] = True
        metadata_dict["intervention_date"] = path_to_timestamp(self.path, time_format, self.origin)
        metadata_dict["intervention_type"] = self.map_intervention_type()

        intervention_dict["metadata"] = metadata_dict

        return intervention_dict

class ExternFlank(BaseModel):
    name: str = Field(alias = "annotationName")
    value: Union[bool, str] = Field(alias = "annotationValue")
    start: int = Field(alias = "startFrame")
    stop: int = Field(alias = "endFrame")

    @validator("name")
    def name_to_lower(cls, v):
        return v.lower()

class ExternVideoFlankAnnotation(BaseModel):
    extern_annotator_id: int = Field(alias = "userId")
    id_extern: int = Field(alias = "videoId")
    video_key: str = Field(alias = "videoName")
    date: datetime = Field(alias = "sessionDate")
    label_group_id: int = Field(alias = "labelGroupId")
    label_group_name: str = Field(alias = "labelGroupName")
    flanks: List[ExternFlank] = Field(alias = "sequences")

    @validator("date")
    def rm_tz(cls, v):
        v = v.replace(microsecond=0)
        return v

class ExternAnnotatedVideo(BaseModel):
    id_extern: int = Field (alias = "videoId")
    video_key: str = Field(alias = "videoName")
    date: datetime = Field(alias = "lastAnnotationSession")

    @validator("date")
    def rm_tz(cls, v):
        v = v.replace(tzinfo=None)
        v = v.replace(microsecond=0)
        return v
