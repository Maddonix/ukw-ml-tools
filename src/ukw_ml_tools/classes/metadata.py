from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt, validator
from datetime import datetime
from typing import Optional
from pathlib import Path


class ImageMetadata(BaseModel):
    frame_number: Optional[int]
    path: Optional[Path]

    is_frame: bool
    is_extracted: bool

    class Config:
        allow_population_by_field_name = True
        # arbitrary_types_allowed = True
        json_encoders = {Path: str}
        schema_extra = {
            "example": {}
        }

    @validator("is_frame")
    def validate_frame_data(cls, v, values):
        if v:
            assert "frame_number" in values

    @validator("is_extracted")
    def validate_path_if_extracted(cls, v, values):
        if v:
            assert "path" in values


class InterventionMetadata(BaseModel):
    dicomaccessionnumberpseudo: Optional[str]
    dicomstydyinstanceuidpseudo: Optional[str]
    fps: Optional[NonNegativeFloat]
    duration: Optional[NonNegativeFloat]
    frames_total: Optional[NonNegativeInt]
    path: Optional[Path]
    is_video: bool
    intervention_date: Optional[datetime]
    intervention_type: str
    sap_case_id: Optional[int]
    sap_pat_id: Optional[int]
    st_export_id: Optional[int]

    class Config:
        allow_population_by_field_name = True
        # arbitrary_types_allowed = True
        json_encoders = {Path: str}
        schema_extra = {
            "example": {}
        }

    @validator("is_video")
    def validate_video_metadata(cls, v, values):
        if v:
            assert "path" in values
            assert "frames_total" in values
            assert "duration" in values
            assert "fps" in values
        return v

    def to_dict(self):
        r = self.dict()