from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt, validator
from datetime import datetime
from typing import Optional, List
from pathlib import Path

class ImageMetadata(BaseModel):
    frame_number: Optional[int]
    path: Optional[Path]
    crop: Optional[List[int]]

    is_frame: bool
    is_extracted: bool

    def __hash__(self):
        return hash(repr(self))

    class Config:
        allow_population_by_field_name = True
        # arbitrary_types_allowed = True
        json_encoders = {Path: str}
        schema_extra = {"example": {}}

    @validator("is_frame")
    def validate_frame_data(cls, v, values):
        if v:
            assert "frame_number" in values
        return v

    @validator("is_extracted")
    def validate_path_if_extracted(cls, v, values):
        if v:
            assert "path" in values
        return v

class InterventionMetadataWü(BaseModel):
    icd : Optional[str]
    icpm : Optional[str]
    asa : Optional[str]
    anamnese : Optional[str]
    endoscope : Optional[str]
    premedication : Optional[str]
    colon_max_visibility : Optional[str]
    report_anus : Optional[str]
    report_stomach : Optional[str]
    report_esophagus: Optional[str]
    report_duodenum : Optional[str]
    report_colon  : Optional[str]
    report_colonoscopy : Optional[str]
    report_histology : Optional[str]
    complication : Optional[str]
    evaluation : Optional[str]
    _evaluation : Optional[str]
    colonoscopy_evaluation_final : Optional[str]
    colonoscopy_recommendation : Optional[str]
    colonoscopy_therapy: Optional[str]
    report_gastroscopy : Optional[str]
    leistungserfassung : Optional[str]
    leistungserfassung_dgvs: Optional[str]
    therapy : Optional[str]
    evaluation : Optional[str]
    report_gastroscopy_final : Optional[str]
    recommendation : Optional[str]
    cwd_intervention_id : Optional[int]
    stay_type: Optional[str]
    intervention_start : Optional[datetime]
    intervention_stop : Optional[datetime]
    documented_withdrawal_time: Optional[float]
    intervention_duration : Optional[float]

    def __hash__(self):
        return hash(repr(self))

    class Config:
        allow_population_by_field_name = True
        # arbitrary_types_allowed = True
        json_encoders = {Path: str}
        schema_extra = {"example": {}}


class InterventionMetadata(BaseModel):
    dicomaccessionnumberpseudo: Optional[str]
    dicomstudyinstanceuidpseudo: Optional[str]
    pidpseudo: Optional[int]
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
    crop: Optional[List[int]] # ymin, ymax, xmin, xmax

    def __hash__(self):
        return hash(repr(self))

    class Config:
        allow_population_by_field_name = True
        # arbitrary_types_allowed = True
        json_encoders = {Path: str}
        schema_extra = {"example": {}}

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
