from pydantic import BaseModel, validator
from typing import List, Optional, Union
from .base import PyObjectId
from .utils import POLYP_EVALUATION_STRUCTURE, split_size_str



class ReportPolypAnnotationResult(BaseModel):
    location_segment: Optional[str]
    location_cm: Optional[int]
    size_category: Optional[str]
    size_mm: Optional[int]
    surface_intact: Union[bool, str]
    rating: Optional[str]
    paris: Optional[List[str]]
    dysplasia: Optional[str]
    histo: Optional[str]
    morphology: Optional[str]
    nice: Optional[str]
    lst: Optional[str]
    non_lifting_sign: Union[bool, str] = "unknown"
    injection: Union[bool, str]

    tool: Optional[str]
    resection_technique: Optional[str]
    resection_status_microscopic: Optional[str]
    salvage: Optional[Union[bool, str]] = "unknown"

    apc_watts: Optional[int]
    number_clips: Optional[int]

    ectomy_wound_care_success: Optional[str]
    ectomy_wound_care_technique: Optional[str]
    ectomy_wound_care: Optional[Union[bool, str]]

    no_resection_reason: Optional[str]

    resection: bool

    polyp_id: Optional[PyObjectId]

    def __hash__(self):
        return hash(repr(self))

    @validator("paris", allow_reuse = True)
    def val_paris(cls, v):
        if v == []:
            v = None
        return v
    
    @validator("size_mm", allow_reuse=True)
    def val_size_mm(cls, v, values):
        if v:
            if v < 0:
                v = None
            else:
                _range = split_size_str(values["size_category"])
                assert _range
                assert _range[0] <= v <= _range[1]

        return v

    @validator("number_clips", "apc_watts", allow_reuse = True)
    def val_int(cls, v):
        if v:
            if v < 0: v = None
            return v      

    @validator("resection", allow_reuse = True)
    def val_resection(cls, v, values):
        req_resection = [
            "resection_technique",
            "resection_status_microscopic",
            "salvage",
            "ectomy_wound_care" # if true, ectomy_wound_care_success
        ]

        req_no_resection = [
            "no_resection_reason"
        ]

        if v:
            for req in req_resection:
                assert req in values

        elif v is False:
            for req in req_no_resection:
                assert req in values

        return v

    @validator("ectomy_wound_care", allow_reuse = True)
    def val_ectomy_wound_care(cls, v, values):
        req_wound_care = ["ectomy_wound_care_success", "ectomy_wound_care_technique"]
        if v:
            for req in req_wound_care:
                assert req in values

        return v

    @validator("ectomy_wound_care_technique", allow_reuse=True)
    def val_ectomy_technique(cls, v, values):
        if v == "apc":
            assert "apc_watts" in values
        elif v == "clip":
            assert "number_clips" in values
        return v

    def evaluate(self):
        polyp_report = self.dict()
        result = {
            "required": [],
            "optional": [],
            "found": []
        }

        for attribute, element in POLYP_EVALUATION_STRUCTURE.items():
            if element["required"]:
                result["required"].append(attribute)

            else:
                required = False
                for requirement in element["required_if"]:
                    required_attribute = requirement["attribute"]
                    if polyp_report[required_attribute] in requirement["values"]:
                        required = True
                        break
                if required:    
                    result["required"].append(attribute)
                else:
                    result["optional"].append(attribute)

        for key, value in polyp_report.items():
            if not value == None and not value == "unknown" and not value == -1:
                result["found"].append(key)

        result["required_found"] = [_ for _ in result["required"] if _ in result["found"]]
        result["optional_found"] = [_ for _ in result["optional"] if _ in result["found"]]

        result["required_missing"] = [_ for _ in result["required"] if _ not in result["found"]]
        result["optional_missing"] = [_ for _ in result["optional"] if _ not in result["found"]]

        return result

class ReportAnnotationResult(BaseModel):
    polyps: List[ReportPolypAnnotationResult]
    intervention_time: Optional[int]
    withdrawal_time: Optional[int]
    sedation: str
    bbps_worst: Optional[int]
    bbps_total: Optional[int]
    n_polyps: int
    n_adenoma: Optional[int]
    other_pathologies: bool
    indication: str
    mark_other: Optional[bool]

    def __hash__(self):
        return hash(repr(self))

    @validator("polyps", allow_reuse = True, pre = True)
    def dict_to_list(cls, v):
        if isinstance(v, dict):
            v = [__ for _, __ in v.items()]
        return v

    @validator(
        "intervention_time",
        "withdrawal_time",
        "bbps_worst",
        "bbps_total",
        "n_polyps",
        allow_reuse = True)
    def del_unknown(cls, v):
        if v:
            if v < 0: v = None
        return v

    @validator("bbps_worst", allow_reuse = True)
    def bbps_worst_range(cls, v):
        if v:
            assert v <= 3
        return v

    @validator("bbps_total", allow_reuse = True)
    def bbps_total_range(cls, v):
        if v:
            assert v <= 9
        return v

    @validator("n_adenoma", allow_reuse = True)
    def set_n_adenoma(cls, v, values):
        return v
