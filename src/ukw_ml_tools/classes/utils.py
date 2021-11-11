from pathlib import Path
from collections import defaultdict
from typing import Any, List
from bson.objectid import ObjectId
from pymongo.collection import Collection
from .fieldnames import *
import os
from datetime import datetime as dt
import re
import cv2

def generate_default_dict(template=dict):
    """Helper function returns a defaultdict generating given template"""
    return defaultdict(dict)

def generate_frame_path(video_key:str, base_path_frames: Path) -> Path:
    return base_path_frames.joinpath(video_key)

# Crud
def match_logical_and_aggregation(condition_list: List[str]):
    agg_element = {
        "$match": {
            "$and": [
                _ for _ in condition_list
            ]
        }
    }

    return agg_element

def logical_or_aggregation(condition_list):
    agg_element = {
        "$or": [_ for _ in condition_list]
    }

    return agg_element

def field_exists_query(fieldname: str, exists: bool = True) -> dict:
    return {
        fieldname: {"$exists": exists}
    }

def field_value_query(fieldname: str, value: Any) -> dict:
    return {
        fieldname: value
        }

def fieldvalue_in_list_query(fieldname: str, values: List[Any]) -> dict:
    return {
        fieldname: {"$in": values}
    }

def fieldvalue_nin_list_query(fieldname: str, values: List[Any]) -> dict:
    return {
        fieldname: {"$nin": values}
    }

def get_intervention_db_template(**kwargs):
    
    template = {
        FIELDNAME_EXTERNAL_ID: None,
        FIELDNAME_FRAMES: {},
        FIELDNAME_FREEZES: {},
        FIELDNAME_VIDEO_KEY: None,
        FIELDNAME_VIDEO_PATH: None,
        FIELDNAME_INTERVENTION_TYPE: None,
        FIELDNAME_TOKENS: {
            FIELDNAME_TOKENS_REPORT: [],
            FIELDNAME_TOKENS_PATHO: []
        }
    }

    # if "intervention_dict" in kwargs:
    #     for key, value in kwargs["intervention_dict"]:
    #         if key in template:
    #             template[key]=value

    return template


def get_image_db_template(
    origin: str = None,
    intervention_id: ObjectId = None,
    path: str = None,
    n_frame: int = None,
    image_type: str = None
):
    """
    """

    template = {
        FIELDNAME_ORIGIN: origin,
        FIELDNAME_INTERVENTION_ID: intervention_id,
        FIELDNAME_IMAGE_PATH: path,
        FIELDNAME_FRAME_NUMBER: n_frame,
        FIELDNAME_IMAGE_TYPE: image_type,
        FIELDNAME_LABELS: {},
        FIELDNAME_PREDICTIONS: {},

    }
    return template


def delete_frame_from_intervention(
    frame_dict: dict,
    db_interventions: Collection
    ):
    n_frame = frame_dict[FIELDNAME_FRAME_NUMBER]
    frame_id = frame_dict["_id"]
    intervention_id = frame_dict[FIELDNAME_INTERVENTION_ID]
    frame_path = Path(frame_dict[FIELDNAME_IMAGE_PATH])
    assert frame_path.exists()
    

    intervention = db_interventions.find_one({"_id": intervention_id})
    assert intervention
    assert "frames" in intervention
    assert str(n_frame) in intervention[FIELDNAME_FRAMES]
    assert frame_id is intervention[FIELDNAME_FRAMES]
    db_interventions.update_one(
        {"_id": intervention_id},
        {"$unset": {f"{FIELDNAME_FRAMES}.{n_frame}": ""}}
        )

    os.remove(frame_path)

    return True


def datetime_from_video_key(video_key, re_pattern, dt_pattern):
    date_string = re.search(re_pattern, video_key)
    if date_string:
        date_string = date_string.group()
        intervention_date = dt.strptime(date_string, dt_pattern)

        return intervention_date

def get_video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(path.as_posix())
    fps = get_fps(cap)
    frames_total = get_frames_total(cap)
    
    meta = {
        FIELDNAME_FPS: fps,
        FIELDNAME_FRAMES_TOTAL: frames_total
    }
    
    return meta


def get_frames_total(cap: cv2.VideoCapture) -> int:
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-2
    assert frames_total
    return frames_total


def get_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps
    return fps