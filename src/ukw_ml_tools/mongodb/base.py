from pymongo import collection
from typing import List, Dict
from bson import ObjectId
from ..fieldnames import *
from collections import defaultdict
from ..classes.intervention import Intervention
from .get_objects import get_intervention
from pathlib import Path

def get_extern_ids(collection):
    _ = collection.aggregate([
        {"$match": {"id_extern": {"$exists": True}}},
        {"$project": {"id_extern": 1}}
    ])
    _ = [_["id_extern"] for _ in _]
    return _

def video_keys(collection):
    _ = collection.aggregate([
        {"$match": {"video_key": {"$exists": True}}},
        {"$project": {"video_key": 1}}
    ])
    _ = [_["id_extern"] for _ in _]
    return _

def get_all_in_batches(collection: collection, limit: int = 10000, aggregation: List = [{"$match": {}}], last_id: ObjectId = None):
    aggregation = aggregation.copy()
    if last_id:
        aggregation.append({"_id": {"$gt": last_id}})
    aggregation.append({"$limit": limit})
    cursor = collection.aggregate(aggregation)

    return cursor


def filter_images_by_frame_diff(img_list: List[dict], min_frame_diff: int = 10):
    # Holy Moly, Bumms ineffizient, muss nachgebessert werden
    filtered_images_dict = defaultdict(list)

    for img in img_list:
        intervention_id = img[IMG_INTERVENTION_ID]

        if not img[METADATA][IMG_IS_FRAME]:
            filtered_images_dict[intervention_id].append(img)
            continue

        append = True
        n_frame = img[METADATA][IMG_FRAME_NUMBER]
        
        if filtered_images_dict[intervention_id]:
            for _img in filtered_images_dict[intervention_id]:
                frame_diff = abs(_img[METADATA][IMG_FRAME_NUMBER] - n_frame)
                if frame_diff < min_frame_diff:
                    append = False
                    break
        
        if append:
            filtered_images_dict[intervention_id].append(img)

    filtered_images = []
    for key, value in filtered_images_dict.items():
        filtered_images.extend(value)
    
    return filtered_images

def filter_existing_frames_by_intervention(video_key, frame_paths, db_interventions) -> Dict[int, Path]:
    intervention = get_intervention(video_key, db_interventions)
    image_paths = {n: path for n, path in frame_paths.items() if n not in intervention.frames}
    return image_paths

def get_base_image_dict_from_intervention(intervention:Intervention):
    base_image_dict = {
        "intervention_id": intervention.id,
        "origin": intervention.origin,
        "video_key": intervention.video_key,
        "predictions": {},
        "annotations": {},
        "metadata": {}
    }
    return base_image_dict