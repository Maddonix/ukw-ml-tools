from typing import Dict, List
from ..classes.annotation import VideoSegmentationAnnotation
from bson import ObjectId
from .get_objects import get_intervention
from ..classes.image import Image
from ..classes.intervention import Intervention
from pymongo.collection import Collection


def update_video_segments_annotation(
    video_key,
    video_segmentation: Dict[str, VideoSegmentationAnnotation],
    db_interventions
):
    video_segmentation = {key: value.dict() for key, value in video_segmentation.items()}
    db_images = db_interventions.database.Images

    for key in video_segmentation.keys():
        db_images.update_many({"video_key": video_key, f"annotations.{key}.source": "web_annotation_flanks"}, {"$unset": {"annotations.{key}":""}})

    r = db_interventions.update_one({"video_key": video_key},
                                    {"$set": {"video_segments_annotation": video_segmentation}})

    return r


def insert_frame(image: Image, db_images: Collection):
    assert not image.id
    r = db_images.insert_one(image.to_dict())
    return r


def insert_new_intervention(intervention: Intervention, db_interventions: Collection):
    assert not intervention.id
    r = db_interventions.insert_one(intervention.to_dict())
    return r


def add_frames_to_framedict(video_key, inserted_ids: Dict[int, ObjectId], db_interventions: Collection):
    intervention = get_intervention(video_key, db_interventions)

    for n_frame, _id in inserted_ids.items():
        assert n_frame not in intervention.frames
        db_interventions.update_one({"video_key": video_key}, {"$set": {f"frames.{n_frame}": _id}})


def update_train_dataset(name, train_set_db, db_train_data):
    db_train_data.update_one({"name": name}, {"$set": train_set_db.dict()}, upsert=True)


def update_many_frame_is_extracted(value: bool, id_list: List[ObjectId], db_images):
    db_images.update_many({"_id": {"$in": id_list}}, {"$set": {"metadata.is_extracted": value}})


def update_terminology_result(result, identifier, _type, db_interventions, id_is_key=True):
    """type must be one of report, histo

    Args:
        result ([type]): [description]
        identifier ([type]): [description]
        _type ([type]): [description]
        db_interventions ([type]): [description]
        id_is_key (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if id_is_key:
        param = "video_key"
    else:
        param = "_id"
    if _type == "histo":
        field = "intervention_histo_text.terminology_result"
    elif _type == "report":
        field = "intervention_report_text.terminology_result"
    else:
        raise Exception
    r = db_interventions.update_one(
        {param: identifier},
        {"$set": {field: result.dict(by_alias=True)}}
    )
    return r
