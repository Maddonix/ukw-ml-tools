from pymongo.collection import Collection
import warnings
from typing import List
from pathlib import Path
import os
import cv2


# get information about db
def get_origin_count(db_collection) -> List:
    r = db_collection.aggregate([{"$group": {"_id": "$origin", "count": {"$sum": 1}}}])
    r = [_ for _ in r]
    return r


def get_label_types(db_images):
    _labels = db_images.distinct("labels.annotations")
    labels = []
    for _ in _labels:
        labels.extend(list(_.keys()))
    labels = list(set(labels))
    return labels


def get_prelabel_types(db_images):
    """
    Expects db image collection. Queries for all prediction types and returns list of unique values.
    """
    _labels = db_images.distinct("labels.predictions")
    labels = []
    for _ in _labels:
        labels.extend(list(_.keys()))
    labels = list(set(labels))
    return labels


def get_categorical_label_count(label: str, db_images):
    if label == "polyp_detection_bbox":
        return None
    r = db_images.aggregate(
        [{"$group": {"_id": f"$labels.annotations.{label}", "count": {"$sum": 1}}}]
    )
    r = [_ for _ in r if _["_id"] is not None]
    return r


def get_predictions_without_annotations_query(label: str):
    """
    Expects a label. Returns query dict for images with predictions but not annotations for this label.
    Additionally filters images out if they are already marked as "in_progress".
    !!! Only works for binary classification !!!
    """
    return {
        "$match": {
            f"labels.predictions.{label}": {"$exists": True},
            f"labels.annotation.{label}": {"$exists": False},
            "in_progress": False,
        }
    }


def get_images_to_prelabel_query(label: str, version: float, limit: int = 100000):
    return [
        {
            "$match": {
                "$and": [
                    {
                        "$or": [
                            {f"labels.predictions.{label}": {"$exists": False}},
                            {f"labels.predictions.{label}.version": {"$lt": version}},
                        ]
                    },
                    {f"labels.annotation.{label}": {"$exists": False}},
                ]
            }
        },
        {"$limit": limit},
    ]


def exctract_frame_list(
    video_key: str, frame_list: List[int], base_path_frames: Path, db_interventions: str
):
    """Function to extract frames.

    Args:
        video_key (str): [description]
        frame_list (List[int]): [description]
        base_path_frames (Path): [description]
        db_interventions (str): [description]
    """
    intervention = db_interventions.find_one({"video_key": video_key})
    assert base_path_frames.exists()

    if not intervention:
        warnings.warn(f"Intervention with video_key {video_key} does not exist")

    frames_path = base_path_frames.joinpath(video_key)

    if not frames_path.exists():
        os.mkdir(frames_path)

    video_path = Path(intervention["video_path"])
    cap = cv2.VideoCapture(video_path.as_posix())

    for n_frame in frame_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(frames_path.joinpath(f"{n_frame}.png"), frame)


def get_images_in_progress(db_images):
    return db_images.find({"in_progress": True})


# DEPRECIATE


# def get_images_to_prelabel(
#     prelabel_type: str, version: float, db_collection: str, batchsize: int = 0
# ):
#     agg = [
#         {
#             "$match": {
#                 "$or": [
#                     {f"labels.predictions.{prelabel_type}": {"$exists": False}},
#                     {f"labels.predictions.{prelabel_type}.version": {"$lt": version}},
#                 ]
#             }
#         }
#     ]
#     if batchsize > 0:
#         agg.append({"$limit": batchsize})

#     return db_collection.aggregate(agg)


# def get_intervention_for_image_id(
#     image_id, db_images: Collection, db_interventions: Collection
# ):
#     r = db_images.find_one({"_id": image_id})

#     if r:
#         intervention_id = r["intervention_id"]
#         intervention = db_interventions.find_one({"_id": intervention_id})

#         return intervention

#     else:
#         warnings.warn(f"No image found for id {image_id}")


# intervention_has_image_paths = {"$match": {"image_paths.1": {"$exists": True}}}


# def get_intervention_text_match_query(
#     keyword_list, intervention_type: str = "Koloskopie"
# ):
#     text_query = {
#         "image_paths.1": {"$exists": True},
#         "intervention_type": intervention_type,
#         "$text": {
#             "$search": " ".join(keyword_list),
#             "$language": "de",
#             "$caseSensitive": False,
#         },
#     }

#     agg = [
#         {"$match": text_query},
#         {
#             "$lookup": {
#                 "from": "images",
#                 "localField": "image_ids",
#                 "foreignField": "_id",
#                 "as": "image_objects",
#             }
#         },
#         {
#             "$project": {
#                 "report": True,
#                 "image_objects": True,
#                 "intervention_date": True,
#                 "age": True,
#                 "gender": True,
#                 "origin": True,
#                 "intervention_type": True,
#             }
#         },
#     ]

#     return agg
