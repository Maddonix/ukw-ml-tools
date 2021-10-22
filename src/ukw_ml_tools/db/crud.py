# from pymongo.collection import Collection
import warnings
from typing import List
from pathlib import Path
import os
from bson.objectid import ObjectId
import cv2
import pandas as pd


def get_binary_train_df(label, neg_label_list, db_images):
    # Positive Images
    agg = [
        {
            "$match": {
                f"labels_new.{label}": True
            }
        }
    ]
    imgs_pos = [_["path"] for _ in db_images.aggregate(agg)]
    n_pos = len(imgs_pos)
    imgs_pos_df = {"file_path": [_ for _ in imgs_pos], "label": [1 for _ in imgs_pos]}

    # Negative Images
    imgs_neg = [_["path"] for _ in db_images.find({f"labels_new.{label}": False})]

    for neg_label, multiplier in neg_label_list:
        agg = [
            {
                "$match": {
                    f"labels_new.{neg_label}": True
                }
            },
            {
                "$limit": n_pos*multiplier
            }
        ]
        _imgs_neg = [_["path"] for _ in db_images.aggregate(agg)]
        imgs_neg.extend(_imgs_neg)

    imgs_neg_df = {"file_path": [_ for _ in imgs_neg], "label": [0 for _ in imgs_neg]}

    df_dict = imgs_pos_df
    df_dict["file_path"].extend(imgs_neg_df["file_path"])
    df_dict["label"].extend(imgs_neg_df["label"])

    label_df = pd.DataFrame.from_dict(df_dict)
    return label_df


def get_image_db_template(origin: str = None, intervention_id: ObjectId = None, path: str = None, n_frame: int = None):
    template = {
        "origin": origin,
        "intervention_id": intervention_id,
        "path": path,
        "labels": {
            "annotations": {},
            "predictions": {},
        },
        "n_frame": n_frame,
        "in_progress": False
    }
    return template


def extract_frames_to_db(frames: List[int], intervention_id: ObjectId, base_path_frames: Path, db_images, db_interventions, verbose=False):
    intervention = db_interventions.find_one({"_id": intervention_id})
    video_key = intervention["video_key"]
    frame_directory = base_path_frames.joinpath(video_key)
    assert frame_directory.exists()
    cap = cv2.VideoCapture(intervention["video_path"])
    if "frames" in intervention:
        intervention_frame_dict = intervention["frames"]
    else:
        intervention_frame_dict = {}
    inserted_image_ids = []
    
    for n_frame in frames:
        if str(n_frame) in intervention_frame_dict:
            if verbose:
                print(f"Frame {n_frame} was already extracted, passing")
            continue
        frame_path = frame_directory.joinpath(f"{n_frame}.png")
        cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
        success, image = cap.read()
        assert success
        cv2.imwrite(frame_path.as_posix(), image)
        
        db_image_entry = get_image_db_template(
            origin=intervention["origin"],
            intervention_id=intervention["_id"],
            path=frame_path.as_posix(),
            n_frame=n_frame
        )
        
        _id = db_images.insert_one(db_image_entry).inserted_id
        intervention_frame_dict[str(n_frame)] = _id
        db_interventions.update_one({"_id": intervention["_id"]}, {"$set": {"frames": intervention_frame_dict}})
        inserted_image_ids.append(_id)
        if verbose:
            print(db_image_entry)
        
    return inserted_image_ids
        
        
def delete_frames_from_db(image_ids: List[ObjectId], db_images, db_interventions, verbose: bool = False):
    for image_id in image_ids:
        image = db_images.find_one({"_id": image_id})
        n_frame = image["n_frame"]
        db_interventions.update_one({"_id": ObjectId("60dae8b12dc144033e693434")}, {"$unset": {f"frames.{n_frame}": ""}})
        db_images.delete_one({"_id": image_id})
        if verbose:
            print(image)
            print("deleted\n\n")


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


def get_images_to_prelabel_query(label: str, version: int, predict_annotated: bool = False, limit: int = 100000):
    """Function expects a label (str) and returns a query to find images which
    have no prediction or an outdated prediction for this label.

    Args:
        label (str): label to query for
        version (float): float value representing the current prelabel AI version
        limit (int, optional): Maximum value of images to return. Defaults to 100000.

    Returns:
        List: List of dictionaries containing the elements for a pymongo query aggregation
    """
    return [
        {
            "$match": {
                "$and": [
                    {
                        "$or": [{
                            f"predictions.{label}": {"$exists": False}
                        }, {
                            f"predictions.{label}.version": {"$lt": version}
                        }]
                    },
                    {f"labels.annotation.{label}": {"$exists": predict_annotated}},
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

####################################################################
# DEPRECIATE


def get_categorical_label_count(label: str, db_images):
    warnings.warn("Depreceation Warning!")
    if label == "polyp_detection_bbox":
        return None
    r = db_images.aggregate(
        [{"$group": {"_id": f"$labels.annotations.{label}", "count": {"$sum": 1}}}]
    )
    r = [_ for _ in r if _["_id"] is not None]
    return r

####################################################################
# DEPRECIATED

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
