# from pymongo.collection import Collection
import warnings
from typing import List
from pathlib import Path
from pymongo.collection import Collection
import os
from bson.objectid import ObjectId
import cv2
import pandas as pd


def get_prediction_task_df(
    label: str,
    version: float,
    db_images: Collection,
    origin: str = None,
    limit: int = 1000,
    predict_annotated: bool = False,
) -> pd.DataFrame:
    """Function queries the given collection for images to predict the given label on.\
        Images which already have an annotation of this label or a prediction with the current\
        AI version are excluded. If given a origin, only images of the given origin are predicted.\
        Returns a pd.DataFrame with columns "_id" (image id in database) and "file_path" (complete filepath on our server)

    Args:
        label (str): label to query for
        version (float): AI version number
        db_images (Collection): collection containing images
        origin (str, optional): Optionally filter for a specific origin. Defaults to None.
        limit (int, optional): Number to limit the output. Defaults to 1000.

    Returns:
        pd.DataFrame: Dataframe with columns "_id" and "file_path"
    """
    if origin:
        agg = [{"$match": {"origin": origin}}]
    else:
        agg = []

    agg.extend(
        get_images_to_prelabel_query(
            label=label,
            version=version,
            limit=limit,
            predict_annotated=predict_annotated,
        )
    )
    result = db_images.aggregate(agg)

    df_dict = {"file_path": [], "_id": []}
    for _ in result:
        df_dict["_id"].append(_["_id"])
        df_dict["file_path"].append(_["path"])

    return pd.DataFrame().from_dict(df_dict)


def get_binary_train_df(
    label: str, neg_label_list: List[str], db_images
) -> pd.DataFrame:
    """Queries db_images for pos. and neg. images of given label / neg label list

    Args:
        label (str): label defined as positive-
        neg_label_list (List[(str, float)]): List of tuples containing labelname and\
            multiplier. Multiplier is used to limit entries of each label to n_pos * multiplier
        db_images (): pymongo collection instance

    Returns:
        pd.DataFrame: pd.Dataframe containing columns "file_path" and "label". Label has value\
            0 or 1.
    """
    # Positive Images
    agg = [{"$match": {f"labels_new.{label}": True}}]
    imgs_pos = [_["path"] for _ in db_images.aggregate(agg)]
    n_pos = len(imgs_pos)
    imgs_pos_df = {"file_path": [_ for _ in imgs_pos], "label": [1 for _ in imgs_pos]}

    # Negative Images
    imgs_neg = [_["path"] for _ in db_images.find({f"labels_new.{label}": False})]

    for neg_label, multiplier in neg_label_list:
        agg = [
            {"$match": {f"labels_new.{neg_label}": True}},
            {"$limit": n_pos * multiplier},
        ]
        _imgs_neg = [_["path"] for _ in db_images.aggregate(agg)]
        imgs_neg.extend(_imgs_neg)

    imgs_neg_df = {"file_path": [_ for _ in imgs_neg], "label": [0 for _ in imgs_neg]}

    df_dict = imgs_pos_df
    df_dict["file_path"].extend(imgs_neg_df["file_path"])
    df_dict["label"].extend(imgs_neg_df["label"])

    label_df = pd.DataFrame.from_dict(df_dict)
    return label_df


def get_image_db_template(
    origin: str = None,
    intervention_id: ObjectId = None,
    path: str = None,
    n_frame: int = None,
):
    """Returns db_images template

    Args:
        origin (str, optional): String to define source. Defaults to None.
        intervention_id (ObjectId, optional): Id of intervention the image originates from. Defaults to None.
        path (str, optional): path to image. Defaults to None.
        n_frame (int, optional): Integer reffering to the frame number in the original video. Defaults to None.

    Returns:
        dict
    """

    template = {
        "origin": origin,
        "intervention_id": intervention_id,
        "path": path,
        "n_frame": n_frame,
        "in_progress": False,
    }
    return template


def extract_frames_to_db(
    frames: List[int],
    intervention_id: ObjectId,
    base_path_frames: Path,
    db_images,
    db_interventions,
    verbose=False,
):
    """Extracts given frames (as List of integers) from intervention with given id.

    Args:
        frames (List[int]): List of integers referring to frame numbers.
        intervention_id (ObjectId): Intervention id to extract from
        base_path_frames (Path): Base Path to store frames in
        db_images ([type]): pymongo collection instance
        db_interventions ([type]): pymongo collection instance
        verbose (bool, optional): Triggers console output. Defaults to False.

    Returns:
        List[ObjectId]: Returns list of inserted ids in db_images
    """
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
            n_frame=n_frame,
        )

        _id = db_images.insert_one(db_image_entry).inserted_id
        intervention_frame_dict[str(n_frame)] = _id
        db_interventions.update_one(
            {"_id": intervention["_id"]}, {"$set": {"frames": intervention_frame_dict}}
        )
        inserted_image_ids.append(_id)
        if verbose:
            print(db_image_entry)

    return inserted_image_ids


def delete_frames_from_db(
    image_ids: List[ObjectId], db_images, db_interventions, verbose: bool = False
):
    """Function expect List of image ids. Deletes them in db_images and also deletes their reference in\
        db_interventions. Only works with extracted frames from videos!

    Args:
        image_ids (List[ObjectId]): List of Ids for frames to delete.
        db_images ([type]): pymongo collection instance. Entries must have fields "intervention_id" and "n_frame"
        db_interventions ([type]): pymongo collection instance. Entries must have field "frames"\
            containing a dictionary with frame number as keys.
        verbose (bool, optional): Enables console output. Defaults to False.
    """
    for image_id in image_ids:
        image = db_images.find_one({"_id": image_id})
        n_frame = image["n_frame"]
        intervention_id = image["intervention_id"]
        db_interventions.update_one(
            {"_id": intervention_id}, {"$unset": {f"frames.{n_frame}": ""}}
        )
        db_images.delete_one({"_id": image_id})
        if verbose:
            print(image)
            print("deleted\n\n")


def get_images_to_prelabel_query(
    label: str, version: int, predict_annotated: bool = False, limit: int = 100000
):
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
                        "$or": [
                            {f"predictions.{label}": {"$exists": False}},
                            {f"predictions.{label}.version": {"$lt": version}},
                        ]
                    },
                    {f"labels.annotation.{label}": {"$exists": predict_annotated}},
                ]
            }
        },
        {"$limit": limit},
    ]


def get_images_for_tasks(
    label: str,
    upper_confidence_threshold: float,
    lower_confidence_threshold: float,
    inclusive: bool,
    limit: int,
    db_images,
):
    """Queries db_images for images to annotate. Images will be selected\
        between confidence threshholds (inclusive True) or outside confidence\
        threshholds (inclusive False).\
        Images with labels_unclear.unclear == True will be excluded.\
        Only images with highest ai-version will be included.


    Args:
        label (str): label to query for
        upper_confidence_threshold (float): value between 0 and 1.
        lower_confidence_threshold (float): value between 0 and 1.
        inclusive (bool): If True, included images have prediction values\
            between threshholds. Otherwise, these images are excluded
        limit (int): Maximum amount of images to return
        db_images ([type]): pymongo collection instance.

    Returns:
        [type]: [description]
    """
    latest_ai_version = max(db_images.distinct(f"predictions.{label}.version"))
    if not inclusive:
        confidence_aggregation = {
            "$or": [
                {f"predictions.{label}.value": {"$gt": upper_confidence_threshold}},
                {f"predictions.{label}.value": {"$lt": lower_confidence_threshold}},
            ]
        }
    else:
        confidence_aggregation = {
            "$and": [
                {f"predictions.{label}.value": {"$lt": upper_confidence_threshold}},
                {f"predictions.{label}.value": {"$gt": lower_confidence_threshold}},
            ]
        }

    _images = db_images.aggregate(
        [
            {
                "$match": {
                    "$and": [
                        {f"labels_new.{label}": {"$exists": False}},  # Unlabeled Images
                        {
                            "$or": [
                                {
                                    "labels_unclear.unclear": False
                                },  # Images have not been labeled as unclear
                                {"labels_unclear.unclear": {"$exists": False}},
                            ]
                        },
                        {f"predictions.{label}.version": latest_ai_version},
                        confidence_aggregation,
                    ]
                }
            },
            {"$sample": {"size": limit}},
        ]
    )

    _ids = []
    images = []
    for image in _images:
        if image["_id"] not in _ids:
            _ids.append(image["_id"])
            images.append(image)

    return images


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
