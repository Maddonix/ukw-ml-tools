from pathlib import Path
import warnings
from typing import List
from collections import Counter


# Validate if no duplicate video keys exist
def validate_video_keys(db_interventions) -> List:
    """Expects db_intervention collection, filters for non unique video_keys

    Args:
        db_interventions (mongoCollection):

    Returns:
        List: List of non-unique video keys
    """
    interventions = db_interventions.find(
        {"video_key": {"$exists": True}}, {"video_key": 1}
    )
    keys = [_["video_key"] for _ in interventions]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]

    if duplicates:
        warnings.warn("Non unique video keys detected")
    return duplicates


# Validate if files exist
def validate_image_paths(db_images) -> List:
    """
    Expects db image collection. Checks all paths if they exist.
    Warns if any paths do not exist and returns list of image ids where path doesn't exist.
    """
    image_ids = []
    images = db_images.find({}, {"path": 1})
    for _ in images:
        if not Path(_["path"]).exists():
            image_ids.append(_["_id"])
    if image_ids:
        warnings.warn(
            "Not all images for paths of given image collection exist. Returning ID's of invalid images"
        )
    return image_ids


# Validate if video files exist
def validate_video_paths(db_interventions):
    """
    Expects db interventions collection. Checks all entries with have "video_key" if the video file exists.
    Warns if any paths do not exist and returns list of intervention ids where path doesn't exist.
    """
    agg = [{"$match": {"video_path": {"$exists": True}}}]
    videos = db_interventions.aggregate(agg)
    video_ids = []

    for _ in videos:
        if not Path(_["video_path"]).exists():
            video_ids.append(_["_id"])

    if video_ids:
        warnings.warn(
            "Not all videos for paths of given intervention collection exist. Returning ID's of invalid interventions"
        )

    return video_ids
