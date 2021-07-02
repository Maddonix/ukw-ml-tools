from pathlib import Path
import warnings
from typing import List
from collections import Counter


def validate_adrian_annotation_json(annotation: dict) -> bool:
    """Function to validate json files from adrians annotation tool. Returns
    False if any of "metadata", "metadata.videofile", "superframes" or "images"
    are missing.

    Args:
        annotation (dict): Dictionarie of read annotation json.

    Returns:
        bool: True if valid, else false.
    """
    if "metadata" not in annotation:
        warnings.warn("Annotation does not contain Metadata")
        return False
    if "images" not in annotation:
        warnings.warn("Annotation does not contain Image-Array")
        return False
    if "superframes" not in annotation:
        warnings.warn("Annotation does not contain Superframe-Array")
        return False
    if "videoFile" not in annotation["metadata"]:
        warnings.warn("Annotation Metadata does not contain a video file name")
        return False
    if "imageCount" not in annotation["metadata"]:
        warnings.warn("Annotation Metadata does not contain an image count")
    return True


def validate_video_keys(db_interventions) -> List:
    """Expects db_intervention collection, filters for non unique video_keys

    Args:
        db_interventions (mongoCollection): 

    Returns:
        List: List of non-unique video keys
    """
    interventions = db_interventions.find({"video_key": {"$exists": True}}, {"video_key": 1})
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
