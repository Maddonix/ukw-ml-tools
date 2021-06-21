from pathlib import Path
import warnings

# Validate if files exist
def validate_image_paths(db_images):
    """
    Expects db image collection. Checks all paths if they exist.
    Warns if any paths do not exist and returns list of image ids where path doesn't exist.
    """
    image_paths = [
        _["_id"]
        for _ in db_images.find({}, {"path": 1})
        if not Path(_["path"]).exists()
    ]
    if image_paths:
        warnings.warn(
            "Not all images for paths of given image collection exist. Returning ID's of invalid images"
        )

    return image_paths


def validate_video_paths(db_interventions):
    """
    Expects db interventions collection. Checks all entries with have "video_key" if the video file exists.
    Warns if any paths do not exist and returns list of intervention ids where path doesn't exist.
    """
    agg = [{"$match": {"video_path": {"$exists": True}}}]
    video_paths = [
        _["_id"]
        for _ in db_interventions.aggregate(agg)
        if not Path(_["video_path"]).exists()
    ]
    if video_paths:
        warnings.warn(
            "Not all videos for paths of given intervention collection exist. Returning ID's of invalid interventions"
        )

    return video_paths
