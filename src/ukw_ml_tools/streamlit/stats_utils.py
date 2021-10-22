from collections import defaultdict
import json
from typing import List


def get_default_dict():
    """Helper function returns a defaultdict generating dictionaries"""
    return defaultdict(dict)


def get_label_count_by_origin(label_list: List[str], db_images):
    count_all = defaultdict(get_default_dict)
    for label in label_list:
        r = db_images.aggregate(
            [
                {
                    "$group": {
                        "_id": {"label": f"$labels_new.{label}", "origin": "$origin"},
                        "count": {"$sum": 1},
                    }
                }
            ]
        )

        r = [_ for _ in r if "origin" in _["_id"] and "label" in _["_id"]]
        # r.sort()
        for category_count in r:
            count_all[label][category_count["_id"]["origin"]][
                category_count["_id"]["label"]
            ] = category_count["count"]

    return count_all


def get_label_count(label_list: List[str], db_images):
    count_all = defaultdict(dict)
    for label in label_list:
        r = db_images.aggregate(
            [{"$group": {"_id": f"$labels_new.{label}", "count": {"$sum": 1}}}]
        )
        r = [_ for _ in r]
        # r.sort()
        for category_count in r:
            if category_count["_id"] is None:
                continue
            count_all[label][category_count["_id"]] = category_count["count"]

    return count_all


def get_origin_count(db_collection) -> List:
    """Function to return all the number of entries for each unique value of the field "origin" in the given collection

    Args:
        db_collection (pymongo.collection.Collection): pymongo collection

    Returns:
        List: List of tuples containing distinct values of the field "origin" and the number of entries
    """
    r = db_collection.aggregate([{"$group": {"_id": "$origin", "count": {"$sum": 1}}}])
    r = {_["_id"]: _["count"] for _ in r}
    return r


def refresh_label_count(label_list, save_path, db_images):
    all_label_count = get_label_count(label_list, db_images)
    with open(save_path, "w") as f:
        json.dump(all_label_count, f)

    return all_label_count


def refresh_label_by_origin_count(label_list, save_path, db_images):
    label_by_origin_count = get_label_count_by_origin(label_list, db_images)
    with open(save_path, "w") as f:
        json.dump(label_by_origin_count, f)

    return label_by_origin_count


def refresh_videos_origin_count(save_path, db_interventions):
    video_origin_count = get_origin_count(db_interventions)
    with open(save_path, "w") as f:
        json.dump(video_origin_count, f)

    return video_origin_count


def refresh_all(
    db_images,
    db_interventions,
    label_list,
    path_video_origin_count,
    path_label_count,
    path_label_by_origin_count,
):
    label_count = refresh_label_count(label_list, path_label_count, db_images)
    label_by_origin_count = refresh_label_by_origin_count(
        label_list, path_label_by_origin_count, db_images
    )
    video_origin_count = refresh_videos_origin_count(
        path_video_origin_count, db_interventions
    )

    return label_count, label_by_origin_count, video_origin_count
