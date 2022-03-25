from ..classes.train_data import TrainData
from ..fieldnames import *


def get_train_data_query(name, exclude_intervention_ids, exclude_origins):
    query = {
        "$match": {
            f"{IMG_ANNOTATIONS}.{name}": {"$exists": True},
            IMG_INTERVENTION_ID: {"$nin": exclude_intervention_ids},
            ORIGIN: {"$nin": exclude_origins},
            f"{METADATA}.is_extracted": True
        }
    }

    return query


def image_list_to_train_data(name, image_list, label_type, choices):
    paths = []
    labels = []
    origins = []
    intervention_ids = []
    image_ids = []
    crop = []
    if "blurry" in choices:
        choices.remove("blurry")

    for image in image_list:

        assert choices == image[IMG_ANNOTATIONS][name]["choices"]
        paths.append(image[METADATA]["path"])
        labels.append(image[IMG_ANNOTATIONS][name]["value"])
        origins.append(image[ORIGIN])
        intervention_ids.append(image[IMG_INTERVENTION_ID])
        image_ids.append(image["_id"])
        crop.append(image["metadata"]["crop"])

    train_data = {
        "name": name,
        "paths": paths,
        "labels": labels,
        "prediction_type": label_type,
        "choices": choices,
        "origins": origins,
        "intervention_ids": intervention_ids,
        "image_ids": image_ids,
        "crop": crop
    }

    train_data = TrainData(**train_data)
    return train_data
