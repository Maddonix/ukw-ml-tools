from ..classes.annotation import MultilabelAnnotation
from collections import defaultdict
from tqdm import tqdm


def get_multiclass_image_dict(label_list, db_images, blurry_as_normal = True):
    updates = defaultdict(list)
    conflicts = []
    for label in label_list:
        check_labels = [_ for _ in label_list if _ != label]
        cursor = db_images.find({f"annotations.{label}.value": True})
        for image in tqdm(cursor):
            conflict = False
            for _label in check_labels:
                if _label in image["annotations"]:
                    if image["annotations"][_label]["value"] == True:
                        conflict = True
            if conflict:
                conflicts.append(image["_id"])
            else:
                updates[label].append(image["_id"])
    
    if blurry_as_normal:
        cursor = db_images.find({"annotations.blurry.value": True})
        for image in tqdm(cursor):
            conflict = False
            for _label in label_list:
                if _label in image["annotations"]:
                    if image["annotations"][_label]["value"] == True:
                        conflict = True
            if conflict:
                conflicts.append(image["_id"])
            else:
                updates["normal"].append(image["_id"])

    conflicts = list(set(conflicts))
    return updates, conflicts


def get_outside_images(db_images):
    _label = "body"
    cursor = db_images.aggregate([{"$match": {f"annotations.{_label}.value": False}}])

    return [_["_id"] for _ in cursor]


def set_compound_labels(
    name,
    base_annotation_dict,
    ai_config,
    db_images,
    exclude_initially=["outside", "blurry"],
):
    label_list = [_ for _ in ai_config.choices if _ not in exclude_initially]
    updates, conflicts = get_multiclass_image_dict(label_list, db_images)

    if "outside" in ai_config.choices:
        updates["outside"] = get_outside_images(db_images)

    image_updates = {}
    for label, ids in updates.items():
        value = ai_config.choices.index(label)
        for _id in ids:
            image_updates[_id] = value

    if "blurry" in ai_config.choices:
        blurry_image_ids = [
            _["_id"] for _ in db_images.find({"annotations.blurry.value": True})
        ]

        value = ai_config.choices.index("blurry")
        for _id in blurry_image_ids:
            image_updates[_id] = value

    image_annotations = {}

    for _id, value in image_updates.items():
        _annotation = base_annotation_dict.copy()
        _annotation["value"] = value
        image_annotations[_id] = MultilabelAnnotation(**_annotation)

    for _id, annotation in tqdm(image_annotations.items()):
        db_images.update_one(
            {"_id": _id}, {"$set": {f"annotations.{name}": annotation.dict()}}
        )
