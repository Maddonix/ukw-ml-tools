from ..classes.intervention import Intervention
from ..classes.image import Image
from ..classes.image import Image
from ..classes.train_data import TrainData, TrainDataDb
import pandas as pd


def get_intervention(identifier, db_interventions, id_is_key=True):
    if id_is_key:
        param = "video_key"
    else:
        param = "_id"

    return Intervention(**db_interventions.find_one({param: identifier}))

def get_image(identifier, db_images):
    return Image(**db_images.find_one({"_id": identifier}))

def get_images_by_id_list(id_list, db_images):
    r = db_images.find({"_id": {"$in": id_list}})
    image_list = [Image(**_) for _ in r]

    return image_list


def get_train_data(name, db_train_data):
    train_data_db = TrainDataDb(**db_train_data.find_one({"name": name}))
    _dict = train_data_db.to_dict_intern()
    train_data = TrainData(**_dict)

    return train_data

def get_annotations(db_annotations):
    annotations = [_ for _ in db_annotations.find({})]
    records = []
    for annotation in annotations:
        img_id = annotation["_id"]
        for annotator_id, value in annotation["annotation"].items():
            record = {
                    "image_id": str(img_id),
                    "annotator_id": int(annotator_id)
                }

            for i, label in enumerate(value["choices"]):
                if i in value["value"]:
                    record[label] = True
                else: record[label] = False
            records.append(record)

    return pd.DataFrame.from_records(records)