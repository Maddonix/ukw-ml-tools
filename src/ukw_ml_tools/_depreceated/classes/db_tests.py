import warnings
from bson.objectid import ObjectId
from pymongo.collection import Collection
import pandas as pd
from typing import List
from pathlib import Path
from .fieldnames import *
from .db_images import DbImages
from collections import defaultdict
from .db_interventions import DbInterventions


class DbTests:
    """[summary]
    """
    def __init__(self, db_tests: Collection, db_images: DbImages, db_interventions: DbInterventions, cfg: dict):
        self.db = db_tests
        self.db_images = db_images
        self.db_interventions = db_interventions
        self.cfg = cfg


    # CRUD
    ## Create / Update
    def read_test_data(self, test_data: List[dict]):
        db_test_data_ids = defaultdict(list)
        db_test_data_keys = defaultdict(list)

        for test_data_dict in test_data:
            path = Path(test_data_dict["path"])
            label_list = test_data_dict["label_list"]

            if path.suffix == ".csv":
                df = pd.read_csv(path)
            elif path.suffix == ".xlsx":
                df = pd.read_excel(path)
            video_keys = df[FIELDNAME_VIDEO_KEY].dropna().to_list()
            test_interventions = self.db_interventions.db.find({FIELDNAME_VIDEO_KEY: {"$in": video_keys}},{"_id": 1})
            intervention_ids = [_["_id"] for _ in test_interventions]

            for label in label_list:
                db_test_data_ids[label].extend(intervention_ids)
                db_test_data_keys[label].extend(video_keys)

        for label in db_test_data_ids.keys():
            self.db.update_one(
                {"label": label},
                {
                    "$set": {
                        "intervention_ids": db_test_data_ids[label],
                        "video_keys": db_test_data_keys[label]
                    }
                    },
                upsert = True)


    ## Read
    def get_test_data_interventions(self, label: str) -> List[ObjectId]:
        test_set = self.db.find_one({"label": label})
        if test_set:
            return test_set["intervention_ids"]

        else:
            warnings.warn(f"No Test Set for Label {label} found")
            return []

