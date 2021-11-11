import warnings
from bson.objectid import ObjectId
from pymongo.collection import Collection
import pandas as pd
from typing import List
from pathlib import Path
from .fieldnames import *
from .db_images import DbImages
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
    def read_test_data(self, path: Path, label_list: List[str]):
        df = pd.read_excel(path)
        video_keys = df[FIELDNAME_VIDEO_KEY].dropna().to_list()
        test_interventions = self.db_interventions.db.find({FIELDNAME_VIDEO_KEY: {"$in": video_keys}},{"_id": 1})
        intervention_ids = [_["_id"] for _ in test_interventions]

        for label in label_list:
            self.db.update_one(
                {"label": label},
                {
                    "$set": {
                        "intervention_ids": intervention_ids,
                        "video_keys": video_keys
                    }
                    },
                upsert = True)


    ## Read
    def get_test_data_interventions(db_tests, label: str) -> List[ObjectId]:
        test_set = db_tests.db.find_one({"label": label})
        if test_set:
            return test_set["intervention_ids"]

        else:
            warnings.warn(f"No Test Set for Label {label} found")
            return []

