from bson.objectid import ObjectId
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from typing import List
import warnings
from pathlib import Path
from .utils import *
from collections import defaultdict
import pandas as pd

from .fieldnames import *

aggregations = {}


class DbImages:
    """Add Documentation
    asd
    """

    def __init__(self, db_images: Collection, cfg: dict):
        self.db = db_images
        self.cfg = cfg
        self.exclude_origins_for_prelabeling = cfg["exclude_origins_for_prelabeling"]
        self.task_modes = cfg["tasks"]["modes"]
        

    # Queries
    ## Read
    def get_all(self, as_list: bool = False) -> (Cursor or List):
        images = self.db.find({})

        if as_list: return [_ for _ in images]
        else: return images

    def get_by_id(self, _id: ObjectId) -> dict:
        return self.db.find_one({"_id": _id})

    def get_by_id_list(self, _ids: List[ObjectId], as_list: bool=True) -> List[dict] or Cursor:
        images = self.db.find({"_id": {"$in": _ids}})

        if as_list: return [_ for _ in images]
        else: return images

    def get_train_images(
        self,
        label: str,
        value,
        test_intervention_ids: List[ObjectId],
        extend_conditions: List = None,
        extend_agg: List = None,
        as_list: bool = False
        ):

        aggregation = get_train_image_query(label, value, test_intervention_ids, extend_conditions, extend_agg)

        # conditions = [
        #     field_value_query(f"{FIELDNAME_LABELS}.{label}", value),
        #     fieldvalue_nin_list_query(FIELDNAME_INTERVENTION_ID, test_intervention_ids)
        # ]

        # if extend_conditions:
        #     conditions.extend(extend_conditions)

        # aggregation = [match_logical_and_aggregation(conditions)]

        # if extend_agg:
        #     aggregation.extend(extend_agg)

        try:
            images = self.db.aggregate(aggregation)
        except:
            print(aggregation)
            raise Exception
            return False
        if as_list:
            images = [_ for _ in images]

        return images


    def get_ls_task_images(
        self,
        label,
        upper_confidence_threshold: float = 50,
        lower_confidence_threshold: float = 50,
        mode: str = "high_conf",
        limit: int = 10000,
        exclude_origins: List[str] = None
    ):
        if not exclude_origins:
            if isinstance(exclude_origins, List):
                pass
            else: 
                exclude_origins = self.exclude_origins_for_prelabeling 

        assert mode in self.task_modes
        latest_ai_version = max(self.db.distinct(
            f"{FIELDNAME_PREDICTIONS}.{label}.{FIELDNAME_AI_VERSION}"
        ))
        if mode == "high_conf":
            has_label = False
            confidence_aggregation = {
                "$or": [
                    {f"{FIELDNAME_PREDICTIONS}.{label}.{FIELDNAME_PREDICTION_VALUE}": {"$gt": upper_confidence_threshold}},
                    {f"{FIELDNAME_PREDICTIONS}.{label}.{FIELDNAME_PREDICTION_VALUE}": {"$lt": lower_confidence_threshold}},
                ]
            }
        elif mode == "low_conf":
            has_label = False
            confidence_aggregation = {
                "$and": [
                    {f"{FIELDNAME_PREDICTIONS}.{label}.{FIELDNAME_PREDICTION_VALUE}": {"$lt": upper_confidence_threshold}},
                    {f"{FIELDNAME_PREDICTIONS}.{label}.{FIELDNAME_PREDICTION_VALUE}": {"$gt": lower_confidence_threshold}},
                ]
            }
        elif mode == "contradicting":
            has_label = True
            confidence_aggregation = {
                "$or": [
                    {
                        "$and": [
                            {f"{FIELDNAME_PREDICTIONS}.{label}.{FIELDNAME_PREDICTION_VALUE}": {"$gt": upper_confidence_threshold}},
                            {f"{FIELDNAME_LABELS}.{label}": False},
                            {f"{FIELDNAME_LABELS_VALIDATED}.{label}": {"$exists": False}}
                        ]
                    },{
                        "$and": [
                            {f"{FIELDNAME_PREDICTIONS}.{label}.{FIELDNAME_PREDICTION_VALUE}": {"$lt": lower_confidence_threshold}},
                            {f"{FIELDNAME_LABELS}.{label}": True},
                            {f"{FIELDNAME_LABELS_VALIDATED}.{label}": {"$exists": False}}
                        ]
                    }
                ]
            }

        _images = self.db.aggregate(
            [
                {
                    "$match": {
                        "$and": [
                            {f"{FIELDNAME_LABELS}.{label}": {"$exists": has_label}},  # Unlabeled Images
                            {
                                "$or": [
                                    {
                                        f"{FIELDNAME_LABELS_UNCLEAR}.{LABEL_UNCLEAR}": False
                                    }, 
                                    {f"{FIELDNAME_LABELS_UNCLEAR}.{LABEL_UNCLEAR}": {"$exists": False}},
                                ]
                            },
                            {f"{FIELDNAME_PREDICTIONS}.{label}.{FIELDNAME_AI_VERSION}": latest_ai_version},
                            {FIELDNAME_ORIGIN: {"$nin": exclude_origins}},
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

    def get_prelabel_images(
        self,
        label: str,
        version: float = None,
        origin: str = None,
        limit: int = 1000,
        intervention_types: List[str] = ["Koloskopie", "Unknown"],
        predict_annotated: bool = False,
        exclude_origins: List[str] = None,
        exclude_interventions: List[ObjectId] = None,
        return_df: bool = False
    ) -> List:
        agg_content = []
        if not exclude_origins:
            if isinstance(exclude_origins, List):
                pass
            else: 
                exclude_origins = self.exclude_origins_for_prelabeling


        if origin:
            agg_content.append({"$match": {FIELDNAME_ORIGIN: origin}})
        
        q1 = logical_or_aggregation([
                {f"{FIELDNAME_PREDICTIONS}.{label}": {"$exists": False}},
                {f"{FIELDNAME_PREDICTIONS}.{label}.{FIELDNAME_AI_VERSION}": {"$lt": version}}
            ])

        agg_content = [
            q1,
            {f"{FIELDNAME_LABELS}.{label}": {"$exists": predict_annotated}},
            {FIELDNAME_ORIGIN: {"$nin": exclude_origins}},
            {FIELDNAME_INTERVENTION_TYPE: {"$in": intervention_types}}
            ]
        
        if exclude_interventions:
            agg_content.append({FIELDNAME_INTERVENTION_ID: {"$nin": exclude_interventions}})

        agg = [match_logical_and_aggregation(agg_content)]

        agg.append({"$sample": {"size": limit}})
        images = [_ for _ in self.db.aggregate(agg)]

        if return_df:
            df_dict = {"file_path": [], "_id": []}
            for _ in images:
                df_dict["_id"].append(_["_id"])
                df_dict["file_path"].append(_[FIELDNAME_IMAGE_PATH])
            images = pd.DataFrame().from_dict(df_dict)

        return images

    def get_label_count(
        self,
        label_list: List[str],
        exclude_intervention_ids: List[ObjectId] = []
    ) -> dict:
        count = defaultdict(dict)
        for label in label_list:
            r = self.db.aggregate(
                [
                    {
                        "$match": {
                            FIELDNAME_INTERVENTION_ID: {"$nin": exclude_intervention_ids}
                        }
                    },
                    {
                        "$group": {
                            "_id": f"{FIELDNAME_LABELS}.{label}",
                            "count": {"$sum": 1}
                        }
                    }
                ]
            )

            for category_count in r:
                if category_count["_id"] is None:
                    continue
                count[label][category_count["_id"]] = category_count["count"]

        return count

    def get_label_count_by_origin(
        self,
        label_list: List[str],
        exclude_intervention_ids: List[ObjectId]
        ):

        count = defaultdict(generate_default_dict)
        for label in label_list:
            r = self.db.aggregate(
                [
                    {
                        "$match": {
                            FIELDNAME_INTERVENTION_ID: {"$exists": True, "$nin": exclude_intervention_ids},
                            FIELDNAME_LABELS: {"$exists": True, "$nin": [{}]},
                            FIELDNAME_ORIGIN: {"$exists": True}
                        }
                    },
                    {
                        "$group": {
                            "_id": {FIELDNAME_LABELS: f"${FIELDNAME_LABELS}.{label}", FIELDNAME_ORIGIN: f"${FIELDNAME_ORIGIN}"},
                            "count": {"$sum": 1},
                        }
                    }
                ]
            )

            for category_count in r:
                count[label][category_count["_id"][FIELDNAME_ORIGIN]][
                    category_count["_id"][FIELDNAME_LABELS]
                ] = category_count["count"]

        return count

    def get_labelcount_by_intervention(self, label_list: List[str] = []):
        count = defaultdict(generate_default_dict)
        for label in label_list:
            r = self.db.aggregate(
                [
                    {
                        "$match": {
                            FIELDNAME_LABELS: {"$exists": True, "$nin": [{}]},
                            FIELDNAME_INTERVENTION_ID: {"$exists": True}
                        }
                    },
                    {
                        "$group": {
                            "_id": {
                                FIELDNAME_LABELS: f"${FIELDNAME_LABELS}.{label}",
                                FIELDNAME_INTERVENTION_ID: f"${FIELDNAME_INTERVENTION_ID}",
                            },
                            "count": {"$sum": 1},
                        }
                    }
                ]
            )

            for category_count in r:
                count[label][category_count["_id"][FIELDNAME_INTERVENTION_ID]][
                    category_count["_id"][FIELDNAME_LABELS]
                ] = category_count["count"]

        return count

    ## Update
    def update_image_predictions(self, label: str, update: List[dict]):
        for _id, _update in update.items():
            self.db.update_one({"_id": _id}, {"$set": {f"{FIELDNAME_PREDICTIONS}.{label}":_update}})


    def calculate_stats(self, return_records: bool = True):
        self.stats_queries = {
            "entity": PREFIX_IMAGE,
            "get_count": [],
            "get_grouped_count": [],
            "get_grouped_dict_count": [
                {
                    # All Labels Count
                    "fieldname": FIELDNAME_LABELS,
                    "additional_match_conditions": None,
                    "additional_group_conditions": None
                },
                {
                    # Labels by Origin
                    "fieldname": FIELDNAME_LABELS,
                    "additional_match_conditions": None,
                    "additional_group_conditions": {FIELDNAME_ORIGIN: "$"+FIELDNAME_ORIGIN}
                },
                # {
                #     # Labels by Intervention_id
                #     "fieldname": FIELDNAME_LABELS,
                #     "additional_match_conditions": None,
                #     "additional_group_conditions": {FIELDNAME_INTERVENTION_ID: "$"+FIELDNAME_INTERVENTION_ID}
                # },
            ]
        }

        stats = calculate_stats_dict(self.db, self.stats_queries, return_records = return_records)
        return stats


    # Stats
    # n_frames
    # n_frames with annotations
    # n_frames with predictions
    # n_frames with annotation + Prediction
    # n_frames by origin
    # n_frames by intervention
    # n_frames with label / value
    # n_frames with prediction / value
    # n_frames by label: annotation and prediction agree
    # n_frames by label: annotation and prediction contradicting 


    # Validation
    def validate_image_paths(self) -> List:
        images = self.get_all()
        no_image = [str(_["_id"]) for _ in images if not Path(_[FIELDNAME_IMAGE_PATH]).exists()]

        if no_image:
            warnings.warn("Images in Db without existing file were found")
            warnings.warn("\n".join(no_image))

        return no_image

    def validate_all_intervention_ids_exist(self) -> List:
        return []
    #     _ids = self.db.distinct(FIELDNAME_INTERVENTION_ID)
    #     intervention_not_found = []
    #     for _ in _ids:
    #         intervention = self.db_interventions.find_one({"_id": _})
    #         if not intervention:
    #             intervention_not_found.append(str(_))

    #     return intervention_not_found