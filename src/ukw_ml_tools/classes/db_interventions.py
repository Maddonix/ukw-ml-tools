from pymongo.collection import Collection
from bson.objectid import ObjectId
from collections import Counter
import warnings
from pathlib import Path
from typing import List
from pymongo.cursor import Cursor
import os

from tqdm import tqdm

from .terminology import Terminology
from .utils import *
from .fieldnames import *
import shutil


class DbInterventions:
    """Add Documentation
    asd
    """
    def __init__(self, db_interventions: Collection, cfg: dict):
        self.db = db_interventions
        self.cfg = cfg
        self.video_key_re_date_pattern = cfg["video_key_re_date_pattern"]
        self.video_key_dt_date_pattern = cfg["video_key_dt_date_pattern"]
        self.paths = {
            "base_path_frames": Path(cfg["base_path_frames"])
        }
        self.terminology = None

    def get_by_id(self, _id: ObjectId):
        return self.db.find_one({"_id": _id})

    # Terminology
    def init_terminology(self, terminology_path: Path = None, terminology_type: int = 0):
        self.terminology = Terminology(self.cfg, terminology_path, terminology_type)

    def apply_terminology(self, intervention_list: List[dict] = None, set_tokens_in_db: bool = False, return_tokens = True, preprocess_text: bool = True):
        if set_tokens_in_db:
            assert return_tokens
        
        if not self.terminology:
            self.init_terminology()

        terminology_type = self.terminology.terminology_type

        if not intervention_list:
            intervention_list = self.get_interventions_with_report_or_patho()
        
        results = {}
        intervention_list = [_ for _ in intervention_list]

        for intervention in tqdm(intervention_list):
            if FIELDNAME_REPORT_RAW in intervention:
                text = intervention[FIELDNAME_REPORT_RAW]
                if isinstance(text, dict):
                    text = dict_of_lists_to_text(text)
                elif isinstance(text, type(None)):
                    text = ""
                result_report = self.terminology.get_terminology_result(text, terminology_type, return_tokens, preprocess_text)
            else:
                result_report = {}
            if FIELDNAME_PATHO_RAW in intervention:
                text = intervention[FIELDNAME_PATHO_RAW]
                if isinstance(text, dict):
                    text = dict_of_lists_to_text(text)
                elif isinstance(text, type(None)):
                    text = ""
                result_patho = self.terminology.get_terminology_result(text, terminology_type, return_tokens, preprocess_text)
            else:
                result_patho = {}
            result = {FIELDNAME_TOKENS_REPORT: result_report, FIELDNAME_TOKENS_PATHO: result_patho}
            results[intervention["_id"]] = result

            if set_tokens_in_db:
                self.db.update_one({"_id": intervention["_id"]}, {"$set": {FIELDNAME_TOKENS: result}})

        return results


    # Queries
    ## Create
    def create_new_from_external(self, intervention_dict: dict):
        video_key = intervention_dict[FIELDNAME_VIDEO_KEY]
        _ = self.db.find_one(field_value_query(FIELDNAME_VIDEO_KEY, video_key))
        if _:
            return False

        intervention_date = datetime_from_video_key(
            video_key,
            self.video_key_re_date_pattern,
            self.video_key_dt_date_pattern
        )

        metadata = get_video_meta(Path(intervention_dict[FIELDNAME_VIDEO_PATH]))
        intervention_dict[FIELDNAME_VIDEO_METADATA] = metadata

        if intervention_date:
            intervention_dict[FIELDNAME_INTERVENTION_DATE] = intervention_date

    
        result = self.db.insert_one(intervention_dict)
        return result.inserted_id

    def create_new_intervention_from_video_file(self, source_video_path: Path, target_video_dir: Path, origin: str, intervention_type: str):
        intervention_dict = get_intervention_db_template()
        
        assert intervention_type in self.cfg["intervention_types"]
        assert source_video_path.is_file()
        assert target_video_dir.is_dir()

        video_key = source_video_path.name
        target_path = target_video_dir.joinpath(video_key)
        assert not target_path.exists()
        assert not self.db.find_one({FIELDNAME_VIDEO_KEY: video_key})
        intervention_date = datetime_from_video_key(
            video_key,
            self.video_key_re_date_pattern,
            self.video_key_dt_date_pattern
        )

        metadata = get_video_meta(source_video_path)

        intervention_dict[FIELDNAME_VIDEO_PATH] = target_path.as_posix()
        intervention_dict[FIELDNAME_VIDEO_KEY] = video_key
        intervention_dict[FIELDNAME_ORIGIN] = origin
        intervention_dict[FIELDNAME_VIDEO_METADATA] = metadata
        intervention_dict[FIELDNAME_INTERVENTION_TYPE] = intervention_type
        if intervention_date:
            intervention_dict[FIELDNAME_INTERVENTION_DATE] = intervention_date

        shutil.copyfile(source_video_path, target_path)
        result = self.db.insert_one(intervention_dict)

        return result.inserted_id
    
    ## Read
    def get_interventions_with_video(self, as_list: bool=False) -> (Cursor or List):
        interventions = self.db.find(field_exists_query(FIELDNAME_VIDEO_PATH))

        if as_list: return [_ for _ in interventions]
        else: return interventions

    def get_interventions_with_freezes(self, as_list: bool = False):
        interventions = self.db.find(fieldvalue_nin_list_query(FIELDNAME_FREEZES, [{}]))

        if as_list: return [_ for _ in interventions]
        else: return interventions

    def get_interventions_with_frames(self, as_list: bool = False):
        interventions = self.db.find(fieldvalue_nin_list_query(FIELDNAME_FRAMES, [{}]))

        if as_list: return [_ for _ in interventions]
        else: return interventions

    def get_interventions_with_reports(self, as_list: bool = False):
        interventions = self.db.find({FIELDNAME_REPORT_RAW: {"$exists": True}})

        if as_list: return [_ for _ in interventions]
        else: return interventions

    def get_interventions_with_patho(self, as_list: bool = False):
        interventions = self.db.find({FIELDNAME_PATHO_RAW: {"$exists": True}})

        if as_list: return [_ for _ in interventions]
        else: return interventions

    def get_interventions_with_report_or_patho(self, as_list: bool = False):
        aggregation = [
            {
                "$match": {
                    "$or": [
                        {FIELDNAME_PATHO_RAW: {"$exists": True, "$nin": [None, ""]}},
                        {FIELDNAME_REPORT_RAW: {"$exists": True, "$nin": [None, ""]}}
                    ]
                }
            }
        ]

        interventions = self.db.aggregate(aggregation)

        if as_list: return [_ for _ in interventions]
        else: return interventions

    def get_distinct_values(self, field_str: str):
        return self.db.distinct(field_str)

    # Stats
    def calculate_stats(self, return_records: bool = True):
        self.stats_queries = {
            "entity": PREFIX_INTERVENTION,
            "get_count": [
                {f"{PREFIX_EXISTS}.{FIELDNAME_PATHO_RAW}": {FIELDNAME_PATHO_RAW: {"$exists": True}}},
                {f"{PREFIX_EXISTS}.{FIELDNAME_REPORT_RAW}": {FIELDNAME_REPORT_RAW: {"$exists": True}}},
                {f"{PREFIX_EXISTS}.{FIELDNAME_REPORT_RAW},{FIELDNAME_PATHO_RAW}": {
                    FIELDNAME_PATHO_RAW: {"$exists": True},
                    FIELDNAME_REPORT_RAW: {"$exists": True}
                }},
                {f"{PREFIX_EXISTS}.{FIELDNAME_VIDEO_KEY}": {FIELDNAME_VIDEO_KEY: {"$exists": True}}},
            ],
            "get_grouped_count": [
                {
                    "fieldname": FIELDNAME_ORIGIN,
                    "additional_match_conditions": None
                },
                {
                    "fieldname": FIELDNAME_ORIGIN,
                    "additional_match_conditions": {
                        f"{PREFIX_EXISTS}.{FIELDNAME_PATHO_RAW}": {FIELDNAME_PATHO_RAW: {"$exists": True}}
                    }
                },
                {
                    "fieldname": FIELDNAME_ORIGIN,
                    "additional_match_conditions": {
                        f"{PREFIX_EXISTS}.{FIELDNAME_REPORT_RAW}": {FIELDNAME_REPORT_RAW: {"$exists": True}}
                    }
                },
                {
                    "fieldname": FIELDNAME_ORIGIN,
                    "additional_match_conditions": {
                        f"{PREFIX_EXISTS}.{FIELDNAME_REPORT_RAW},{FIELDNAME_PATHO_RAW}": {FIELDNAME_REPORT_RAW: {"$exists": True}, FIELDNAME_PATHO_RAW: {"$exists": True}}
                    }
                },
            ],
            "get_grouped_dict_count": []
        }

        stats = calculate_stats_dict(self.db, self.stats_queries, return_records = return_records)
        return stats

    # Update
    def update_n_extracted_frames(self):
        aggregation = [
            {
                "$project": {
                    "_id": 1,
                    "frame_count": {
                        "$size": {
                            "$objectToArray": f"${FIELDNAME_FRAMES}"
                        }
                    }
                }
            },
            {
                "$match": {
                }
            }
        ]
        interventions = self.db.aggregate(aggregation)
        for _ in interventions:
            self.db.update_one({"_id": _["_id"]}, {"$set": {FIELDNAME_N_EXTRACTED_FRAMES: _["frame_count"]}})
    # Validation
    def validate_video_keys(self) -> List[str]:
        """Filters for non unique video_keys

        Args:
            db_interventions (Collection):

        Returns:
            List: List of non-unique video keys
        """
        interventions = self.db.find(
            {FIELDNAME_VIDEO_KEY: {"$exists": True}}, {FIELDNAME_VIDEO_KEY: 1}
        )
        keys = [_[FIELDNAME_VIDEO_KEY] for _ in interventions]
        duplicates = [key for key, count in Counter(keys).items() if count > 1]

        if duplicates:
            warnings.warn("Non unique video keys detected")
            warnings.warn("\n".join(duplicates))

        return duplicates

    def validate_video_paths(self) -> List[dict]:
        _interventions = self.get_interventions_with_video(as_list = True)

        path_is_none = []
        interventions = []

        for _ in _interventions:
            if not _[FIELDNAME_VIDEO_PATH]:
                path_is_none.append(str(_["_id"]))
            else:
                interventions.append(_)

        not_exists = [
            str(_["_id"]) for _ in interventions if not Path(_[FIELDNAME_VIDEO_PATH]).exists()
        ]

        if not_exists:
            warnings.warn("Videos pointing to non existing file detected")

        if path_is_none:
            warnings.warn("Videos with path == None or path == '' found")

        return not_exists, path_is_none

    def validate_frame_dirs(self, create_if_missing: bool = False) -> List[dict]:
        interventions = self.get_interventions_with_video(as_list = False)
        
        no_frame_dir = [
            str(_["_id"]) for _ in interventions if not generate_frame_path(
                video_key=_[FIELDNAME_VIDEO_KEY],
                base_path_frames=self.paths["base_path_frames"]
            ).exists()
        ]

        if no_frame_dir:
            if create_if_missing:
                for intervention in no_frame_dir:
                    _path = generate_frame_path(
                        video_key=intervention[FIELDNAME_VIDEO_KEY],
                        base_path_frames=self.paths["base_path_frames"]
                    )
                    os.mkdir(_path)
            else:
                warnings.warn("Videos without frame directoy detected")

        return no_frame_dir

