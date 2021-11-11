from pymongo.collection import Collection
from bson.objectid import ObjectId
from collections import Counter
import warnings
from pathlib import Path
from typing import List
from pymongo.cursor import Cursor
import os
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

    def get_by_id(self, _id: ObjectId):
        return self.db.find_one({"_id": _id})


    # Queries
    ## Create
    def create_new_from_external(self, intervention_dict: dict):
        video_key = intervention_dict[FIELDNAME_VIDEO_KEY]
        intervention_date = datetime_from_video_key(
            video_key,
            self.video_key_re_date_pattern,
            self.video_key_dt_date_pattern
        )

        metadata = get_video_meta(Path(intervention_dict[FIELDNAME_VIDEO_PATH]))
        intervention_dict[FIELDNAME_VIDEO_METADATA] = metadata

        if intervention_date:
            intervention_dict[FIELDNAME_INTERVENTION_DATE] = intervention_date

        _ = self.db.find_one(field_value_query(FIELDNAME_VIDEO_KEY, video_key))

        if _:
            return False
        else:
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
        interventions = self.get_interventions_with_video(as_list = True)
    
        no_path = [
            _ for _ in interventions if not Path(_[FIELDNAME_VIDEO_PATH]).exists()
        ]

        if no_path:
            warnings.warn("Videos pointing to non existing file detected")
            warnings.warn("\n".join(no_path))

        return no_path


    def validate_frame_dirs(self, create_if_missing: bool = False) -> List[dict]:
        interventions = self.get_interventions_with_video(as_list = False)
        
        no_frame_dir = [
            _ for _ in interventions if not generate_frame_path(
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

