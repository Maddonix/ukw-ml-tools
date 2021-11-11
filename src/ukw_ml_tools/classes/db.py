from pathlib import Path
import json
import warnings
from bson.objectid import ObjectId
from pymongo import MongoClient, aggregation
from .db_interventions import DbInterventions
from .db_images import DbImages
from .db_tests import DbTests
from .intervention import Intervention
from .utils import *
from .fastcat_annotation import FastCatAnnotation
from .labelstudio import Labelstudio
from typing import List, Tuple
from .fieldnames import *
import pandas as pd
from tqdm import tqdm
from .db_extern import DbExtern


class Db:
    '''
    Add Documentation
    '''
    def __init__(self, path_config: Path):
        with open(path_config, "r", encoding = "utf-8") as f:
            self.cfg = json.load(f)
        
        # URLS
        self.url_mongodb = self.cfg["uri"]
        self.url_endobox_extreme = self.cfg["url_endobox_extreme"]
        self.url_gpu_server = self.cfg["url_gpu_server"]
        port_fileserver = self.cfg["port_fileserver"]
        port_labelstudio = self.cfg["port_labelstudio"]

        self.fileserver_prefix = self.url_endobox_extreme + ":" + port_fileserver + "/"
        self.labelstudio_prefix = self.url_endobox_extreme + ":" + port_labelstudio + "/api/"
        self.labelstudio_token = self.cfg["labelstudio_token"]
        self.streamlit_config = self.cfg["streamlit_config"]
        
        # Paths
        base_path_stats = Path(self.cfg["base_path_stats"])
        self.paths = {
            "models": Path(self.cfg["base_path_models"]),
            "stats": {
                "base": base_path_stats,
                "label_count": base_path_stats.joinpath("labelcount.json"),
                "label_by_origin": base_path_stats.joinpath("label_by_origin_count.json"),
                "label_by_video": base_path_stats.joinpath("video_origin_count.json")
            },
            "frames": Path(self.cfg["base_path_frames"]),
            "ls_configs": Path(self.cfg["path_label_configs"])
        }
        
        # Misc
        self.fast_cat_allowed_labels = self.cfg["fast_cat_allowed_labels"]
        self.skip_frame_factor = self.cfg["skip_frame_factor"]
        self.ai_model_settings = self.cfg["models"]
        self.exclude_origins_for_frame_diff_filter = self.cfg["exclude_origins_for_frame_diff_filter"]

        # Setup Mongo Db connection
        self.mongo_client = MongoClient(self.url_mongodb, connectTimeoutMS=200, retryWrites=True)
        self.db_images = DbImages(self.mongo_client.EndoData.images, self.cfg)
        self.db_interventions = DbInterventions(self.mongo_client.EndoData.interventions, self.cfg)
        self.db_tests = DbTests(self.mongo_client.EndoData.test_data, self.db_images, self.db_interventions, self.cfg)

        # Setup db_extern
        self.db_extern = DbExtern(self.cfg)

        # Setup Labelstudio connection
        self.labelstudio = Labelstudio(
            ls_url=self.labelstudio_prefix,
            ls_token = self.labelstudio_token,
            ls_config_path=self.paths["ls_configs"],
            cfg = self.cfg
            )


    def get_train_df(self, label: str, min_frame_diff: int = None, verbose: bool = False):
        settings = self.ai_model_settings[label]

        if settings["prediction_type"] == "binary":
            pos_imgs = self.get_train_img_list(label, True, min_frame_diff, verbose)
            n_pos = len(pos_imgs)

            neg_imgs = self.get_train_img_list(label, False, min_frame_diff, verbose)

            for _label, _multiplier in settings["neg_label_list"].items():
                if _label == "normal":
                    ext_conditions = [{f"{FIELDNAME_LABELS}.{label}": {"$exists": False}}]

                    extend_agg = [{"$sample": {"size": int(_multiplier * n_pos)}}]
                    _neg_imgs = self.get_train_img_list(_label, True, min_frame_diff, extend_conditions=ext_conditions, extend_agg = extend_agg)
                    neg_imgs.extend(_neg_imgs)

                elif self.ai_model_settings[_label]["prediction_type"] == "binary":
                    ext_conditions = [{f"{FIELDNAME_LABELS}.{label}": {"$exists": False}}]

                    extend_agg = [{"$sample": {"size": int(_multiplier * n_pos)}}]
                    _neg_imgs = self.get_train_img_list(_label, True, min_frame_diff, extend_conditions=ext_conditions, extend_agg = extend_agg)
                    neg_imgs.extend(_neg_imgs)
                
                else:
                    assert 0 == 1

            df_dict = {
                "file_path": [_["path"] for _ in pos_imgs],
                "label": [1 for _ in pos_imgs]
                }
            df_dict["file_path"].extend([_["path"] for _ in neg_imgs])
            df_dict["label"].extend([0 for _ in neg_imgs])

            label_df = pd.DataFrame.from_dict(df_dict).drop_duplicates()

            return label_df

    def import_fast_cat_annotation(self, video_key: str, annotation_path: Path):
        intervention = [_ for _ in self.db_interventions.db.find({FIELDNAME_VIDEO_KEY: video_key})]
        assert len(intervention) == 1
        assert annotation_path.exists()
        intervention = intervention[0]
        annotation = FastCatAnnotation(annotation_path, self.cfg)
        intervention.add_frame_labels(annotation.labels) 

    def get_intervention_frame_labels(self, intervention_id: ObjectId = None):
        if intervention_id:
            interventions = self.db_interventions.db.find({"_id": intervention_id})
        else:
            interventions = self.db_interventions.db.find({"frames": {"$exists": True}})
        
        intervention_label_dict = {}

        for intervention in interventions:
            intervention = Intervention(intervention["_id"], self.db_images.db, self.db_interventions.db, self.cfg)
            unique_labels = intervention.get_unique_frame_labels()
            intervention_label_dict[intervention["_id"]] = unique_labels
            self.db_interventions.db.update_one({"_id": intervention["_id"]}, {"$set": {FIELDNAME_LABELS_FRAMES: unique_labels}})

        return intervention_label_dict

    def get_train_img_list(self, label: str, value, min_frame_diff: int = None, extend_conditions: List = None, extend_agg: List = None, verbose: bool = False):
        test_intervention_ids = self.db_tests.get_test_data_interventions(label) 
        imgs_true = self.db_images.get_train_images(label, value, test_intervention_ids, extend_conditions, extend_agg, as_list = True)

        if min_frame_diff:
            imgs_true = self.filter_images_by_frame_diff(imgs_true, self.min_frame_diff)

        return imgs_true       

    def filter_images_by_frame_diff(self, img_list: List[dict], min_frame_diff: int = 10):
        filtered_images_dict = defaultdict(list)

        for img in img_list:
            intervention_id = img[FIELDNAME_INTERVENTION_ID]

            if img[FIELDNAME_ORIGIN] in self.exclude_origins_for_frame_diff_filter:
                filtered_images_dict[intervention_id].append(img)
                continue

            append = True
            n_frame = img[FIELDNAME_FRAME_NUMBER]
            
            if filtered_images_dict[intervention_id]:
                for _img in filtered_images_dict[intervention_id]:
                    frame_diff = abs(_img[FIELDNAME_FRAME_NUMBER] - n_frame)
                    if frame_diff < min_frame_diff:
                        append = False
                        break
            
            if append:
                filtered_images_dict[intervention_id].append(img)

        filtered_images = []
        for key, value in filtered_images_dict.items():
            filtered_images.extend(value)
        
        return filtered_images

    # Validation
    def validate_intervention_id_of_frames(self) -> tuple((dict, List[ObjectId])):
        interventions = [_ for _ in self.db_interventions.db.find(field_exists_query(FIELDNAME_FRAMES))]
        wrong_ids = defaultdict(list)
        not_all_frames_exist = []

        for intervention in tqdm(interventions):
            frame_ids = [_id for n_frame, _id in intervention[FIELDNAME_FRAMES].items()]
            images = self.db_images.db.find(
                fieldvalue_in_list_query("_id", frame_ids)
            )

            n_frames_extracted = len([_ for _ in intervention[FIELDNAME_FRAMES].keys()])
            n_frames_found = 0
            
            for image in images:
                n_frames_found += 1
                if not image[FIELDNAME_INTERVENTION_ID] == intervention["_id"]:
                    wrong_ids[intervention["_id"]].append(image["_id"])

            if n_frames_found != n_frames_extracted:
                not_all_frames_exist.append(intervention["_id"])

        if wrong_ids:
            warnings.warn("Interventions have frames whose intervention id doesnt match")

        if not_all_frames_exist:
            warnings.warn("Interventions have extracted frames which were not found")

        return (wrong_ids, not_all_frames_exist)            

    def validate_all_intervention_ids_exist(self) -> List:
        _ids = self.db_images.db.distinct(FIELDNAME_INTERVENTION_ID)
        interventions = [_ for _ in self.db_interventions.find(fieldvalue_in_list_query("_id", _ids))]

        if not len(_ids) == len(interventions):
            warnings.warn("Image DB points to interventions which were not found")

    def set_image_type(self):
        interventions = self.db_interventions.get_interventions_with_frames()
        for intervention in tqdm(interventions):
            img_ids = [_ for _ in intervention[FIELDNAME_FRAMES].keys()]
            self.db_images.db.update_many(fieldvalue_in_list_query("_id", img_ids), {"$set": {FIELDNAME_IMAGE_TYPE: IMAGETYPE_FRAME}})

        interventions = self.db_interventions.get_interventions_with_freezes()
        for intervention in tqdm(interventions):
            img_ids = [_ for _ in intervention[FIELDNAME_FREEZES].keys()]
            self.db_images.db.update_many(fieldvalue_in_list_query("_id", img_ids), {"$set": {FIELDNAME_IMAGE_TYPE: IMAGETYPE_FREEZE}})

    def validate_all(self):
        self.db_interventions.validate_video_keys()
        self.db_interventions.validate_video_paths()
        self.db_interventions.validate_frame_dirs()
        self.db_images.validate_image_paths()
        self.db_images.validate_all_intervention_ids_exist()
        self.validate_intervention_id_of_frames()
        self.validate_all_intervention_ids_exist()

    def validate_config(self, cfg:dict):
        pass
