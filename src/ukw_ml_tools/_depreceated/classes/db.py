from pathlib import Path
import json
import warnings
from bson.objectid import ObjectId
from pymongo import MongoClient, aggregation, UpdateOne
from .db_interventions import DbInterventions
from .db_images import DbImages
from .db_tests import DbTests
from .intervention import Intervention
from .utils import *
from .fastcat_annotation import FastCatAnnotation
from .labelstudio import Labelstudio
from typing import List, Tuple
from collections import Counter
from .fieldnames import *
import pandas as pd
from tqdm import tqdm
from .db_extern import DbExtern
from datetime import datetime as dt


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
        self.db_extern_time_format = self.cfg["db_extern_date_pattern"]
        self.db_extern_user_priority = self.cfg["db_extern_annotation_user_priority"]
        self.db_extern_test_data_annotation_group_ids = self.cfg["db_extern_test_data_annotation_group_ids"]

        # Setup Mongo Db connection
        self.mongo_client = MongoClient(self.url_mongodb, connectTimeoutMS=200, retryWrites=True)
        self.db_images = DbImages(self.mongo_client.EndoData.images, self.cfg)
        self.db_interventions = DbInterventions(self.mongo_client.EndoData.interventions, self.cfg)
        self.db_tests = DbTests(self.mongo_client.EndoData.test_data, self.db_images, self.db_interventions, self.cfg)
        self.db_stats = self.mongo_client.EndoData.stats

        # Setup db_extern
        self.db_extern = DbExtern(self.cfg)

        # Setup Labelstudio connection
        self.labelstudio = Labelstudio(
            ls_url=self.labelstudio_prefix,
            ls_token = self.labelstudio_token,
            ls_config_path=self.paths["ls_configs"],
            cfg = self.cfg
            )

    def import_new_extern_interventions(self):
        new_interventions = self.db_extern.get_extern_interventions()
        new_interventions = [self.db_extern.convert_intervention_external_to_internal(_) for _ in new_interventions]
        print(f"Found {len(new_interventions)} interventions in external db")
        imported = 0
        failed = []
        for intervention in tqdm(new_interventions):
            try:
                _ = self.db_interventions.create_new_from_external(intervention)
            except:
                _ = None
                failed.append(intervention)

            if _:
                imported += 1
        print(f"Imported {imported} new interventions")
        print(f"{len(failed)} failed")

        return failed

    def calculate_stats(self):
        stats = self.db_interventions.calculate_stats() + self.db_images.calculate_stats()
        self.db_stats.insert_one({"type": "count_data", "data": stats})

    def update_all(self):
        log = {}
        print(dt.now(), "Import new from Extern")
        log["import_extern"] = self.import_new_extern_interventions()
        print(dt.now(), "Calculate Stats")
        self.calculate_stats()
        print(dt.now(), "Update Video Metadata")
        log["metadata"] = self.update_all_video_metadata(only_new = True)
        print(dt.now(), "Update Interventions' Image Annotations")
        self.update_all_intervention_image_annotations()
        print(dt.now(), "Update Interventions' Image Predictions")
        self.update_all_intervention_image_predictions()
        print(dt.now(), "Applying Terminology")
        self.db_interventions.apply_terminology(set_tokens_in_db = True, return_tokens = True)
        return log

    def update_all_intervention_image_annotations(self):
        interventions = self.db_interventions.db.find({})
        interventions = [_ for _ in interventions]
        
        for intervention in tqdm(interventions):
            intervention = Intervention(intervention["_id"], self.db_images.db, self.db_interventions.db, self.cfg)
            intervention.update_image_annotations()

    def update_all_intervention_image_predictions(self):
        interventions = self.db_interventions.db.find({})
        interventions = [_ for _ in interventions]
        
        for intervention in tqdm(interventions):
            intervention = Intervention(intervention["_id"], self.db_images.db, self.db_interventions.db, self.cfg)
            intervention.update_image_predictions()

    def update_all_video_metadata(self, only_new: bool = True):
        q = {FIELDNAME_VIDEO_PATH: {"$exists": True, "$nin": ["", None]}}
        if only_new:
            q[FIELDNAME_VIDEO_METADATA] = {"$exists": False}

        interventions = self.db_interventions.db.aggregate([{"$match":q}])
        interventions = [_ for _ in interventions]

        failed = []

        for intervention in tqdm(interventions):
            try:
                intervention = Intervention(intervention["_id"], self.db_images.db, self.db_interventions.db, self.cfg)
                meta = intervention.get_video_metadata()
                self.db_interventions.db.update_one({"_id": intervention._id}, {"$set": {FIELDNAME_VIDEO_METADATA: meta}})
            except:
                failed.append(str(intervention._id))

        if failed:
            print(f"Could not create metadata for {len(failed)} videos")

        return failed

    def get_stats(self, as_df: bool = True):
        _records = self.db_stats.find({"type": "count_data"})
        records = []
        for record in _records:
            records.extend((record["data"]))

        if as_df:
            return pd.DataFrame.from_records(records)
        else:
            return records 

    def get_metadata_for_train_data(self, metadata, label, value, extend_conditions = None, extend_agg = None):
        # assert label not in metadata
        metadata = metadata.copy()

        test_intervention_ids = self.db_tests.get_test_data_interventions(label) 
        aggregation = get_train_image_query(label, value, test_intervention_ids, extend_conditions, extend_agg)

        aggregation.append(
            {
                "$group": {
                    "_id": {
                        FIELDNAME_ORIGIN: "$"+FIELDNAME_ORIGIN,
                        FIELDNAME_INTERVENTION_ID: "$"+FIELDNAME_INTERVENTION_ID,
                        label: f"${FIELDNAME_LABELS}.{label}"
                    },
                    "count": {"$sum":1}
                }
            }
        )
        _counts = self.db_images.db.aggregate(aggregation)
        origin_count = defaultdict(generate_list_default_dict)

        for _count in _counts:
            origin = _count["_id"][FIELDNAME_ORIGIN]
            intervention = str(_count["_id"][FIELDNAME_INTERVENTION_ID])
            value = _count["_id"][label]
            origin_count[origin][intervention].append(_count["count"])

        for origin in origin_count.keys():
            for intervention in origin_count[origin].keys():
                count = origin_count[origin][intervention]
                origin_count[origin][intervention] = count

        aggregation = get_train_image_query(label, value, test_intervention_ids, extend_conditions, extend_agg)
        aggregation.append(
            {
                "$group": {
                    "_id": {
                        FIELDNAME_ORIGIN: "$"+FIELDNAME_ORIGIN,
                        # FIELDNAME_INTERVENTION_ID: "$"+FIELDNAME_INTERVENTION_ID,
                        label: f"${FIELDNAME_LABELS}.{label}"
                    },
                    "count": {"$sum":1}
                }
            }
        )
        _counts = self.db_images.db.aggregate(aggregation)
        origin_total = {_["_id"][FIELDNAME_ORIGIN]: _["count"] for _ in _counts}


        intervention_dates = {}
        for origin in origin_count.keys():
            _intervention_ids = [ObjectId(_) for _ in origin_count[origin].keys()]
            r = self.db_interventions.db.aggregate([
                {"$match": {"_id": {"$in": _intervention_ids}}},
                {"$group": {
                    "_id": None,
                    "min_date": {"$min": "$"+FIELDNAME_INTERVENTION_DATE},
                    "max_date": {"$max": "$"+FIELDNAME_INTERVENTION_DATE}
                }}
            ]).next()
            if r["min_date"]:
                min_date = dt.strftime(r["min_date"], "%Y-%-m-%d")
                max_date = dt.strftime(r["max_date"], "%Y-%-m-%d")
            else:
                min_date = None
                max_date = None

            intervention_dates[origin] = (min_date,max_date)
        
        n_interventions_origin = {origin: len(list(origin_count[origin].keys())) for origin in origin_count.keys()}
        metadata[label][value] = {
            "origin_total_interventions": n_interventions_origin,
            "origin_total_images": origin_total,
            "intervention_dates": intervention_dates,
            "origin_count": origin_count,
        }

        return metadata

    def get_train_df(self, label: str, min_frame_diff: int = None, verbose: bool = False, return_meta: bool = False):
        settings = self.ai_model_settings[label]
        if return_meta:
            metadata = defaultdict(dict)

        if settings["prediction_type"] == "binary":
            pos_imgs = self.get_train_img_list(label, True, min_frame_diff, verbose)              
            n_pos = len(pos_imgs)
            if return_meta:
                metadata = self.get_metadata_for_train_data(metadata, label, True)
                metadata[label][True]["n_images"] = n_pos

            neg_imgs = self.get_train_img_list(label, False, min_frame_diff, verbose)
            if return_meta:
                metadata = self.get_metadata_for_train_data(metadata, label, False)
                metadata[label][False]["n_images"] = len(neg_imgs)

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

                if return_meta:
                    metadata = self.get_metadata_for_train_data(metadata, _label, True, ext_conditions, extend_agg)
                    metadata[label][False]["n_images"] = len(neg_imgs)

            df_dict = {
                "file_path": [_["path"] for _ in pos_imgs],
                "label": [1 for _ in pos_imgs]
                }

            df_dict["file_path"].extend([_["path"] for _ in neg_imgs])
            df_dict["label"].extend([0 for _ in neg_imgs])

            label_df = pd.DataFrame.from_dict(df_dict).drop_duplicates()

            if return_meta:
                return label_df, metadata
            else: 
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
            imgs_true = self.filter_images_by_frame_diff(imgs_true, min_frame_diff)

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

    def get_videos_with_new_annotations(self):
        annotations = self.db_extern.get_extern_annotations()
        annotations = [
            _ for _ in annotations if 
            check_extern_video_annotation_timestamp(
                _["videoName"], _["lastAnnotationSession"], self.db_interventions.db)
            ]
        
        return annotations

    def update_intervention_annotations_from_extern(self, video_key, str_timestamp, time_format = None, annotation_priority_user_id = None):
        if not time_format:
            time_format = self.db_extern_time_format
        if not annotation_priority_user_id:
            annotation_priority_user_id = self.db_extern_user_priority

        timestamp = dt.strptime(str_timestamp, time_format)

        intervention = Intervention(video_key, self.db_images.db, self.db_interventions.db, self.cfg, True)
        annotations = self.db_extern.get_extern_video_annotation(video_key)

        video_annotation_dict = generate_db_extern_video_annotation_dict(annotations, time_format)
        video_annotations = select_relevant_extern_video_annotations(video_annotation_dict, annotation_priority_user_id)
        test_label_groups = []
        for _ in video_annotations:
            if str(_["label_group_id"]) in self.db_extern_test_data_annotation_group_ids:
                test_label_groups.append(
                    self.db_extern_test_data_annotation_group_ids[str(_["label_group_id"])]
                )

        if test_label_groups:
            for test_label_group in test_label_groups:
                video_annotations = add_false_labels_for_test_data_annotations(
                    intervention.intervention, video_annotations, test_label_group
                )

        video_flanks = defaultdict(dict)
        for _annotation in video_annotations:
            _label = _annotation["label"]
            _value = str(_annotation["value"])
            if _value not in video_flanks[_label]:
                video_flanks[_label][_value] = []
            video_flanks[_label][_value].append(_annotation["flank"])
        # video_annotations

        # For Michael Frame Extraction, just extract flanks
        frames = []
        for _annotation in video_annotations:
            frames.extend([_ for _ in range(_annotation["flank"][0], _annotation["flank"][1])])
        intervention.extract_frames(frames)

        updates = []
        for flank in video_annotations:
            flank_updates = [
                UpdateOne(
                    {"_id": intervention.intervention[FIELDNAME_FRAMES][str(_)]},
                    {
                        "$set": {
                            f"{FIELDNAME_LABELS}.{flank['label']}": flank["value"]
                        }
                    }
                )
                for _ in range(flank["flank"][0], flank["flank"][1]-2)
            ]

            updates.extend(flank_updates)
        r = self.db_images.db.bulk_write(updates)
        r = self.db_interventions.db.update_one(
            {"_id": intervention._id},
            {"$set": {FIELDNAME_ANNOTATION_FLANKS:video_flanks}})

        intervention.update_image_annotations()

        # Set Latest intervention_date
        self.db_interventions.db.update_one({"_id": intervention._id}, {"$set": {FIELDNAME_VIDEO_ANNOTATION_TIMESTAMP: timestamp}})

        return video_annotations



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
        interventions = [_ for _ in self.db_interventions.db.find(fieldvalue_in_list_query("_id", _ids))]

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
        validation_errors = {}
        print(dt.now(), "Validate Video Keys")
        validation_errors["video_keys"] = self.db_interventions.validate_video_keys()
        print(dt.now(), "Validate video_paths")
        validation_errors["video_paths"] = self.db_interventions.validate_video_paths()
        print(dt.now(), "Validate frame_dirs")
        validation_errors["frame_dirs"] = self.db_interventions.validate_frame_dirs()
        print(dt.now(), "Validate image_paths")
        validation_errors["image_paths"] = self.db_images.validate_image_paths()
        print(dt.now(), "Validate intervention_ids")
        validation_errors["intervention_ids"] = self.db_images.validate_all_intervention_ids_exist()
        print(dt.now(), "Validate frame_intervention_ids")
        validation_errors["frame_intervention_ids"] = self.validate_intervention_id_of_frames()

        return validation_errors

    def validate_config(self, cfg:dict):
        pass
