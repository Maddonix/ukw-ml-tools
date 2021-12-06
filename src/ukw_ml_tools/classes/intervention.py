from typing import Collection, Iterable
from bson.objectid import ObjectId
from pymongo.collection import Collection
from tqdm import tqdm
from .ai_predictor import AiPredictor
from .fieldnames import FIELDNAME_VIDEO_KEY, FIELDNAME_VIDEO_PATH
from .terminology import Terminology
from ukw_ml_tools.db.crud import delete_frames_from_db
from .utils import *
from .fieldnames import *
import cv2
import pandas as pd

class Intervention:
    """asd
    asd
    """
    def __init__(self, identifier, db_images: Collection, db_interventions: Collection, cfg: dict, id_is_video_key: bool = False):
        self.db_images = db_images
        self.db_interventions = db_interventions
        self.cfg = cfg
        self.base_path_frames = Path(cfg["base_path_frames"])
        if id_is_video_key:
            self.intervention = self.db_interventions.find_one({FIELDNAME_VIDEO_KEY: identifier})
        else:
            self.intervention = self.db_interventions.find_one({"_id": identifier})
        
        assert self.intervention
        self._id = self.intervention["_id"]
        self.video_key = get_if_in_dict(self.intervention, [FIELDNAME_VIDEO_KEY])
        if self.video_key != None:
            self.frame_dir = self.base_path_frames.joinpath(self.intervention[FIELDNAME_VIDEO_KEY])
        else: self.frame_dir = None
        self.skip_frame_factor = cfg["skip_frame_factor"]
        # self.prediction_result: pd.DataFrame = None
        if not FIELDNAME_INTERVENTION_IMAGE_PREDICTIONS in self.intervention:
            self.update_image_predictions()


    def update_image_annotations(self) -> List[str]:
        labels = get_intervention_image_annotations(self._id, self.db_images)
        self.db_interventions.update_one({"_id": self._id}, {"$set": {FIELDNAME_INTERVENTION_IMAGE_ANNOTATIONS: labels}})
        self.refresh()

        return labels

    def update_image_predictions(self):
        predictions = get_intervention_image_predictions(self._id, self.db_images)
        self.db_interventions.update_one({"_id": self._id}, {"$set": {FIELDNAME_INTERVENTION_IMAGE_PREDICTIONS: predictions}})        
        self.refresh()
        return predictions

    def update_smooth_labels(self):
        assert FIELDNAME_VIDEO_METADATA in self.intervention
        assert FIELDNAME_FPS in self.intervention[FIELDNAME_VIDEO_METADATA]

        ####################################
        n_smooth = int(self.intervention[FIELDNAME_VIDEO_METADATA][FIELDNAME_FPS]/2)
        threshold = 0.5
        ####################################

        df = self.get_frame_result_df(n_smooth, threshold)
        df = df.loc[df[COLNAME_RESULT_TYPE]==RESULT_TYPE_PREDICTION_SMOOTH]
        ####################################
        df = df.loc[df[COLNAME_RESULT_NAME]!= "blurry"]
        ####################################

        for index, row in df.iterrows():
            self.db_images.update_one(
                {"_id": row["_id"]},
                {"$set": {f"{FIELDNAME_PREDICTIONS_SMOOTH}.{row[COLNAME_RESULT_NAME]}": row[COLNAME_RESULT_LABEL]}})

        self.refresh()

    def calculate_test_result(self, n_smooth, use_future, running_mean_treshold):
        self.get_frame_result_df(n_smooth, running_mean_treshold, use_future)
        

    def update_tokens(self):
        terminology = Terminology(self.cfg)

        report = get_if_in_dict(self.intervention, [FIELDNAME_REPORT_RAW])
        if report:
            tokens_report = terminology.get_terminology_result(report)
        else: 
            tokens_report = []
        patho = get_if_in_dict(self.intervention, [FIELDNAME_PATHO_RAW])
        if patho:
            tokens_patho = terminology.get_terminology_result(patho)
        else: 
            tokens_patho = []

        update = {
            FIELDNAME_TOKENS: {
                FIELDNAME_TOKENS_REPORT: tokens_report,
                FIELDNAME_TOKENS_PATHO: tokens_patho
            }
        }
        self.db_interventions.update_one({"_id": self._id}, {"$set": update})
        self.refresh()
        return update

    def get_video_metadata(self):
        return get_video_meta(Path(self.intervention[FIELDNAME_VIDEO_PATH]))

    def get_summary(self):
        flanks = get_if_in_dict(self.intervention, [FIELDNAME_PREDICTION_FLANKS])
        if flanks:
            flanks = {key: value[FIELDNAME_FLANK_RANGES] for key, value in flanks.items()}
        self.summary = {
            "_id": self.intervention["_id"],
            FIELDNAME_N_EXTRACTED_FRAMES: get_if_in_dict(self.intervention, [FIELDNAME_N_EXTRACTED_FRAMES]),
            FIELDNAME_FRAMES_TOTAL: get_if_in_dict(self.intervention, [FIELDNAME_FRAMES_TOTAL]),
            FIELDNAME_VIDEO_KEY: get_if_in_dict(self.intervention, [FIELDNAME_VIDEO_KEY]),
            FIELDNAME_VIDEO_PATH: get_if_in_dict(self.intervention, [FIELDNAME_VIDEO_PATH]),
            FIELDNAME_REPORT_RAW: get_if_in_dict(self.intervention, [FIELDNAME_REPORT_RAW]),
            FIELDNAME_PATHO_RAW: get_if_in_dict(self.intervention, [FIELDNAME_PATHO_RAW]),
            FIELDNAME_TOKENS: get_if_in_dict(self.intervention, [FIELDNAME_TOKENS]),
            FIELDNAME_INTERVENTION_IMAGE_PREDICTIONS: get_if_in_dict(self.intervention, [FIELDNAME_INTERVENTION_IMAGE_PREDICTIONS]),
            FIELDNAME_INTERVENTION_IMAGE_ANNOTATIONS: get_if_in_dict(self.intervention, [FIELDNAME_INTERVENTION_IMAGE_ANNOTATIONS]),
            FIELDNAME_PREDICTION_FLANKS: flanks
        }

        return self.summary

    def get_frame_id(self, n_frame: int):
        return self.intervention[FIELDNAME_FRAMES][str(n_frame)]

    def get_images_with_predictions(self, as_list: bool = False) -> Iterable:
        frame_ids = self.get_all_frame_ids()
        frames = self.db_images.find(
            {
                "_id": {
                    "$in": frame_ids
                },
                FIELDNAME_PREDICTIONS: {
                    "$nin": [{}]
                }
            }
        )

        if as_list: return [_ for _ in frames]
        else: return frames

    def get_images_with_annotations(self, as_list: bool = False):
        frame_ids = self.get_all_frame_ids()
        frames = self.db_images.find(
            {
                "_id": {
                    "$in": frame_ids
                },
                FIELDNAME_LABELS: {
                    "$nin": [{}]
                }
            }
        )
        if as_list: return [_ for _ in frames]
        else: return frames

    def get_all_frame_ids(self) -> List[ObjectId]:
        return [_id for n, _id in self.intervention[FIELDNAME_FRAMES].items()]

    def refresh(self):
        self.intervention = self.db_interventions.find_one({"_id": self._id})
        return self.intervention

    def get_prediction_result(self, as_list: bool = False) -> pd.DataFrame:
        frames_with_prediction = self.get_images_with_predictions()
        df_records = []

        for frame in frames_with_prediction:
            _predictions = frame[FIELDNAME_PREDICTIONS]
            predictions = [{
                "_id": frame["_id"],
                FIELDNAME_FRAME_NUMBER: frame[FIELDNAME_FRAME_NUMBER],
                COLNAME_RESULT_NAME: ai_name,
                COLNAME_RESULT_TYPE: RESULT_TYPE_PREDICTION,
                COLNAME_PREDICTION_VALUE: label_prediction[COLNAME_PREDICTION_VALUE],
                COLNAME_RESULT_LABEL: label_prediction[COLNAME_RESULT_LABEL],
                FIELDNAME_AI_VERSION: label_prediction[FIELDNAME_AI_VERSION] 
            } for ai_name, label_prediction in _predictions.items()]
            df_records.extend(predictions)

        if df_records:
            df = pd.DataFrame.from_records(df_records)
            df.sort_values([FIELDNAME_FRAME_NUMBER, COLNAME_PREDICTION_VALUE], ignore_index = True, inplace = True)
            self.prediction_df = df
        else:
            self.prediction_df = pd.DataFrame(
                columns = [
                    "_id",
                    FIELDNAME_FRAME_NUMBER,
                    COLNAME_RESULT_NAME,
                    COLNAME_RESULT_TYPE,
                    COLNAME_PREDICTION_VALUE,
                    COLNAME_RESULT_LABEL,
                    FIELDNAME_AI_VERSION                    
                ]
            ) 

        if as_list: return df_records
        else: return self.prediction_df

    def get_smooth_prediction_result(self, n_smooth: int = 10, threshold: float = 0.5, use_future: bool = True):
        names = self.prediction_df[COLNAME_RESULT_NAME].unique()

        for name in names:
            if name == "blurry":
                continue
            self.add_smooth_label_prediction(name, n_smooth, threshold, use_future)

    def get_annotation_result(self, as_list: bool = False) -> pd.DataFrame:
        frames = self.get_images_with_annotations()
        df_records = []

        for frame in frames:
            _labels = frame[FIELDNAME_LABELS]
            labels = [{
                "_id": frame["_id"],
                FIELDNAME_FRAME_NUMBER: frame[FIELDNAME_FRAME_NUMBER],
                COLNAME_RESULT_NAME: label,
                COLNAME_RESULT_TYPE: "annotation",
                COLNAME_RESULT_LABEL: choice,
            } for label, choice in _labels.items()]
            df_records.extend(labels)

        if df_records:
            df = pd.DataFrame.from_records(df_records)
            df.sort_values([FIELDNAME_FRAME_NUMBER], ignore_index = True,inplace=True)
            self.annotation_df = df
        else: 
            self.annotation_df = pd.DataFrame(columns = ["_id", FIELDNAME_FRAME_NUMBER, COLNAME_RESULT_NAME, COLNAME_RESULT_TYPE, COLNAME_RESULT_LABEL])

        if as_list: return df_records
        else: return self.annotation_df

    def get_frame_result_df(self, n_smooth: int = None, threshold: float = 0.5, use_future: bool = True):
        if n_smooth == None:
            n_smooth = int(self.intervention[FIELDNAME_VIDEO_METADATA][FIELDNAME_FPS]/1)
        self.get_annotation_result()
        self.get_prediction_result()
        self.get_smooth_prediction_result(n_smooth, threshold, use_future)
        self.frame_result_df = self.annotation_df.append(self.prediction_df)
        get_prediction_flanks(self)
        
        return self.frame_result_df

    def add_smooth_label_prediction(self, label, n_smooth: int = 10, threshold: float = 0.5, use_future: bool = False):
        smooth_prediction = get_smooth_prediction_df(self.prediction_df, label, n_smooth, threshold)
        self.prediction_df = self.prediction_df.append(smooth_prediction)

        return self.prediction_df

    def predict_all_frames(self, ai_name):
        frame_list = self.get_frame_list(as_list = True)

        predictor = AiPredictor(ai_name, self.cfg, self.db_images)
        predictor.base_paths_models = Path(self.cfg["base_path_models"])

        predictor.load_model()
        predictor.load_dataset(frame_list)
        predictor.load_dataloader()

        print(predictor.ai_name)
        print(predictor.specs)

        targets, predicted_batches = predictor.predict_images()

        for batch in predicted_batches:
            for _id, prediction in batch.items():
                self.db_images.update_one({"_id": _id}, {"$set": {f"{FIELDNAME_PREDICTIONS}.{ai_name}": prediction}})

        self.update_image_predictions()

    def prediction_df_long_to_wide(self, prediction_df: pd.DataFrame = None) -> pd.DataFrame:
        if not prediction_df:
            if self.prediction_df:
                prediction_df = self.prediction_df
            else:
                self.set_prediction_result()
                prediction_df = self.prediction_df

        return prediction_df_long_to_wide(prediction_df)
        
    def add_frame_labels(self, frame_labels: dict):
        for n_frame, _label in frame_labels.items():
            update_dict = self.frame_label_dict_to_update_dict(_label)
            self.db_images.update_one({"_id": self.get_frame_id(n_frame)}, {"$set": update_dict})

        self.refresh()

        return self.intervention

    def frame_label_dict_to_update_dict(self, frame_label_dict: dict):
        update_dict = {}
        for label, value in frame_label_dict.items():
            update_dict[f"{FIELDNAME_LABELS}.{label}"] = value

        return update_dict

    def get_frame_list(self, as_list: bool = True) -> List[dict]:
        frame_ids = [_id for n_frame, _id in self.intervention[FIELDNAME_FRAMES].items()]
        frame_cursor = self.db_images.find({"_id": {"$in": frame_ids}}) 
        if as_list:
            frame_list = [_ for _ in frame_cursor]
            return frame_list
        else:
            return frame_cursor

    def extract_frames(self, frame_list: List[int] = None, frame_suffix: str = ".jpg"):
        cap = cv2.VideoCapture(self.intervention[FIELDNAME_VIDEO_PATH])
        if not frame_list:
            frames_total = get_frames_total(cap)
            frame_list = [_ for _ in range(frames_total)]
            
        
        frame_list = [n_frame for n_frame in frame_list if str(n_frame) not in self.intervention[FIELDNAME_FRAMES]]
        frame_list.sort()
        # frame_dicts = []
        if not self.frame_dir.exists():
            os.mkdir(self.frame_dir)


        for n_frame in tqdm(frame_list):
            frame_path = self.frame_dir.joinpath(f"{n_frame}{frame_suffix}")
            

            cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
            success, image = cap.read()
            assert success

            assert cv2.imwrite(frame_path.as_posix(), image)

            db_image_entry = get_image_db_template(
                origin=self.intervention[FIELDNAME_ORIGIN],
                intervention_id=self.intervention["_id"],
                path=frame_path.as_posix(),
                n_frame=n_frame,
                image_type= IMAGETYPE_FRAME
            )
            self.db_images.insert(db_image_entry)
            # frame_dicts.append(db_image_entry)

        cap.release()
        if frame_list:
            frames = self.db_images.find({FIELDNAME_INTERVENTION_ID: self._id})
            update = {
                FIELDNAME_FRAMES: {
                    str(frame[FIELDNAME_FRAME_NUMBER]): frame["_id"]
                    for frame in frames 
                    } 
                }

            self.db_interventions.update_one({"_id": self._id}, {"$set": update})
            self.intervention = self.refresh()

            return frames

        self.refresh()

    def delete_frames(self, frame_list: List[int]):
        frame_ids = [self.intervention[FIELDNAME_FRAMES][str(n_frame)] for n_frame in frame_list]
        frames = self.db_images.find({"_id": {"$in": frame_ids}})
        for frame in frames:
            os.remove(frame[FIELDNAME_IMAGE_PATH])
        self.db_images.delete_many({"_id": {"$in": frame_ids}})
        self.db_interventions.update_one(
            {
                "_id": self._id
            },
            {
                "$unset": {f"{FIELDNAME_FRAMES}.{n_frame}": "" for n_frame in frame_list}
            }              
        )

        self.refresh()

