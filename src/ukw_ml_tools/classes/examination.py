from pathlib import Path
import pandas as pd
import plotly.express as px
from ..mongodb.results import get_intervention_labels
import os
from ..wt.metrics import calculate_metrics, calculate_times
from ..wt.utils import split_result_df, pred_update_to_object
from ..datasets.wt import get_dataset
from torch.utils.data import DataLoader
from ..models.wt import load_model, predict_batch
from ..labels.smoothing import calculate_intervention_smooth_labels
from tqdm import tqdm

COLOR_MAP = {0: "blue", 1: "aqua", 2: "red", 3: "green", 4: "darkgoldenrod", 5: "hotpink",
    "outside": "blue", "insertion": "aqua", "caecum": "yellow", "withdrawal": "orange", "resection": "red",
    "low_quality": "grey", "water_jet": "blue", "snare": "green", "grasper": "hotpink", "nbi": "purple"
}

class Examination:
    def __init__(self, intervention, db = None, tmp_dir = Path("tmp")):
        self.intervention = intervention
        self.label_types = ["annotations", "predictions", "predictions_smooth"]
        self.db = db
        self.result_df = None
        self.segmentation_df = None
        if not tmp_dir.exists():
            os.mkdir(tmp_dir)
        self.tmp_path = tmp_dir.joinpath(self.intervention.video_key).with_suffix(".feather")

    def get_results(self, db_images = None, frame_list = None, refresh = False):
        if not db_images:
            db_images = self.db.Images
        df = pd.DataFrame()
        tmp_exists = self.tmp_path.exists()
        if not tmp_exists or refresh:
            for label_type in self.label_types:
                _records = get_intervention_labels(self.intervention.id, db_images, label_type)
                if _records:
                    _df = pd.DataFrame.from_records(_records)
                    _df = _df.loc[:,["video_key", "frame_number", "_type", "name", "value"]]

                    _df = _df.sort_values("frame_number")
                    if frame_list:
                        select = _df.frame_number.isin(frame_list)
                        _df = _df[select]
                    df = df.append(_df)

            df = df.loc[~df.name.isin(["colo_segmentation","caecum"])]
            df["value"] = df["value"].astype(bool)
            _df = df.reset_index(drop = True)
            _df.to_feather(self.tmp_path)
        else:
            df = pd.read_feather(self.tmp_path)
        self.result_df = df
        return df

    def df_dict(self):
        fps = self.intervention.metadata.fps

        return split_result_df(
            self.result_df,
            fps = fps
            )

    def label_count(self):
        df_dict = self.df_dict()
        summary = []
        for key, value in df_dict.items():
            record = {
                "name": key,
            }
            for label in value.columns:
                record[label] = value[label].sum()

            summary.append(record)
        label_count = pd.DataFrame.from_records(summary)
        return label_count

    def get_plot(self, width = 1100, height = 300, font_size = 18, title_font_size = 18, marker_size = 20):
        video_key = self.intervention.video_key
        if self.result_df is None:
            assert self.db
            db_images = self.db.Images
            self.get_results(db_images)

        _colo_result = self.result_df[self.result_df.value==True]
        _colo_result = _colo_result.loc[_colo_result.name.isin([
            "outside", "low_quality", "tool", "nbi", "water_jet",
            "appendix", "ileum", "caecum", "ileocaecalvalve",
            "clip", "grasper", "snare", "needle",
            "polyp"
            ])]

        if not len(_colo_result):
            return None
        plot = px.scatter(
            _colo_result,
            x="frame_number",
            y="_type",
            color="name",
            title=video_key,
            symbol="name",
            symbol_sequence=["line-ns", "line-ns"],
            width=width,
            height=height,
            category_orders={
                    "_type": ["annotations", "predictions", "predictions_smooth"],
                    "name": ["outside","caecum", "tool", "nbi"]
            },
            color_discrete_map= COLOR_MAP
            )
        plot.update_traces(
            marker=dict(size=marker_size),
            selector=dict(mode='markers'),    
            )

        plot.update_layout(
            font_size = font_size,
            title_font_size=title_font_size
        )
        self.plot = plot
        return plot

    def get_segmentation_plot(self, width = 1100, height = 500, font_size = 18, title_font_size = 18, marker_size = 20):
        video_key = self.intervention.video_key
        if not len(self.segmentation_df):
            self.calculate_metrics()

        segmentation_df = self.segmentation_df

        if not len(segmentation_df):
            return None
        plot = px.scatter(
            segmentation_df,
            x="frame_number",
            y="_type",
            color="segment",
            title=video_key,
            symbol="segment",
            symbol_sequence=["line-ns", "line-ns"],
            width=width,
            height=height,
            category_orders={
                    "_type": ["annotation", "prediction", "prediction_smooth", "y_pred_post_processing"], 
                    "segment": ["outside","insertion", "withdrawal", "caecum", "resection"],
            },
            color_discrete_map= COLOR_MAP
            )
        plot.update_traces(
            marker=dict(size=marker_size),
            selector=dict(mode='markers'),    
            )

        plot.update_layout(
            font_size = font_size,
            title_font_size=title_font_size
        )
        self.plot = plot
        return plot

    def generate_summary(self, refresh = False):
        self.get_results(refresh = refresh)
        self.calculate_metrics()
        records = []

        record = self.times["annotation"].copy()
        record.update({"_type": "annotation", "video_key": self.intervention.video_key})
        records.append(record)

        record = self.times["y_pred_post_processing"].copy()
        record.update({"_type": "y_pred_post_processing", "video_key": self.intervention.video_key})
        records.append(record)

        return records        

    def calculate_metrics(self):
        if not len(self.result_df):
            self.get_results()
        result_df = self.result_df.copy()
        fps = self.intervention.metadata.fps
        df_dict = split_result_df(result_df)

        self.times = {}
        self.segmentation = {}
        self.times["annotation"], self.segmentation["annotation"] = calculate_times(df_dict["annotations"].copy(), fps)
        self.times["prediction"], self.segmentation["prediction"] = calculate_times(df_dict["predictions"].copy(), fps)
        self.times["prediction_smooth"], self.segmentation["prediction_smooth"] = calculate_times(df_dict["predictions_smooth"].copy(), fps)
        self.times["y_pred_post_processing"], self.segmentation["y_pred_post_processing"] = calculate_times(df_dict["y_pred_post_processing"].copy(), fps)

        self.segmentation_df = pd.DataFrame()
        for key, value in self.segmentation.items():
            value["_type"] = key
            self.segmentation_df = self.segmentation_df.append(value)

        metrics, metrics_smooth = calculate_metrics(df_dict)
        self.metrics = {
            "metrics": metrics,
            "mertrics_smooth": metrics_smooth,
        }

    def dataloader(self):
        db_images = self.db.Images
        intervention = self.intervention
        images = db_images.find({"video_key": intervention.video_key})
        
        images = [_ for _ in images]
        ds = get_dataset(images)
        loader = DataLoader(
            ds, batch_size = 32, num_workers = 12,
            shuffle = False
        )

        print(f"N Frames: {self.intervention.metadata.frames_total}; found: {len(ds)}")

        return loader

    def load_model(self, model_path):
        return load_model(model_path)

    def reset_intervention_predictions(self, labels):
        intervention = self.intervention
        db_images = self.db.Images
        for name in labels:
            db_images.update_many({"intervention_id": intervention.id}, {"$set": {f"predictions.{name}": {
                "name": name,
                "version": -1,
                "value": False,
                "raw": [],
                "choices": [False, True],
                "label_type": "multilabel"
            }}})
            db_images.update_many({"intervention_id": intervention.id}, {"$set": {f"predictions_smooth.{name}": {
                "name": name,
                "version": -1,
                "value": False,
                "raw": [],
                "choices": [False, True],
                "label_type": "multilabel"
            }}})

    def predict(self, model_path, conv_len = None, future_frames = True):
        loader = self.dataloader()
        db_images = self.db.Images
        intervention = self.intervention
        model = load_model(model_path)
        if not conv_len:
            conv_len = self.intervention.metadata.fps

        print("Reset Predictions")
        self.reset_intervention_predictions(model.labels)
        
        print("Predict")
        print(len(loader.dataset))
        for batch in tqdm(loader):
            prediction_updates = predict_batch(batch, model)
            i = True
            for key, value in prediction_updates.items():
                _update = pred_update_to_object(value, threshold = 0.5)
                if i:
                    print(_update)
                    i = False
                
                db_images.update_one({"_id": key}, {"$set": {"predictions": _update}})

        print("Calculate Smooth Labels")
        for label in model.labels:
            print(label)
            calculate_intervention_smooth_labels(label, conv_len, intervention.id, -1, db_images, future_frames)



