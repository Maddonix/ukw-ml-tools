from bson.objectid import ObjectId
from pydantic import BaseModel, validator
from typing import Union, List, Optional, Set
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .base import PyObjectId
from ..labels.strat_group_split import GradientSolver, generate_counts, create_initial_solution
from sklearn.utils import shuffle
from .utils import ORIGIN_LOOKUP
from ..metrics.base import group_stats


class TrainDataSamplerSettings(BaseModel):
    min_frame_diff: int
    test_sets: List[str]
    exclude_origins: List[str]


class TrainDataBase(BaseModel):
    name: str
    prediction_type: str
    choices: List[str]
    weights: Optional[List[float]]

    df_columns: Set = {
        "labels",
        "paths",
        "origins",
        "intervention_ids",
        "image_ids",
        "is_val",
        "crop"
    }

    class Config:
        allow_population_by_field_name = True
        # arbitrary_types_allowed = True
        json_encoders = {Path: str, ObjectId: str, Set: list}
        schema_extra = {
            "example": {}
        }

    def get_df_path(self, base_path_train_data):
        return base_path_train_data.joinpath(self.name).with_suffix(".csv")


class TrainDataDb(TrainDataBase):
    path: Path

    class Config:
        allow_population_by_field_name = True
        # arbitrary_types_allowed = True
        json_encoders = {Path: str, ObjectId: str, Set: list}
        schema_extra = {
            "example": {}
        }

    @validator("path")
    def path_to_str(cls, v):
        return v.as_posix()

    @validator("df_columns")
    def df_columns_to_list(cls, v):
        return list(v)

    def to_dict_intern(self):
        _dict = self.dict()
        df = pd.read_csv(self.path)
        df_dict = df.to_dict(orient="list")
        _dict.update(df_dict)
        return _dict


class TrainData(TrainDataBase):
    labels: Union[List[bool], List[int], List[List[int]]]
    origins: List[str]
    intervention_ids: List[PyObjectId]
    paths: List[Path]
    image_ids: List[PyObjectId]
    is_val: Optional[List[bool]]
    crop: List[List[int]]

    @validator("crop", pre=True)
    def str_to_list(cls, v):
        if isinstance(v[0], str):
            v = [_.split(";") for _ in v]
        return v

    def parsed_dict(self):
        return json.loads(self.json(by_alias=True, exclude_none=True))

    def n_classes(self):
        return len(self.choices)

    def get_loss_weights(self, target=None):
        if not target:
            target = self.labels
        class_sample_count = np.array(
            [len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        return weight

    def get_weights(self, target=None):
        if not target:
            target = self.labels
        class_sample_count = np.array(
            [len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target])
        return samples_weight

    def get_df(self):
        _dict = self.dict(
            include=self.df_columns
        )
        df = pd.DataFrame.from_dict(_dict)
        df.crop = df.crop.apply(lambda x : ";".join([str(_) for _ in x]))
        return df

    def save_df(self, base_path_train_data):
        path = self.get_df_path(base_path_train_data)
        df = self.get_df()
        df.to_csv(path, index=False)
        return path

    def to_db_dict(self, base_path_train_data):
        db_dict = self.dict(exclude=self.df_columns)
        db_dict["path"] = self.get_df_path(base_path_train_data)
        return db_dict

    def get_stats_df(self, choices):
        df = self.get_df()
        df = df.drop(labels=["paths", "image_ids", "crop"], axis=1)
        df["origins"] = df["origins"].apply(lambda x: ORIGIN_LOOKUP[x.lower()])
        df = df.sort_values("labels")
        df["labels"] = df["labels"].apply(lambda x: choices[x])
        stats_df = pd.DataFrame(columns=["is_val", "origins", "labels"])
        # stats_df = stats_df.append(group_stats(None, ["is_val", "origins", "labels"], df))
        stats_df = stats_df.append(group_stats("is_val", ["origins", "labels"], df))
        stats_df = stats_df.append(group_stats("origins", ["is_val", "labels"], df))
        stats_df = stats_df.append(group_stats("labels", ["is_val", "origins"], df))
        stats_df = stats_df.append(group_stats(["labels", "origins"], ["is_val"], df))
        return stats_df

    def group_strat_split(
        self,
        test_size=0.1,
        return_val_ids=False,
        max_empty_iterations=500,
        max_intensity_iterations=100,
        min_cost=100,
        verbose=False
    ):

        train_df = self.get_df()
        counts = train_df.groupby("intervention_ids")["labels"].value_counts().unstack().fillna(0)
        sample_cnt = counts.to_numpy()

        solution_arr = create_initial_solution(sample_cnt, test_size)
        g_solver = GradientSolver(sample_cnt, solution_arr, test_size)
        solution = g_solver.solve(min_cost, max_empty_iterations, max_intensity_iterations, verbose=False)
        solution_array = solution.index

        val_ids = counts.index[solution_array]
        train_ids = counts.index[~solution_array]

        if return_val_ids:
            return val_ids

        _train_df = train_df[train_df["intervention_ids"].isin(train_ids)]
        _train_df = shuffle(_train_df)
        _train_df.reset_index(inplace=True, drop=True)

        _val_df = train_df[train_df["intervention_ids"].isin(val_ids)]
        _val_df = shuffle(_val_df)
        _val_df.reset_index(inplace=True, drop=True)

        x_train = np.array(_train_df["paths"].astype(str))
        x_val = np.array(_val_df["paths"].astype(str))

        y_train = np.array(_train_df["labels"])
        y_val = np.array(_val_df["labels"])

        if verbose:
            print("Length Validation:", len(_val_df))
            print("Length Train:", len(_train_df))
            print(f"Percentage Val Data (Target: {test_size}):", len(_val_df) / (len(_val_df)+len(_train_df)))
            train_counts = _train_df["labels"].value_counts().sort_index()
            val_counts = _val_df["labels"].value_counts().sort_index()

            print("Class counts Validation")
            print(val_counts)
            print("Class counts Training")
            print(train_counts)
            print("Validation Class Percentages of Total")
            print(val_counts / (train_counts + val_counts))

        return x_train, x_val, y_train, y_val
