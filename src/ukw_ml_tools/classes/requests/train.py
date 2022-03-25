from pydantic import BaseModel, validator
from ..config import Configuration, AiLabelConfig
from ..train_data import TrainDataDb, TrainData
from typing import Optional
from pathlib import Path


class TrainRequest(BaseModel):
    name: Path
    base_path_models: Path
    train_data_db: TrainDataDb

    def get_train_data(self):
        v = self.train_data_db
        return TrainData(**v.to_dict_intern())


class ContinueTrainRequest(BaseModel):
    name: str
    path_checkpoint: Path
    train_data_db: TrainDataDb

    def get_train_data(self):
        v = self.train_data_db
        return TrainData(**v.to_dict_intern())
