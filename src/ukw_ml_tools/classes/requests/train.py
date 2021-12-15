from pydantic import BaseModel, validator
from ..config import Configuration, AiLabelConfig
from ..train_data import TrainDataDb, TrainData
from typing import Optional


class TrainRequest(BaseModel):
    cfg: Configuration
    ai_config: AiLabelConfig
    train_data_db: TrainDataDb
    train_data: Optional[TrainData]

    @validator("train_data", always=True)
    def set_train_data(cls, v, values):
        v = values["train_data_db"]
        v = TrainData(**v)
        return v
