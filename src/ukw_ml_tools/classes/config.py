from pydantic import BaseModel, validator, constr
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
from pymongo.collection import Collection
from pymongo import MongoClient
from .train_data import TrainDataSamplerSettings
from .annotation import BinaryAnnotation, MultilabelAnnotation, MultichoiceAnnotation
from .prediction import BinaryPrediction, MultilabelPrediction, MultichoicePrediction

LOOKUP_LABEL_CLASS = {
    "annotation": {
        "binary": BinaryAnnotation,
        "multilabel": MultilabelAnnotation,
        "multiclass": MultichoiceAnnotation,
    },
    "prediction": {
        "binary": BinaryPrediction,
        "multilabel": MultilabelPrediction,
        "multiclass": MultichoicePrediction,
    },
}

# LOOKUP_MODEL_CLASS =


class AiSettings(BaseModel):
    version: float
    image_scaling: int
    base_model_name: str

    def __hash__(self):
        return hash(repr(self))


class AiLabelConfig(BaseModel):
    name: str
    label_type: str
    is_compound: bool
    annotation_class: Optional[
        Union[BinaryAnnotation, MultilabelAnnotation, MultichoiceAnnotation]
    ]
    prediction_class: Optional[
        Union[BinaryPrediction, MultilabelPrediction, MultichoicePrediction]
    ]
    label_type: str
    choices: List[Union[bool, str]]
    ai_settings: AiSettings
    sampler_settings: TrainDataSamplerSettings
    paths: Dict[int, Path]

    class Config:
        json_encoders = {Path: str}

    @validator("annotation_class", always=True)
    def set_annotation_class(cls, v, values):
        v = LOOKUP_LABEL_CLASS["annotation"][values["label_type"]]
        return v

    @validator("prediction_class", always=True)
    def set_prediction_class(cls, v, values):
        v = LOOKUP_LABEL_CLASS["annotation"][values["label_type"]]
        return v

    def get_latest_model_path(self):
        _ = self.paths
        _ = list(_.keys())
        if _:
            _.sort(reverse = True)
            path = self.paths[_[0]]
            
            return path

        print("no models found")
        return None

    def __hash__(self):
        return hash(repr(self))

    # def get_primary_choices(self):
    #     """Returns choices without the labels "outside" and "blurry" """


class LabelSettings(BaseModel):
    default_values: Dict[int, Any]
    skip_labels_flank_to_image: List[str]

    def __hash__(self):
        return hash(repr(self))


class BasePaths(BaseModel):
    frames: Path
    models: Path
    train_data: Path
    terminology: Path

    class Config:
        allow_population_by_field_name = True
        json_encoders = {Path: str}
        schema_extra = {"example": {}}

    def __hash__(self):
        return hash(repr(self))


class TimeFormat(BaseModel):
    video_extern: str

    def __hash__(self):
        return hash(repr(self))


class Auth(BaseModel):
    labelstudio_token: str
    extern_user: str
    extern_pw: str
    extern_auth: Optional[Tuple[str]]

    @validator("extern_auth", always=True)
    def add_auth(cls, v, values):
        v = (values["extern_user"], values["extern_pw"])
        return v

    def __hash__(self):
        return hash(repr(self))

class WebSettings(BaseModel):
    ip_gpu_server: str
    ip_endobox_extreme: str
    port_extern: str
    url_labelstudio: str
    port_fileserver: str
    url_mongodb: str
    url_extern: Optional[str]

    def __hash__(self):
        return hash(repr(self))

    @validator("url_extern", always=True)
    def add_url_extern(cls, v, values):
        v = "https://"
        v += values["ip_endobox_extreme"]
        v += ":"
        v += values["port_extern"]
        v += "/data"
        return v



class Databases(BaseModel):
    url_mongodb: str

    mongo_client: Optional[Any]

    @validator("mongo_client", pre=True, always=True)
    def get_client(cls, v, values):
        v = MongoClient(values["url_mongodb"], connectTimeoutMS=200, retryWrites=True)
        return v

    def __hash__(self):
        return hash(repr(self))
   

class Configuration(BaseModel):
    label_settings: LabelSettings
    base_paths: BasePaths
    auth: Auth
    web: WebSettings

    db: Optional[Databases]

    def __hash__(self):
        return hash(repr(self))

    @validator("db", pre=True, always=True)
    def setup_db(cls, v, values):
        v = Databases(url_mongodb=values["web"].url_mongodb)
        return v

    def get_databases(self):
        """
        Returns (db_images, db_interventions, db_test_data, db_train_data)
        """
        mongo_client = self.db.mongo_client
        db_interventions = self.db.mongo_client
        db_interventions = mongo_client.EndoData2.Interventions
        db_images = mongo_client.EndoData2.Images
        db_test_data = mongo_client.EndoData2.TestData
        db_train_data = mongo_client.EndoData2.TrainData

        return db_images, db_interventions, db_test_data, db_train_data

    def get_test_databases(self):
        """
        Returns collections from TestEndoData(db_images, db_interventions, db_test_data, db_train_data)
        """
        mongo_client = self.db.mongo_client
        db_interventions = self.db.mongo_client
        db_interventions = mongo_client.TestEndoData.Interventions
        db_images = mongo_client.TestEndoData.Images
        db_test_data = mongo_client.TestEndoData.TestData
        db_train_data = mongo_client.TestEndoData.TrainData

        return db_images, db_interventions, db_test_data, db_train_data

    def get_extern_tuple(self):
        """Returns (url, auth)"""
        url = self.web.url_extern
        auth = self.auth.extern_auth
        return url, auth
