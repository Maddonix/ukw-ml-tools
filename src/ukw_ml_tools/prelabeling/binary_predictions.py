from pathlib import Path
from pymongo.collection import Collection
import pandas as pd
from ..db.crud import get_images_to_prelabel_query
import warnings
import torch
from typing import List
from torch.utils.data import Dataset
from ..models.pl_ileum_detection_resnet import IleumDetectionResnet
from ..models.pl_tool_detection_resnet import ToolDetectionResnet
from ..datasets.binary_image_classification_ds import BinaryImageClassificationDS

AVAILABLE_AI_MODELS = ["polyp", "ileum", "tool", "ileocaecalvalve", "appendix"]


def get_ckpt_path(ai_name: str, version: float, path_models: Path) -> Path:
    """Function selects folder which matches the label in the given folder and then returns a Path object of the corresponding model version

    Args:
        ai_name (str): name of the ai
        version (float): version number (one decimal!)
        path_models (Path): Path object pointing to the base model folder

    Returns:
        Path: Path object pointing to the desired model
    """

    version = "{:.1f}".format(version)
    _path = path_models.joinpath(f"{ai_name}/{version}.ckpt")
    if not _path.exists():
        warnings.warn(f"Path {_path} does not exist!")
        print(_path)

    return _path


def get_prediction_task_df(label: str, version: float, db_images: Collection, origin: str = None, limit: int = 1000,
    predict_annotated: bool = False) -> pd.DataFrame:
    """Function queries the given collection for images to predict the given label on.
    Images which already have an annotation of this label or a prediction with the current
    AI version are excluded. If given a origin, only images of the given origin are predicted.
    Returns a pd.DataFrame with columns "_id" (image id in database) and "file_path" (complete filepath on our server)

    Args:
        label (str): label to query for
        version (float): AI version number
        db_images (Collection): collection containing images
        origin (str, optional): Optionally filter for a specific origin. Defaults to None.
        limit (int, optional): Number to limit the output. Defaults to 1000.

    Returns:
        pd.DataFrame: Dataframe with columns "_id" and "file_path"
    """
    if origin:
        agg = [{
            "$match": {"origin": origin}
        }]
    else:
        agg = []

    agg.extend(get_images_to_prelabel_query(label=label, version=version, limit=limit, predict_annotated=predict_annotated))
    result = db_images.aggregate(agg)

    df_dict = {"file_path": [], "_id": []}
    for _ in result: 
        df_dict["_id"].append(_["_id"])
        df_dict["file_path"].append(_["path"])

    return pd.DataFrame().from_dict(df_dict)


def load_model(model_name: str, version: float, eval: bool, base_path_models: Path):
    """Function loads model for given name.

    Args:
        model_name (str): One of "polyp", "ileum", "tool", "ileocaecalvalve", "appendix"
        version (float): version number.
        eval (bool): If True, model is loaded in eval mode.
        base_path_models(Path): Points to directory containing folders named by models, containing checkpoints named by 
            version numbers (e.g. '0.1.ckpt')

    Returns:
        [PL Model]: Model for given name.
    """
    assert model_name in AVAILABLE_AI_MODELS

    ckpt_path = get_ckpt_path(model_name, version, base_path_models)
    if model_name == "polyp":
        warnings.warn("Polyps are not yet supported")
        return None
    if model_name == "ileum":
        Model = IleumDetectionResnet
    if model_name == "tool":
        Model = ToolDetectionResnet
    if model_name == "ileocaecalvalve":
        Model = IleumDetectionResnet
    if model_name == "appendix":
        Model = IleumDetectionResnet

    trained_model = Model.load_from_checkpoint(checkpoint_path=ckpt_path)

    if torch.cuda.is_available():
        trained_model.cuda(0)
    # trained_model.to(0)
    if eval:
        trained_model.train(False)
        trained_model.freeze()

    return trained_model


def load_binary_image_classification_dataset(paths: List, ids: List, scaling: int, training: bool) -> Dataset:
    dataset = BinaryImageClassificationDS(paths, ids, scaling=scaling, training=training)

    return dataset
