from typing import Collection, List
import os
import shutil
from pathlib import Path
import warnings
from bson.objectid import ObjectId
import json
import torch
from datetime import datetime as dt
from typing import Optional

from .crud import get_intervention_for_image_id

from ..utils.export import rename_path_if_exists


def get_ls_task(
    img_path: str,
    img_id: str,
    intervention_id: str,
    targets: List,
    predictions: List,
    origin: str,
    report: Optional[str] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    intervention_date: Optional[str] = None,
    intervention_type: Optional[str] = None,
) -> dict:
    """Function to generate a labelstudio task element

    Args:
        img_path (str): Path to load image in labelstudio. In most cases this should be a URL like 'http://localhost:8000/{...}'
            or an absolute path to the image.
        img_id (str): String of the MongoDb ObjectId of the image to label
        targets (List): List of targets to label
        report (str): String of the report associated with Image, can be empty.
        origin (str): Source of the image.
        age (int): Age of the individual the image was taken from at the time of the intervention
        gender (str): Biological gender of the individual the image was taken from.
        intervention_date (str): Date string in format "%Y-%m-%d" specifying the intervention date.
        intervention_id (str): String of the MongoDb ObjectId of the image's intervention.
        intervention_type (str): Intervention Type, e.g. "Koloskopie".
        predictions (List): List of the image's predictions.

    Returns:
        dict: Returns dictionary of a single labelstudio task with the keys "data"
        (contains: "image", "_id", "report", "origin", "age", "gender", "intervention_date", "intervention_id", "intervention_type",
        "targets) and "predictions"
    """
    _dict = {
        "data": {
            "image": img_path,
            "_id": img_id,
            "report": report,
            "origin": origin,
            "age": age,
            "gender": gender,
            "intervention_date": intervention_date,
            "intervention_id": intervention_id,
            "intervention_type": intervention_type,
            "targets": targets,
        },
        "predictions": predictions,
    }
    delete = []
    for key, value in _dict["data"].items():
        if value is None:
            delete.append(key)
    for key in delete:
        del _dict["data"][key]

    return _dict


def db_to_ls_classification_labels(db_label: dict) -> dict:
    """Function to convert database classification labels to labelstudio labels.

    Args:
        db_label (dict): Dictionary contains labels as keys with associated bool values.

    Returns:
        dict: Returns dictionary with key "choices" and a list of all labels which were evaluated as true.
    """
    ls_label = []
    for target, value in db_label.items():
        if value == 1:
            ls_label.append(target)

    return {"choices": ls_label}


def ls_to_db_labels(ls_label) -> dict:
    """Fuction to convert labelstudio classification annotations to database compatible label dictionaries

    Args:
        ls_label (dict): Expects a labelstudio task element with an annotation

    Returns:
        dict: database label element
    """
    db_label = {}
    db_label["_id"] = ls_label["data"]["_id"]
    db_label["targets"] = ls_label["data"]["targets"]
    db_label["annotations"] = [_["result"] for _ in ls_label["annotations"]]
    print(db_label["annotations"])
    if db_label["annotations"]:
        assert len(db_label["annotations"]) == 1
        db_label["annotations"] = db_label["annotations"][0]
        if db_label["annotations"]:
            assert len(db_label["annotations"]) == 1
            db_label["annotations"] = db_label["annotations"][0]
            db_label["labels"] = {_: False for _ in db_label["targets"]}

            for key, value in db_label["annotations"]["value"].items():
                # Add Transformation for boxes ###################
                if key == "choices":
                    for _label in value:
                        db_label["labels"][_label] = True

            return db_label


def report_to_string(report: dict) -> str:
    """Function to merge the elements of the report field to a single string element.

    Args:
        report (dict): Dict with report categories as keys and lists of strings as values

    Returns:
        str: Concatenated string of the report.
    """
    report_string = []
    for key, value in report.items():
        _report_string = "\n".join(value)
        _report_string = "\n".join([key, _report_string])
        _report_string += "\n"
        report_string.append(_report_string)

    report_string = "\n".join(report_string)
    return report_string


def classification_pred_to_db_value(
    pred: torch.Tensor, ids: List, version: float, targets: List, creation_date=dt.now()
):
    """Function expects tensor of a predicted batch and list of corresponding database ids.
    Appplies sigmoid function and assigns True or False values at a threshold of >= 0.5.

    Args:
        pred (torch.Tensor): Batch of tensor. Number of predictions for each image must match length of targets.
        ids (List): List of ObjectIds matching the images in database. Length must match length of batch.
        version (float): Version of the according prelabel algorithm.
        targets (List): string list of labels which were predicted. Order and length must match predictions of network
        creation_date ([type], optional): Datetime of prediction. Defaults to dt.now().

    Returns:
        [dict]: dictionary with ObjectIds as keys and according db values as labels.
        Should be inserted at "labels.predictions.{prelabel_engine}
    """
    pred = torch.sigmoid(pred).cpu().numpy()
    assert len(pred[0]) is len(targets)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    #     pred = pred.astype(int)

    values = {
        ObjectId(ids[j]): {
            "version": version,
            "targets": targets,
            "creation_date": creation_date,
            "labels": {target: bool(_pred[i]) for i, target in enumerate(targets)},
        }
        for j, _pred in enumerate(pred)
    }

    return values


def get_tasks_from_image_list(
    image_list,
    prelabel_type: str,
    targets: List,
    host_prefix: str,
    export_path: str,
    result_type: str,
    from_name: str,
    to_name: str,
    db_images: Collection,
    db_interventions: Collection,
):
    print(export_path)
    if export_path.exists():
        warnings.warn(
            f"{export_path} already existed, deleted folder and all contents before proceeding"
        )
        shutil.rmtree(export_path)
    os.mkdir(export_path)
    images_path = export_path.joinpath("images")
    tasks_path = export_path.joinpath("tasks")
    os.mkdir(images_path)
    os.mkdir(tasks_path)

    tasks = []
    for image in image_list:
        intervention = get_intervention_for_image_id(
            image["_id"], db_images, db_interventions
        )

        origin = intervention["origin"]
        intervention_id = intervention["_id"]
        # Add optionally available metadata
        if "age" in intervention:
            age = intervention["age"]
        else:
            age = None

        if "gender" in intervention:
            gender = intervention["gender"]
        else:
            gender = None

        if "intervention_type" in intervention:
            intervention_type = intervention["intervention_type"]
        else:
            intervention_type = None

        if "report" in intervention:
            report = report_to_string(intervention["report"])
        else:
            report = None

        if "initervention_date" in intervention:
            intervention_date = intervention["intervention_date"].strftime("%Y-%m-%d")
        else:
            intervention_date = None

        img_name = Path(image["path"]).name
        _img_export_path = images_path.joinpath(img_name)
        _img_export_path = rename_path_if_exists(_img_export_path)

        shutil.copyfile(image["path"], _img_export_path.as_posix())

        img_path = host_prefix + img_name

        predictions = [
            {
                "result": [
                    {
                        "type": result_type,
                        "from_name": from_name,
                        "to_name": to_name,
                        "value": db_to_ls_classification_labels(
                            image["labels"]["predictions"][prelabel_type]["labels"]
                        ),
                    }
                ]
            }
        ]

        _task = get_ls_task(
            img_path=img_path,
            img_id=str(image["_id"]),
            targets=targets,
            report=report,
            origin=origin,
            age=age,
            gender=gender,
            intervention_date=intervention_date,
            intervention_id=str(intervention_id),
            intervention_type=intervention_type,
            predictions=predictions,
        )

        tasks.append(_task)

    # Export Tasks
    json_export_path = export_path.joinpath("tasks/tasks.json")
    json_export_path = rename_path_if_exists(json_export_path)

    with open(json_export_path, "w") as f:
        json.dump(tasks, f)

    return tasks


def get_tasks_from_intervention(
    intervention,
    prelabel_type: str,
    host_prefix: str,
    export_path: Path,
    result_type: str,
    from_name: str,
    to_name: str,
):

    # assert export_path.exists() == False
    if export_path.exists():
        warnings.warn(
            f"{export_path} already existed, deleted folder and all contents before proceeding"
        )
        shutil.rmtree(export_path)
    os.mkdir(export_path)
    images_path = export_path.joinpath("images")
    tasks_path = export_path.joinpath("tasks")
    os.mkdir(images_path)
    os.mkdir(tasks_path)

    tasks = []

    intervention_id = intervention["_id"]
    age = intervention["age"]
    origin = intervention["origin"]
    intervention_date = intervention["intervention_date"]
    gender = intervention["gender"]
    intervention_type = intervention["intervention_type"]
    report = report_to_string(intervention["report"])

    for _image in intervention["image_objects"]:
        if _image["in_progress"]:
            warnings.warn(
                f"Image with ID {_image['_id']} ist already queued, skipping."
            )
            continue
        else:
            # Export Image
            img_name = Path(_image["path"]).name
            _img_export_path = images_path.joinpath(img_name)
            _img_export_path = rename_path_if_exists(_img_export_path)

            shutil.copyfile(_image["path"], _img_export_path.as_posix())

            img_path = host_prefix + img_name

            predictions = [
                {
                    "result": [
                        {
                            "type": result_type,
                            "from_name": from_name,
                            "to_name": to_name,
                            "value": db_to_ls_classification_labels(
                                _image["labels"]["predictions"][prelabel_type]["labels"]
                            ),
                        }
                    ]
                }
            ]

            _task = get_ls_task(
                img_path=img_path,
                report=report,
                origin=origin,
                age=age,
                gender=gender,
                intervention_date=intervention_date,
                intervention_id=intervention_id,
                intervention_type=intervention_type,
                predictions=predictions,
            )

            tasks.append(_task)

    # Export Tasks
    json_export_path = export_path.joinpath("tasks/tasks.json")
    json_export_path = rename_path_if_exists(json_export_path)

    with open(json_export_path, "w") as f:
        json.dump(tasks, f)

    return tasks
