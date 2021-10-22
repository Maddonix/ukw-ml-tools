from typing import List
from bson.objectid import ObjectId
import torch
from datetime import datetime as dt


def report_to_string(report: dict[str]) -> str:
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


def binary_classification_pred_to_db_value(
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
    bool_pred = pred.copy()
    bool_pred[bool_pred >= 0.5] = 1
    bool_pred[bool_pred < 0.5] = 0

    values = {
        ObjectId(ids[i]): {
            "version": version,
            "targets": targets,
            "value": float(pred[i][0]),
            "creation_date": creation_date,
            "label": bool(bool_pred[i][0]),
        }
        for i in range(len(bool_pred))
    }

    return values
