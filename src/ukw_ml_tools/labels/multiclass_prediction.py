from torch import Tensor
from torch.nn import LogSoftmax
import torch
from datetime import datetime as dt
from ..classes.prediction import MultilabelPrediction
from typing import List

SOFTMAX = LogSoftmax(dim=-1)


def mc_base_pred_dict(ai_config, timestamp=dt.now()):
    base_prediction_dict = {
        "name": ai_config.name,
        "version": ai_config.ai_settings.version,
        "date": timestamp,
        "value": None,
        "raw": None,
        "choices": ai_config.choices,
    }
    return base_prediction_dict


def mc_batch_result_raw_to_pred_values(preds: Tensor, softmax=None):
    if not softmax:
        softmax = SOFTMAX
    raw = preds
    values = softmax(preds)
    values = torch.argmax(values, dim=-1)

    if preds.is_cuda:
        values = values.cpu().numpy()
        raw = raw.cpu().numpy().tolist()
    else:
        values = values.numpy()
        raw = raw.numpy().tolist()

    return values, raw


def mc_prediction_to_object(value: int, raw_pred: List[int], base_pred_dict):
    prediction_dict = base_pred_dict.copy()
    prediction_dict["value"] = value
    prediction_dict["raw"] = raw_pred
    prediction = MultilabelPrediction(**prediction_dict)
    return prediction
