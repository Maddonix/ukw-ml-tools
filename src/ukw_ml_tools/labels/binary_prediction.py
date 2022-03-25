from datetime import datetime as dt
import torch
from typing import List
from ..classes.prediction import BinaryPrediction

def b_base_pred_dict(ai_config, timestamp = dt.now()):
    base_prediction_dict = {
        "name": ai_config.name,
        "version": ai_config.ai_settings.version,
        "date": timestamp,
        "value": None,
        "raw": None,
        "choices": ai_config.choices,
    }
    return base_prediction_dict

def b_raw_to_value(raw: torch.Tensor):
    raw[raw>0.5] = True
    raw[raw<=0.5] = False
    return raw.bool()

def b_batch_result_raw_to_pred_values(preds: torch.Tensor):
    raw = torch.squeeze(preds, -1)
    raw = torch.sigmoid(raw)
    values = b_raw_to_value(raw.detach().clone())

    if preds.is_cuda:
        values = values.cpu().numpy()
        raw = raw.cpu().numpy().tolist()
    else:
        values = values.numpy()
        raw = raw.numpy().tolist()

    return values, raw

def b_prediction_to_object(value: bool, raw_pred: float, base_pred_dict):
    prediction_dict = base_pred_dict.copy()
    prediction_dict["value"] = value
    prediction_dict["raw"] = raw_pred
    prediction = BinaryPrediction(**prediction_dict)
    return prediction