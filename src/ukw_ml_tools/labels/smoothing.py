import numpy as np
import pandas as pd
from ..mongodb.results import get_intervention_labels
from ..classes.prediction import BinaryPrediction
from datetime import datetime as dt


def running_mean(preds, conv_len, weight_main=None, threshold=0.5, future_frames=True):
    if not conv_len % 2:
        conv_len += 1

    if not weight_main:
        weight_main = 1/conv_len

    weight_main = conv_len * weight_main

    if future_frames:
        conv = np.ones(int(conv_len))

    else:
        conv = np.zeros(int(conv_len-1))
        conv = np.append(conv, np.ones(conv_len))

    mid = int(conv_len/2)+1
    conv[mid] = weight_main
    conv = conv/sum(conv)

    smooth_prediction = np.convolve(preds, conv, mode="same")
    return smooth_prediction


def smooth_binary_labels(preds, conv_len, weight_main=None, threshold=0.5, future_frames=None):
    """
    Weight main of 0.5 would mean that the "current" frame is weighted 
    """
    smooth_prediction = running_mean(preds, conv_len, weight_main=None, threshold=0.5, future_frames=future_frames)
    smooth_prediction[smooth_prediction > threshold] = True
    smooth_prediction[smooth_prediction <= threshold] = False
    return smooth_prediction


def calculate_intervention_smooth_labels(
    name,
    conv_len,
    intervention_id,
    version,
    db_images,
    future_frames=None,
    main_weight=None,
    threshold=0.5,
    timestamp=None
):
    if not timestamp:
        timestamp = dt.now()
    _records = get_intervention_labels(intervention_id, db_images, "predictions")
    pred_df = pd.DataFrame.from_records(_records)
    print(pred_df.head())
    select_df = pred_df[pred_df.name == name]
    select_df = select_df.sort_values("frame_number")
    preds = select_df.value.to_numpy()
    print(preds)
    preds = smooth_binary_labels(preds, conv_len, main_weight, threshold, future_frames)
    print(preds)
    for i, img_id in enumerate(select_df._id):
        value = preds[i]
        _pred = BinaryPrediction(**{
            "name": name,
            "version": version,
            "date": timestamp,
            "value": value,
            "raw": value,
            "choices": [False, True]}
        )
        db_images.update_one({"_id": img_id}, {"$set": {f"predictions_smooth.{name}": _pred.dict()}})
