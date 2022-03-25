from sklearn.metrics import precision_recall_fscore_support as prfs
from .utils import split_result_df
import pandas as pd
from ..labels.conversions import get_consecutive_ranges

LABELS = [
  "anal_channel",
  "appendix",
  "blood",
  "bubbles",
  "cleaning",
  "clip",
  "diverticule",
  "endocuff",
  "endoscope_reflection",
  "endoscope_retroflection",
  "grasper",
  "ileocaecalvalve",
  "ileum",
  "inflammation",
  "low_quality",
  "nbi",
  "needle",
  "outside",
  "petechia",
  "polyp",
  "scar",
  "snare",
  "stool",
  "water_jet",
  "wound"
]

TARGET_LABELS = [
    "appendix",
    "ileum",
    "ileocaecalvalve",
    "polyp",
    "needle",
    "snare",
    "grasper",
    "clip",
    # "blood",
    "nbi",
    # "cleaning",
    "diverticule",
    "outside",
    "water_jet",
    "low_quality"
]

TARGET_LABELS = [_ for _ in LABELS if _ in TARGET_LABELS]

def get_withdrawal_segments(matrix):
    select_anatomical = matrix.caecum == True
    first_anatomical = matrix.loc[select_anatomical].frame_number.min()
    last_anatomical = matrix.loc[select_anatomical].frame_number.max()
    select_outside = matrix.outside == True
    first_half = matrix.frame_number < len(matrix)/2
    start = matrix.loc[select_outside & first_half].frame_number

    if len(start) == 0:
        start = 0
    else: 
        start = start.max()

    end = matrix.loc[select_outside & ~first_half].frame_number
    if len(end) == 0:
        end = len(matrix)
    else:
        end = end.min()
    try:
        segments = {
            "start": int(start),
            "first_anatomical": int(first_anatomical),
            "last_anatomical": int(last_anatomical),
            "end": int(end)
        }
    except:
        print("Failed to calculate withdrawaltime segments: get_withdrawal_segments()")
        segments = {
            "start": start,
            "first_anatomical": int(len(matrix)/2),
            "last_anatomical": int(len(matrix)/2)+1,
            "end": end
        }

    return segments

def get_deductible_frames(matrix, labels):
    labels = [_ for _ in labels if _ in matrix.columns]
    frames = matrix.loc[matrix.loc[:, labels].any(axis = 1), "frame_number"]
    return frames

def calculate_times(matrix, fps):
    matrix["frame_number"] = matrix.index
    
    anatomical_segments = get_withdrawal_segments(matrix)

    start = anatomical_segments["start"]
    first_anatomical = anatomical_segments["first_anatomical"]
    last_anatomical = anatomical_segments["last_anatomical"]
    end = anatomical_segments["end"]

    _resection_frames = get_deductible_frames(matrix, ["resection", "tool", "polyp", "nbi"])
    # _resection_flanks = get_consecutive_ranges(_resection_frames)
    

    select_resection = matrix.frame_number.isin(_resection_frames)
    select_in_withdrawal_1 = matrix.frame_number > last_anatomical
    select_in_withdrawal_2 = matrix.frame_number < end

    resection_frames = matrix.loc[select_resection & select_in_withdrawal_1 & select_in_withdrawal_2]

    # segmentation = pd.DataFrame(index = matrix.frame_number.copy())
    records = []
    records.extend([
        {"segment": "outside", "frame_number": i} for i in range(0,start)
    ])
    records.extend([
        {"segment": "insertion", "frame_number": i} for i in range(start,first_anatomical)
    ])
    records.extend([
        {"segment": "withdrawal", "frame_number": i} for i in range(last_anatomical,end)
    ])
    records.extend([
        {"segment": "caecum", "frame_number": i} for i in range(first_anatomical,last_anatomical)
    ])
    records.extend([
        {"segment": "resection", "frame_number": i} for i in _resection_frames
    ])
    segmentation = pd.DataFrame.from_records(records)

    intervention_time = (end - start) / fps / 60
    insertion_time = (first_anatomical - start) / fps / 60
    caecum_inspection_time = (last_anatomical - first_anatomical) / fps / 60
    withdrawal_time = (end - last_anatomical) / fps / 60
    resection_time = len(resection_frames) / fps / 60

    corrected_withdrawal_time = withdrawal_time - resection_time

    result_dict = {
        "examination_time_min": intervention_time,
        "insertion_time_min": insertion_time,
        "caecum_inspection_time_min": caecum_inspection_time,
        "withdrawal_time_min": withdrawal_time,
        "resection_time_min": resection_time,
        "corrected_withdrawal_time_min": corrected_withdrawal_time
    }
    return result_dict, segmentation

def calculate_metrics(df_dict):
    y_pred = df_dict["predictions"]
    y_true = df_dict["annotations"]
    y_pred_smooth = df_dict["predictions_smooth"]
    
    metrics = prfs_matrix_to_df(prfs(y_true, y_pred), y_true.columns)
    metrics_smooth = prfs_matrix_to_df(prfs(y_true, y_pred_smooth), y_true.columns)

    return metrics,metrics_smooth

def prfs_matrix_to_df(matrix, cols):
    metrics = pd.DataFrame(matrix)
    metrics.columns=cols
    metrics.index = ["presicion", "recall", "f1", "support"]
    metrics = metrics.T

    return metrics