import pandas as pd
from ..mongodb.results import get_intervention_labels
from ..labels.conversions import get_consecutive_ranges
from ..labels.smoothing import smooth_binary_labels
from datetime import datetime as dt
from ..classes.prediction import BinaryPrediction

def pred_update_to_object(prediction, version = -1, threshold = 0.5):
    prediction = parse_prediction(prediction)
    update = {}
    for pred in prediction:
        name = pred[0]
        raw = pred[1]
        prediction_dict =  {
        "version": version,
        "date": dt.now(),
        "choices": [False, True]
    }
        prediction_dict.update({
            "value": raw > threshold,
            "raw": raw,
            "name": name
        })
        update[name] = BinaryPrediction(**prediction_dict).dict()

    return update

def filter_exclusive_labels(prediction, label_list):
    _caecum_predictions = [_ for _ in prediction if _[0] in label_list]
    _p = 0
    _label = None
    for _ in _caecum_predictions:
        if _[1]>_p:
            _p = _[1]
        _label = __[0]
    prediction = [_ for _ in prediction if _[0] not in label_list]
    if _label:
        prediction.append(_label)

    return prediction

def parse_prediction(
        prediction,
        threshold = 0.5,
        # caecum_labels = ["ileocaecalvalve", "appendix", "ileum"],
        # tool_labels = ["grasper", "snare", "needle"]
    ):
    
    prediction = [_ for _ in prediction if _[1] > threshold]
    # prediction = filter_exclusive_labels(prediction, caecum_labels)
    # prediction = filter_exclusive_labels(prediction, tool_labels)

    return prediction

def get_result_df(intervention_id, db_images, frame_list=None):
    label_types = [_ for _ in ["annotations" , "predictions", "predictions_smooth"]]#[:n_type]]
    df = pd.DataFrame()
    for label_type in label_types:
        _records = get_intervention_labels(intervention_id, db_images, label_type)
        if _records:
            _df = pd.DataFrame.from_records(_records)
            if label_type == "predictions":
                _df = _df.loc[:,["video_key", "frame_number", "_type", "name", "value", "raw"]]
            else:
                _df = _df.loc[:,["video_key", "frame_number", "_type", "name", "value"]]

            _df = _df.sort_values("frame_number")
            if frame_list:
                select = _df.frame_number.isin(frame_list)
                _df = _df[select]
            df = df.append(_df)

    df = df.loc[df.value==True]
    return df

def set_summary_label(result_matrix, name, include):
    include = [_ for _ in include if _ in result_matrix.columns]
    result_matrix[name] = result_matrix.loc[:, include].any(axis = 1)
    return result_matrix
    

def filter_ranges_by_length(ranges, len_frames):
    select = []
    for i, _range in enumerate(ranges):
        if (_range[1]- _range[0]) > len_frames:
            select.append(i)

    ranges = [ranges[i] for i in select]
    return ranges

def merge_close_ranges(ranges, frame_diff):
    ranges_merged = []
    last = None

    for i, _range in enumerate(ranges):
        if i == 0:
            last = _range
            continue
        
        _last = last[1]
        _current = _range[0]
        delta = _current - _last

        if delta > frame_diff:
            ranges_merged.append(last)
            last = _range
        
        else:
            last = (last[0], _range[1])

    if last:
        ranges_merged.append(last)

    return ranges_merged

def range_list_to_frame_list(ranges):
    frames = []
    for _range in ranges:
        frames.extend([_ for _ in range(_range[0], _range[1]+1)])

    return frames

def post_process_label(df, fps, min_s, merge_diff_s):
    df = df.copy()
    _frames = df.loc[df == True].index.tolist()

    _frames.sort()
    _ranges = get_consecutive_ranges(_frames)
    _ranges = filter_ranges_by_length(_ranges, fps*min_s)
    _ranges = merge_close_ranges(_ranges, fps*merge_diff_s)
    _frames = range_list_to_frame_list(_ranges)

    df.loc[df.index.isin(_frames)] = True
    df.loc[~df.index.isin(_frames)] = False

    return df


def split_result_df(result_df, fps = 50, threshold_tool = 0.6, threshold_caecum = 0.8):
    evaluate = ["appendix", "ileocaecalvalve", "ileum", "caecum", "tool"]
    _evaluate = evaluate + ["outside"]
    df_dict = {}
    _types = result_df._type.unique()
    _types.sort()
    for _type in _types:
        _df = result_df.loc[result_df._type == _type]
        _df = _df.pivot(index="frame_number", columns="name", values="value")

        if _type == "predictions":
            __df = pd.DataFrame(index=df_dict["annotations"].index.copy())
            _df = __df.merge(_df, how="outer", left_index=True, right_index =True)

        # _df = set_summary_label(_df, name="caecum", include = ["ileum", "caecum"])
        # if _type != "annotations":
        _df = set_summary_label(_df, name="caecum", include = ["appendix", "ileum", "caecum", "ileocaecalvalve"])
        _df = set_summary_label(_df, name="tool", include = ["tool", "clip", "grasper", "snare", "needle", "nbi"])
        
        if "outside" not in _df.columns:
            _df["outside"] = False
        _df = _df.sort_values("frame_number")

        if "body" in _df.columns:
            _df["outside"] = ~(_df.body.astype("bool")==True)

        for label in evaluate:
            if not label in _df.columns:
                _df[label] = False
        _df = _df.loc[:, _evaluate]
        _df =_df.reindex(sorted(_df.columns), axis = 1)
        _df = _df.fillna(False)
        df_dict[_type]=_df

        if _type == "predictions_smooth":
            processed_df = _df.copy()
            if "low_quality" in _df.columns:
                processed_df.loc[_df.low_quality==True, evaluate] = False
            processed_df.loc[_df.outside==True, evaluate] = False
            df_dict["y_pred_post_processing"] = processed_df

        
    y_pred = df_dict["y_pred_post_processing"]
    y_pred_post_processing = pd.DataFrame()
    for col in y_pred.columns:
        threshold = threshold_caecum
        if col == "tool":
            threshold = threshold_tool
        y_pred_post_processing[col] = smooth_binary_labels(y_pred[col], fps, threshold = threshold, future_frames=True)
        y_pred_post_processing[col] = y_pred_post_processing[col].astype(bool)
        if col == "caecum":
            y_pred_post_processing.loc[:, col] = post_process_label(y_pred_post_processing[col], fps, 2, 10)
        if col == "tool":
            y_pred_post_processing.loc[:, col] = post_process_label(y_pred_post_processing[col], fps, 2, 10)
    y_pred_post_processing.index = y_pred.index
    y_pred_post_processing.columns = y_pred.columns
    df_dict["y_pred_post_processing"] = y_pred_post_processing

    return df_dict

