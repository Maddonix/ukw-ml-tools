import warnings
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Any, List
from bson.objectid import ObjectId
from pymongo.collection import Collection
from .fieldnames import *
import os
from datetime import datetime as dt
import numpy as np
import re
import subprocess
import json
import cv2
from operator import itemgetter

def generate_default_dict():
    return defaultdict(dict)

def generate_list_default_dict():
    return defaultdict(list)

def generate_frame_path(video_key:str, base_path_frames: Path) -> Path:
    return base_path_frames.joinpath(video_key)

# Crud
def match_logical_and_aggregation(condition_list: List[str]):
    agg_element = {
        "$match": {
            "$and": [
                _ for _ in condition_list
            ]
        }
    }

    return agg_element

def logical_or_aggregation(condition_list):
    agg_element = {
        "$or": [_ for _ in condition_list]
    }

    return agg_element

def field_exists_query(fieldname: str, exists: bool = True) -> dict:
    return {
        fieldname: {"$exists": exists}
    }

def field_value_query(fieldname: str, value: Any) -> dict:
    return {
        fieldname: value
        }

def fieldvalue_in_list_query(fieldname: str, values: List[Any]) -> dict:
    return {
        fieldname: {"$in": values}
    }

def fieldvalue_nin_list_query(fieldname: str, values: List[Any]) -> dict:
    return {
        fieldname: {"$nin": values}
    }

def get_intervention_db_template(**kwargs):
    
    template = {
        FIELDNAME_EXTERNAL_ID: None,
        FIELDNAME_FRAMES: {},
        FIELDNAME_FREEZES: {},
        FIELDNAME_VIDEO_KEY: None,
        FIELDNAME_VIDEO_PATH: None,
        FIELDNAME_INTERVENTION_TYPE: None,
        FIELDNAME_TOKENS: {
            FIELDNAME_TOKENS_REPORT: [],
            FIELDNAME_TOKENS_PATHO: []
        }
    }

    # if "intervention_dict" in kwargs:
    #     for key, value in kwargs["intervention_dict"]:
    #         if key in template:
    #             template[key]=value

    return template


def get_image_db_template(
    origin: str = None,
    intervention_id: ObjectId = None,
    path: str = None,
    n_frame: int = None,
    image_type: str = None
):
    """
    """

    template = {
        FIELDNAME_ORIGIN: origin,
        FIELDNAME_INTERVENTION_ID: intervention_id,
        FIELDNAME_IMAGE_PATH: path,
        FIELDNAME_FRAME_NUMBER: n_frame,
        FIELDNAME_IMAGE_TYPE: image_type,
        FIELDNAME_LABELS: {},
        FIELDNAME_PREDICTIONS: {},

    }
    return template


def delete_frame_from_intervention(
    frame_dict: dict,
    db_interventions: Collection
    ):
    n_frame = frame_dict[FIELDNAME_FRAME_NUMBER]
    frame_id = frame_dict["_id"]
    intervention_id = frame_dict[FIELDNAME_INTERVENTION_ID]
    frame_path = Path(frame_dict[FIELDNAME_IMAGE_PATH])
    assert frame_path.exists()
    

    intervention = db_interventions.find_one({"_id": intervention_id})
    assert intervention
    assert "frames" in intervention
    assert str(n_frame) in intervention[FIELDNAME_FRAMES]
    assert frame_id is intervention[FIELDNAME_FRAMES]
    db_interventions.update_one(
        {"_id": intervention_id},
        {"$unset": {f"{FIELDNAME_FRAMES}.{n_frame}": ""}}
        )

    os.remove(frame_path)

    return True


def datetime_from_video_key(video_key, re_pattern, dt_pattern):
    date_string = re.search(re_pattern, video_key)
    if date_string:
        date_string = date_string.group()
        intervention_date = dt.strptime(date_string, dt_pattern)

        return intervention_date

def get_video_meta(path: Path) -> dict:
    command = f"ffprobe -hide_banner -loglevel fatal \
        -show_error -show_format -show_streams -show_chapters \
            -show_private_data -print_format json '{path.as_posix()}'"

    _ = subprocess.run(command, capture_output=True, shell = True)

    meta = json.loads(_.stdout)
    # assert len(meta["streams"]) == 1
    try:
        stream = meta["streams"][0]
        fps_strings = stream["r_frame_rate"].split("/")
        fps = float(fps_strings[0])/float(fps_strings[1])
        duration = float(meta["format"]["duration"])
        frames_total = fps*duration
    
    except:
        warnings.warn(f"Could not Generate Metadata for {path}, using cv2")
        cap = cv2.VideoCapture(path.as_posix())
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames_total = 0
        success, frame = cap.read()

        while success:
            frames_total +=1
            success, frame = cap.read()

        duration = frames_total / fps

    meta = {
        FIELDNAME_FPS: fps,
        FIELDNAME_FRAMES_TOTAL: frames_total,
        FIELDNAME_VIDEO_DURATION: duration
    }

    return meta

def get_frames_total(cap: cv2.VideoCapture) -> int:
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-2
    assert frames_total
    return frames_total


def get_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps
    return fps

def prediction_df_long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    return df_long.pivot_table(
        index=[FIELDNAME_FRAME_NUMBER, "_id"],
        columns=[COLNAME_AI_NAME],
        values=[FIELDNAME_PREDICTION_LABEL, FIELDNAME_PREDICTION_VALUE]
        )

def parse_stats_dict_name(name, value):
    keys = name.split(".")
    record = {}
    record[DF_COL_ENTITY] = keys[0]
    record[DF_COL_VALUE_TYPE] = keys[1]
    record[DF_COL_DATE] = dt.now()
    record[DF_COL_VALUE] = value
    
    new_attribute = []
    for i, key in enumerate(keys[2:]):
        new_attribute.append(key)
        if len(new_attribute) == 2:
            record.update({
                new_attribute[0]: new_attribute[1]
            })
            new_attribute = []
    return record

def get_count(db_images, entity, match_conditions_dict) -> dict:
    count = {}
    prefixes = [entity, PREFIX_COUNT]
    match_conditions = {}

    for prefix, match_condition in match_conditions_dict.items():
        prefixes.append(prefix)
        match_conditions.update(match_condition)

    prefix = ".".join(prefixes)
    count[prefix] = db_images.count_documents(match_conditions)
    return count


def get_grouped_count(db_images, entity, fieldname, additional_match_conditions: dict = None) -> dict:
    match_conditions = {fieldname: {"$nin": ["", None, [], {}]}}
    prefixes = [entity, PREFIX_COUNT]

    if additional_match_conditions:
        for prefix, _match_conditions in additional_match_conditions.items():
            prefixes.append(prefix)
            match_conditions.update(_match_conditions)

    _grouped_count = db_images.aggregate([
        {
            "$match": match_conditions
        },
        {
            "$group": {
                "_id": "$"+fieldname,
                PREFIX_COUNT: {"$sum": 1} 
            }
        }
    ])

    prefix = ".".join(prefixes)

    grouped_count = {
        f"{prefix}.{fieldname}.{_['_id']}": _[PREFIX_COUNT] for _ in _grouped_count
    }

    return grouped_count


def get_grouped_dict_count(db_images, entity, fieldname, additional_match_conditions: dict = None, additional_group_conditions: dict = None):
    match_conditions = {fieldname: {"$nin": ["", None, [], {}]}}
    prefixes = [entity, PREFIX_COUNT]

    if additional_match_conditions:
            for prefix, _match_conditions in additional_match_conditions.items():
                prefixes.append(prefix)
                match_conditions.update(_match_conditions)

    _group_id = {
        "key": "$target.k",
        "value": "$target.v"
    }

    _project_conditions = {
        "target": {"$objectToArray": "$" + fieldname}
    }

    if additional_group_conditions:
        _group_id.update(additional_group_conditions)
        _project_conditions.update(additional_group_conditions)

    aggregation = [
        {
            "$match": match_conditions
        },
        {
            "$project": _project_conditions
        },
        {"$unwind": "$target"},
        {
            "$group": {
                "_id": _group_id,
                "count": {"$sum": 1}
            }
        }
    ]

    results = [_ for _ in db_images.aggregate(aggregation)]

    grouped_count = {}
    for result in results:
        _id_values = result["_id"]
        target_keys = []
        _prefixes = prefixes.copy()
        _prefixes.extend([result["_id"]["key"],result["_id"]["value"]])
        # prefix = ".".join(_prefixes)
        for key, value in _id_values.items():
            if key == "key" or key == "value":
                continue
            _prefixes.extend([str(key), str(value)])
        _prefixes = [str(_) for _ in _prefixes]
        dict_key = ".".join(_prefixes)
        _count = result["count"]

        grouped_count[dict_key] = _count


    return grouped_count

def calculate_stats_dict(db, stats_queries, return_records: bool = True):
    stats_dict = {}
    entity = stats_queries["entity"]
    stats_dict[f"{entity}.{PREFIX_COUNT}"] = db.count_documents({})
    for query in stats_queries["get_count"]:
        stats_dict.update(get_count(db, entity, query))
    for query in stats_queries["get_grouped_count"]:
        stats_dict.update(get_grouped_count(db, entity, **query))
    for query in stats_queries["get_grouped_dict_count"]:
        stats_dict.update(get_grouped_dict_count(db, entity, **query))

    
    if return_records:
        records = []
        for name, value in stats_dict.items():
            record = parse_stats_dict_name(name, value)
            records.append(record)
        return records
        

    else:
        return stats_dict


def get_intervention_image_predictions(intervention_id, db_images):
    aggregation = [
        {
            "$match": {
                FIELDNAME_INTERVENTION_ID: intervention_id
            }
        },
        {
            "$project": {
                "predictions": {"$objectToArray": "$predictions"}
            }
        },
        {
            "$unwind": "$predictions"
        },
        {
            "$project": {
                "label": "$predictions.k",
                "value": "$predictions.v.label"
            }
        },
        {
            "$group": {
                "_id": {
                    "label": "$label",
                    "value": "$value"
                },
                "count": {"$sum": 1}
            }
        }
    ]

    _predictions = db_images.aggregate(aggregation)
    predictions = defaultdict(generate_default_dict)
    for _ in _predictions:
        predictions[str(_["_id"]["label"])][str(_["_id"]["value"])] = _["count"]

    return predictions

def get_intervention_image_annotations(intervention_id, db_images):
    additional_match_conditions = {
        f"{FIELDNAME_INTERVENTION_ID}.{str(intervention_id)}": {FIELDNAME_INTERVENTION_ID: intervention_id}
    }

    _label_count = get_grouped_dict_count(db_images, PREFIX_IMAGE, FIELDNAME_LABELS, additional_match_conditions)

    label_count = defaultdict(generate_default_dict)

    for key, value in _label_count.items():
        components = key.split(".")
        label_count[components[-2]][components[-1]]=value

    return label_count


def get_if_in_dict(_dict, fieldnames):
    value = _dict

    for fieldname in fieldnames:
        if fieldname in value:
            value = value[fieldname]
        else: 
            value = None
            break

    return value


def smooth_predictions(df, label, n_smooth, use_future: bool = True):
    select_1 = df[COLNAME_RESULT_NAME] == label
    select_2 = df[COLNAME_RESULT_TYPE] == RESULT_TYPE_PREDICTION
    select = select_1 & select_2

    selection = df.loc[select].sort_values(FIELDNAME_FRAME_NUMBER, ascending = True)
    consecutive_df_list = split_df_to_consecutive_frame_dfs(selection)
    convolution_result = np.empty(0, dtype = np.bool)
    ids = []
    frame_numbers = []

    for _selection in consecutive_df_list:
        if not len(_selection)>n_smooth:
            _frame_numbers = _selection[FIELDNAME_FRAME_NUMBER].to_list()
            _ids = _selection["_id"].to_list()
            _convolution_result = np.empty((len(_selection)))
            _convolution_result[:] = np.nan
        else:
            prediction_array = np.array(_selection[COLNAME_RESULT_LABEL])
            _frame_numbers = _selection[FIELDNAME_FRAME_NUMBER].to_list()
            _ids = _selection["_id"].to_list()
            if use_future:
                convolution_filter = np.ones(n_smooth)/n_smooth
            else: 
                convolution_filter = np.ones(n_smooth)/(int(n_smooth/2))
                convolution_filter[int(n_smooth/2)+1:] = 0
                
            _convolution_result = np.convolve(prediction_array, convolution_filter, mode = "same")

        convolution_result = np.append(convolution_result, _convolution_result)
        ids.extend(_ids)
        frame_numbers.extend(_frame_numbers)

    return convolution_result, frame_numbers, ids

def values_to_bool(values, threshold):
    values[values>threshold] = 1
    values[values<=threshold] = 0
    values = values.astype(bool)
    return values

def get_smooth_prediction_df(df, label, n_smooth, threshold, use_future: bool = True):
    values_smooth, frame_numbers, ids = smooth_predictions(df, label, n_smooth)
    labels_smooth = values_to_bool(values_smooth, threshold)

    records = []

    for n in range(len(values_smooth)):
        records.append({
            "_id": ids[n],
            FIELDNAME_FRAME_NUMBER: frame_numbers[n],
            COLNAME_PREDICTION_VALUE: values_smooth[n],
            COLNAME_RESULT_LABEL: labels_smooth[n],
            COLNAME_RESULT_NAME: label,
            COLNAME_RESULT_TYPE: RESULT_TYPE_PREDICTION_SMOOTH
        })

    df = pd.DataFrame.from_records(records)

    return df

def get_prediction_flanks(intervention, instance_min_seconds: int = 1):
    from tqdm import tqdm
    has_framecount = False
    frames_total = None
    fps = None
    _flanks = defaultdict(list)
    flanks = defaultdict(dict)

    has_meta = FIELDNAME_VIDEO_METADATA in intervention.intervention
    if has_meta:
        has_framecount = FIELDNAME_FRAMES_TOTAL in intervention.intervention[FIELDNAME_VIDEO_METADATA]

    if has_framecount:
        frames_total = int(intervention.intervention[FIELDNAME_VIDEO_METADATA][FIELDNAME_FRAMES_TOTAL])

    if frames_total:
        fps = int(intervention.intervention[FIELDNAME_VIDEO_METADATA][FIELDNAME_FPS])

    if not fps:
        warnings.warn("Insufficient Metadata")
    else:
        instance_min_n_frame = instance_min_seconds * fps

        # intervention.update_smooth_labels()

        aggregation = [
            {
                "$match": {FIELDNAME_INTERVENTION_ID: intervention._id}
            },
            {
                "$project": {FIELDNAME_PREDICTIONS_SMOOTH: {"$objectToArray": f"${FIELDNAME_PREDICTIONS_SMOOTH}"}, FIELDNAME_FRAME_NUMBER:1}
            },
            {"$unwind": "$"+FIELDNAME_PREDICTIONS_SMOOTH},
        ]

        images = intervention.db_images.aggregate(aggregation)
        for img in images:
            if img[FIELDNAME_PREDICTIONS_SMOOTH]["v"]:
                _flanks[f"{img[FIELDNAME_PREDICTIONS_SMOOTH]['k']}.{img[FIELDNAME_PREDICTIONS_SMOOTH]['v']}"].append(img[FIELDNAME_FRAME_NUMBER])
        
        for key in _flanks:
            _flanks[key].sort()

        for label_choice, framelist in _flanks.items():
            _start = framelist[0]
            _end = None
            flanks[label_choice][FIELDNAME_FLANK_RANGES] = []
            flanks[label_choice][FIELDNAME_FLANK_FRAMES] = []

            for i, n_frame in enumerate(framelist):
                n_previous_frame = framelist[i-1]

                _new_instance = (n_frame-n_previous_frame) > 1
                if _new_instance:
                    _end = n_previous_frame
                    n_frames_flank = _end - _start
                    if n_frames_flank > instance_min_n_frame:
                        flanks[label_choice][FIELDNAME_FLANK_RANGES].append((_start, _end))
                        _start = n_frame
                        _end = None
                    else:
                        _start = n_frame
                        _end = None
                    
            for range_tuple in flanks[label_choice][FIELDNAME_FLANK_RANGES]:
                flanks[label_choice][FIELDNAME_FLANK_FRAMES].append(
                    [
                        intervention.intervention[FIELDNAME_FRAMES][str(_)] 
                        for _ in range(range_tuple[0], range_tuple[1]+1)
                    ]
                )
        
        intervention.db_interventions.update_one({"_id": intervention._id}, {"$set": {FIELDNAME_PREDICTION_FLANKS: dict(flanks)}})
        intervention.refresh()
        return flanks

def split_df_to_consecutive_frame_dfs(df):
    last_n = None
    groups = []
    group = []

    for index, row in df.iterrows():
        if not last_n:
            last_n = row[FIELDNAME_FRAME_NUMBER]
            group.append(index)
            continue   

        frame_diff = row[FIELDNAME_FRAME_NUMBER] - last_n
        if frame_diff>1:
            groups.append(df.loc[group])
            group = []
        
        group.append(index)
        last_n = row[FIELDNAME_FRAME_NUMBER]

    groups.append(df.loc[group])
    return groups



def dict_of_lists_to_text(_dict) -> str:
    text = ""

    for key, value in _dict.items():
        if isinstance(value, list):
            new = "\n".join(value)
        new = key + "\n" + new
        text = text + "\n" + new

    return text


def generate_token_value_query(report_tokens, patho_tokens, _or: bool = True):
    if _or: condition = "$or"
    else: condition = "$and"

    query = {
        condition: {
            f"{FIELDNAME_TOKENS}.\
                {FIELDNAME_TOKENS_REPORT}.\
                {FIELDNAME_TOKENS_VALUE}": {"$in": report_tokens},
            f"{FIELDNAME_TOKENS}.\
                {FIELDNAME_TOKENS_PATHO}.\
                {FIELDNAME_TOKENS_VALUE}": {"$in": patho_tokens},
        }
    }

    return query

def get_train_image_query(label, value, test_intervention_ids, extend_conditions, extend_agg):
    conditions = [
        field_value_query(f"{FIELDNAME_LABELS}.{label}", value),
        fieldvalue_nin_list_query(FIELDNAME_INTERVENTION_ID, test_intervention_ids)
    ]

    if extend_conditions:
        conditions.extend(extend_conditions)

    aggregation = [match_logical_and_aggregation(conditions)]

    if extend_agg:
        aggregation.extend(extend_agg)

    return aggregation


def check_extern_video_annotation_timestamp(video_key, str_timestamp, db_interventions, time_format = "%Y-%m-%dT%H:%M:%S.%fZ"):
    timestamp = dt.strptime(str_timestamp, time_format)
    intervention = db_interventions.find_one({FIELDNAME_VIDEO_KEY: video_key}, {FIELDNAME_VIDEO_ANNOTATION_TIMESTAMP: 1, FIELDNAME_VIDEO_KEY: 1})

    if FIELDNAME_VIDEO_ANNOTATION_TIMESTAMP in intervention:
        if intervention[FIELDNAME_VIDEO_ANNOTATION_TIMESTAMP]<timestamp:
            return False
        else: 
            return True
    else: 
        return True

def generate_db_extern_video_annotation_dict(annotations, time_format):
    video_annotations = {}

    for annotation in annotations:
        annotation_group_ids = [_["annotationGroupId"] for _ in annotation["sequences"]]
        annotation_group_ids = list(set(annotation_group_ids))
        user_id = annotation["userId"]
        session_date = dt.strptime(annotation["sessionDate"], time_format)
        label_group_id = annotation["labelGroupId"]
        
        for sequence in annotation["sequences"]:
            label = sequence["annotationName"].lower()
            value = sequence["annotationValue"]
            if isinstance(value, str):
                if "true" in value:
                    value = True
                elif "false" in value:
                    value = False
            flank = (sequence["startFrame"],sequence["endFrame"])
            

            if label not in video_annotations:
                video_annotations[label] = {}
            if user_id not in video_annotations[label]:
                video_annotations[label][user_id] = {}
            if session_date not in video_annotations[label][user_id]:
                video_annotations[label][user_id][session_date] = []

            video_annotations[label][user_id][session_date].append(
                {
                    "label": label,
                    "value": value,
                    "flank": flank,
                    "label_group_id": label_group_id
                }
            )

    

    return video_annotations


def select_relevant_extern_video_annotations(video_annotations: dict, annotation_priority_user_id: List):
    selected_annotations = []
    
    for label, user_annotations in video_annotations.items():
        for user_id in annotation_priority_user_id:
            if user_id in user_annotations:
                dates = [_ for _ in user_annotations[user_id].keys()]
                dates.sort(reverse = True)
                selected_annotations.extend(user_annotations[user_id][dates[0]])
    
    return selected_annotations


def add_false_labels_for_test_data_annotations(intervention, extern_video_annotations, annotation_group_labels):
    n_frames_total = int(intervention[FIELDNAME_VIDEO_METADATA][FIELDNAME_FRAMES_TOTAL])
    extern_video_annotations = extern_video_annotations.copy()

    flanks = {_:defaultdict(list) for _ in annotation_group_labels}

    for annotation in extern_video_annotations:
        flanks[annotation["label"]][annotation["value"]].append(annotation["flank"])
        flanks[annotation["label"]]["set_false"] = []
        flanks[annotation["label"]]["not_false"] = []

    for label in flanks.keys():
        for value in flanks[label].keys():
            if value == False or value == "set_false" or value=="not_false":
                continue
            flanks[label]["not_false"].extend(flanks[label][value])


        flanks[label]["not_false"] = list(set(flanks[label]["not_false"]))
        flanks[label]["not_false"].sort(key=itemgetter(0))
        last_end = None
        for sequence in flanks[label]["not_false"]:
            current_start = sequence[0]

            if not last_end:
                if current_start != 0:
                    last_end = 0
                else: 
                    last_end = sequence[1]
                    continue
       
            flanks[label]["set_false"].append((last_end, current_start))
            last_end = sequence[1]
        
        if not last_end:
            flanks[label]["set_false"].append((0, n_frames_total))

        elif last_end < n_frames_total:
            flanks[label]["set_false"].append((last_end, n_frames_total))
            
        elif last_end > n_frames_total:
            last_flank = flanks[label]["set_false"][-1]
            flanks[label]["set_false"][-1] = (last_flank[0], n_frames_total)


        extern_video_annotations.extend([
            {"label": label, "value": False, "flank": flank} for flank in flanks[label]["set_false"]
        ])

        _compare = []
        _unique = []
        for _ in extern_video_annotations:
            to_compare = (_["label"], _["flank"])
            if to_compare in _compare:
                continue
            else: 
                _unique.append(_)
                _compare.append(to_compare)

    return _unique
        