import re
import warnings
from pathlib import Path
import json
from .validation import validate_adrian_annotation_json

# Patterns to parse first round annotations
regex_patterns = {
    "anatomy_ileocecalvalve": re.compile("anatomie.ileocecalvalve|ileocecal\svalve"),
    "anatomy_ileum": re.compile("(.*)\.*(ileum)"),
    "anatomy_appendix": re.compile("appendix"),
    "polyp": re.compile("p\d\.*(.*)"),
    "polyp_nice": re.compile("nice(.*)"),
    "normal": re.compile("normal\.(.*)|normalmucosa\.(.*)"),
    "polyp_paris": re.compile("paris(.*)"),
    "polyp_size": re.compile("size(.*)"),
    "igien": re.compile("igien"),
    "diverticula": re.compile("diverticula"),
}


def process_webserver_json(
    video_json: dict, base_path_frames: Path, skip_frame_factor: int, path_logs: Path
) -> dict:
    """Processes single json from webserver. Returns intervention insert dict and list of frame insert dicts.

    Args:
        video_json (dict): json from webserver
        base_path_frames (Path): Path object pointing to directory where all frame paths are stored
        skip_frame_factor (int): every n-th frame was annotated in first round
        path_logs (Path): Path object pointing to directory where logs should be stored

    Returns:
        dict: [description]
    """
    video_path = Path(video_json["videoPath"])
    video_hash = video_path.parent.name
    annotation_paths = [Path(_) for _ in video_json["annotationPaths"]]
    intervention_type = get_intervention_type(video_json["videoType"])

    annotations = []
    for _ in annotation_paths:
        with open(_, "r", encoding="latin1") as f:
            annotations.append(json.load(f))

    # Select only latest annotation
    annotation = annotations[-1]

    # Validate
    if not validate_adrian_annotation_json(annotation):
        return None
    else:
        intervention_content = get_intervention_content(
            annotation, video_path, intervention_type, skip_frame_factor
        )
        intervention_content["origin"] = video_json["center"].lower()
        insert_intervention, insert_frames = make_insert_dicts(
            intervention_content, base_path_frames
        )
   
        insert_intervention["video_hash"] = video_hash
        if "patho" in video_json:
            insert_intervention["patho_raw"] = video_json["patho"]
        if "report" in video_json:
            insert_intervention["report_raw"] = video_json["report"]

        if Path(insert_frames[0]["path"]).parent.exists():
            with open(
                path_logs.joinpath(f"{Path(video_json['videoPath']).name}.json"), "w"
            ) as f:
                json.dump(video_json, f)
            warnings.warn(f"Path already exists!\n{video_path}")
            return None
        else:
            # if "frames" in insert_intervention:
            #     insert_intervention["frames"] = [str(_) for _ in insert_intervention["frames"]]
            
            return {
                "insert_intervention": insert_intervention,
                "insert_frames": insert_frames,
            }


def get_intervention_content(
    annotation: dict, video_path: Path, intervention_type: str, skip_frame_factor: int
):
    """Create an intervention_content dictionary containing.

    Args:
        annotation (dict): Adrian annotation file
        video_path (Path): Path object pointing to video file
        intervention_type (str): type of intervention
        skip_frame_factor (int): Indicates how many frames were initially skipped for first round annotation, e.g.:
        3 means that every third frame was extracted

    Returns:
        dict: Dictionary containing
        "video_key", "video_path", "annotation_raw", "frames_total", "annotation", "origin", "intervention_type"
    """
    intervention_content = {}

    extracted_annotation = annotation_conversion(annotation, skip_frame_factor)
    key = extracted_annotation["video_key"]

    intervention_content["video_key"] = key
    intervention_content["video_path"] = video_path.as_posix()
    intervention_content["annotation_raw"] = annotation
    intervention_content["frames_total"] = annotation["metadata"]["imageCount"]
    intervention_content["annotation"] = extracted_annotation
    intervention_content["intervention_type"] = intervention_type

    return intervention_content


def match_labels(label, regex_patterns):
    for key, pattern in regex_patterns.items():
        _ = pattern.match(label)
        if _:
            if key == "polyp_nice":
                _ = {"polyp_nice": _.groups()[0]}
            if key == "polyp_paris":
                _ = {"polyp_paris": _.groups()[0]}
            if key == "igien":
                _ = {"igien": True}
            if key == "polyp_size":
                _ = {"polyp_size": _.groups()[0]}
            if key == "polyp":
                if _.groups()[0]:
                    _ = {"polyp": True, "type": _.groups()[0]}
                else:
                    _ = {"polyp": True, "type": "single"}
            if key == "normal":
                _type = None
                for _group in _.groups():
                    if _group:
                        if _group == "start":
                            _type = "n_start"
                        elif "end" in _group:
                            _type = "n_ende"
                if not _type:
                    _type = "single"
                _ = {"polyp": False, "type": _type}
            if key == "anatomy_ileocecalvalve":
                _ = {"anatomy": "ileocecalvalve"}
            if key == "anatomy_ileum":
                if _.groups()[0]:
                    _ = {"anatomy": {"ileum": _.groups()[0].replace(".", "")}}
                else:
                    _ = {"anatomy": "ileum"}

            if key == "anatomy_appendix":
                _ = {"anatomy": "appendix"}
            if key == "diverticula":
                _ = {"diverticulum": True}

            return _


def get_intervention_type(_type):
    _type = _type.strip().lower()
    if _type == "kolo":
        _type = "Koloskopie"
    elif _type == "coloscopy":
        _type = "Koloskopie"
    elif _type == "gastro":
        _type = "Gastroskopie"
    elif _type == "ögd":
        _type = "Gastroskopie"
    elif _type == "gastro/kolo":
        _type = "Gastroskopie/Koloskopie"
    elif _type == "ögd/kolo":
        _type = "Gastroskopie/Koloskopie"
    elif _type == "kolo0":
        _type = "Koloskopie"
    else:
        warnings.warn(f"unknown type:{_type}")

    return _type


def extract_superframes(superframes) -> dict:
    annotations = []
    frame_numbers = []

    for frame in superframes["childFrames"]:
        label = [match_labels(_.lower(), regex_patterns) for _ in frame["classes"]]
        annotations.append(label)
        frame_numbers.append(frame["frameNumber"])

    labels = {_: [] for _ in frame_numbers}
    for i, label in enumerate(annotations):
        for _class in label:
            labels[frame_numbers[i]].append(_class)

    return labels


def extract_box_images(box_images):
    annotations = {}
    for image in box_images:
        bbs = []
        for _bb in image["boundingBox"]:
            if "secondaryLabels" in _bb:
                sec_labels = _bb["secondaryLabels"].copy()
            else:
                sec_labels = []
            bb = {
                "x": _bb["x"],
                "y": _bb["y"],
                "width": _bb["width"],
                "height": _bb["height"],
                "labels": sec_labels,
            }
            bb["labels"].append(_bb["label"])
            bb["labels"] = [
                match_labels(_.lower(), regex_patterns) for _ in bb["labels"]
            ]
            bbs.append(bb)
        annotations[image["frameNumber"]] = bbs
        if image["label"]:
            _ = match_labels(image["label"].lower(), regex_patterns)
            annotations[image["frameNumber"]].append(_)

    return annotations


def fill_polyp_frames(video_object):
    state = None
    updates = []
    start_frame = None
    n_start_frame = None
    for frame_number, annotation_list in video_object["classifications"].items():
        for annotation in annotation_list:
            if annotation:
                if "polyp" in annotation:
                    if annotation["polyp"] is True:
                        state = annotation["type"]

                        if state == "start":
                            start_frame = frame_number

                        elif state == "single":
                            pass
                        elif state == "end":
                            if start_frame:
                                updates.extend(
                                    [
                                        {_: {"polyp": True, "type": "filled"}}
                                        for _ in range(start_frame, frame_number + 1)
                                    ]
                                )
                                start_frame = None
                            else:
                                warnings.warn(
                                    f"End frame of annotation recognized before start frame. \
                                        This annotation will be omitted: \n{annotation}"
                                )

                    if annotation["polyp"] is False:
                        n_state = annotation["type"]
                        if n_state == "n_start":
                            n_start_frame = frame_number
                        elif n_state == "single":
                            pass
                        elif n_state == "n_ende":
                            if n_start_frame:
                                updates.extend(
                                    [
                                        {_: {"polyp": False, "type": "filled"}}
                                        for _ in range(n_start_frame, frame_number + 1)
                                    ]
                                )
                                n_start_frame = None
                            else:
                                warnings.warn(
                                    f"End frame of annotation recognized before start frame. \
                                        This annotation will be omitted: \n{annotation}"
                                )

    for update in updates:
        if update:
            _frame_number = list(update.keys())[0]
            if _frame_number in video_object["classifications"]:
                video_object["classifications"][_frame_number].append(
                    update[_frame_number]
                )
            else:
                video_object["classifications"][_frame_number] = [update[_frame_number]]

    return video_object


def fill_ileum_frames(video_object):
    state = None
    updates = []
    for frame_number, annotation_list in video_object["classifications"].items():
        for annotation in annotation_list:
            if annotation:
                if "anatomy" in annotation:
                    if isinstance(annotation["anatomy"], dict):
                        if "ileum" in annotation["anatomy"]:
                            state = annotation["anatomy"]["ileum"]
                            if state == "enter":
                                start_frame = frame_number
                            elif state == "single":
                                pass
                            elif state == "exit":
                                updates.extend(
                                    [
                                        {_: {"anatomy": "ileum"}}
                                        for _ in range(start_frame, frame_number + 1)
                                    ]
                                )

    for update in updates:
        if update:
            _frame_number = list(update.keys())[0]
            if _frame_number in video_object["classifications"]:
                video_object["classifications"][_frame_number].append(
                    update[_frame_number]
                )
            else:
                video_object["classifications"][_frame_number] = [update[_frame_number]]

    return video_object


def multiply_frame_keys(frame_dict: dict, skip_frame_factor: int):
    _classifications = {}
    for _key, value in frame_dict.items():
        new_key = _key * skip_frame_factor
        _classifications[new_key] = value
    return _classifications


def annotation_conversion(annotation: dict, skip_image_factor: int):
    video_object = {
        "video_key": annotation["metadata"]["videoFile"],
        "width": annotation["metadata"]["resolutionWidth"],
        "height": annotation["metadata"]["resolutionHeight"],
        "image_count": annotation["metadata"]["imageCount"] * skip_image_factor,
        "classifications": [
            extract_superframes(superframe)
            for superframe in annotation["superframes"]
            if extract_superframes(superframe)
        ],
        "bounding_boxes": extract_box_images(annotation["images"]),
    }
    _tmp = {}
    for _ in video_object["classifications"]:
        for key, value in _.items():
            if key in _tmp:
                _tmp[key].extend(value)
            else:
                _tmp[key] = value

    video_object["classifications"] = _tmp

    video_object["classifications"] = multiply_frame_keys(
        video_object["classifications"], skip_image_factor
    )
    video_object["bounding_boxes"] = multiply_frame_keys(
        video_object["bounding_boxes"], skip_image_factor
    )

    video_object = fill_polyp_frames(video_object)
    video_object = fill_ileum_frames(video_object)
    return video_object


def make_insert_dicts(intervention: dict, base_path_frames: Path):
    frames = []
    frames.extend(list(intervention["annotation"]["classifications"].keys()))
    frames.extend(list(intervention["annotation"]["bounding_boxes"].keys()))
    frames = list(set(frames))
    frames.sort()

    # intervention["frames"] = [str(_) for _ in frames]

    image_dicts = []
    for frame in frames:
        _dict = {
            "origin": intervention["origin"],
            "video_key": intervention["video_key"],
            "intervention_id": None,
            "path": base_path_frames.joinpath(intervention["video_key"])
            .joinpath(f"{frame}.png")
            .as_posix(),
            "labels": {"annotations": {}, "predictions": {}},
            "n_frame": frame,
            "in_progress": False,
        }

        _annotations = []
        if frame in intervention["annotation"]["bounding_boxes"]:
            bb_labels = []
            for bbox in intervention["annotation"]["bounding_boxes"][frame]:
                bbox["labels"] = [_ for _ in bbox["labels"] if _]
                for _bb_label in bbox["labels"]:
                    for key, value in _bb_label.items():
                        bb_labels.append({key: value})
                bb_labels.append(
                    {
                        "polyp_detection_bbox": {
                            key: value for key, value in bbox.items() if key != "labels"
                        }
                    }
                )
            _annotations.extend(bb_labels)

        if frame in intervention["annotation"]["classifications"]:
            _annotations.extend(intervention["annotation"]["classifications"][frame])

        annotations = {}
        for _ in _annotations:
            if _:
                key = list(_.keys())[0]
                if key in annotations:
                    pass
                elif key == "type":
                    pass
                else:
                    annotations[key] = _[key]

        _dict["labels"]["annotations"] = annotations
        # del intervention["annotation"]
        image_dicts.append(_dict)

    return intervention, image_dicts
