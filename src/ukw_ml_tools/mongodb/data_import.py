import requests
from ..classes.extern import ExternAnnotatedVideo, VideoExtern
from typing import List, Tuple
from .base import get_extern_ids, get_base_image_dict_from_intervention
from ..classes.intervention import Intervention
from ..classes.annotation import VideoSegmentationAnnotation
from ..classes.image import Image
from ..extern.requests import (
    get_extern_annotations,
    get_extern_video_annotation,
    get_extern_interventions
)
from ..extern.conversions import (
    annotations_by_label_group,
    get_flank_dict,
    get_labels_with_test_annotation
)
from collections import defaultdict
from .get_objects import get_intervention
from ..labels.conversions import get_default_label_flanks
from .instance_import import polyps_to_instances

def filter_duplicate_keys(interventions):
    _duplicates = []
    video_keys = []
    names = [_.path.name for _ in interventions]
    for _ in names:
        if _ in video_keys:
            _duplicates.append(_)
        else:
            video_keys.append(_)

    duplicates = [(_.path.name, _.id_extern) for _ in interventions if _.path.name in _duplicates]
    interventions = [_ for _ in interventions if not _.path.name in _duplicates]
    print(f"Dropped videos for {len(duplicates)} unique video_keys")

    return interventions, duplicates

def filter_extern_annotation_by_latest_date(annotated_video_list: List[ExternAnnotatedVideo], db_interventions):
    new_annotated_videos = []
    for annotated_video in annotated_video_list:
        intervention = get_intervention(annotated_video.video_key, db_interventions)
        if intervention.video_segments_annotation:
            dates = [annotation.date for name, annotation in intervention.video_segments_annotation.items()]
        _is_new = False
        for date in dates:
            if annotated_video.date > date:
                _is_new = True
                break
        if _is_new:
            new_annotated_videos.append(annotated_video)

    return new_annotated_videos

def get_new_extern_interventions(
        url_extern: str, auth_extern: Tuple[str], db_interventions
    ) -> Tuple[List[VideoExtern]]:
    """Calls get_extern_interventions, filters list of VideoExtern\
    Objects for duplicate keys and for extern_ids which already exist in\
    the given db.

    Args:
        url_extern (str): e.g. "https://10.235.14.33:8443/data"
        auth_extern (str): e.g. ("user", "password")
        db_interventions ([type]): 

    Returns:
        Tuple[List[VideoExtern]]: Returns Tuple containing 3 Lists:\
            interventions, failed conversions, duplicates
    """
    videos = get_extern_interventions(url_extern, auth_extern)

    videos, duplicates = filter_duplicate_keys(videos)

    existing_extern_video_ids = get_extern_ids(db_interventions)
    videos = [_ for _ in videos if _.id_extern not in existing_extern_video_ids]
    interventions = []
    failed = []
    for _ in videos:
        try:
            interventions.append(Intervention(**_.to_intervention_dict(db_interventions)))
        except:
            failed.append(_)
    print(f"Failed to convert {len(failed)} interventions")
    
    return interventions, failed, duplicates

def get_video_segmentation_extern(video_key, latest_session, default_values_dict, url, auth, db_interventions):
    video_annotations = get_extern_video_annotation(video_key, url, auth)

    video_annotation_dict = annotations_by_label_group(video_annotations)
    flanks, lookup = get_flank_dict(video_annotation_dict, default_values_dict)
    flanks["test"].extend(get_default_label_flanks(flanks["test"], lookup, video_key, db_interventions))
    flanks = flanks["train"] + flanks["test"]
    video_segmentation = {}
    for flank in flanks:
        name = flank.name
        if not name in video_segmentation:
            video_segmentation[name] = VideoSegmentationAnnotation(
                source = "web_annotation_flanks",
                annotator_id=0,
                date = latest_session,
                name = name,
                value = []
                )
        video_segmentation[name].value.append(flank)

    return video_segmentation

def generate_db_images_for_frame_extraction(video_key, paths, db_interventions):
    intervention = get_intervention(video_key, db_interventions)
    base_image_dict = get_base_image_dict_from_intervention(intervention)
    images = []

    for n_frame, path in paths.items():
        image_dict = base_image_dict.copy()
        image_dict["metadata"] = {
            "frame_number": n_frame,
            "path": path,
            "is_extracted": False,
            "is_frame": True
        }
        images.append(Image(**image_dict))

    return images

from ..classes.config import AiLabelConfig

def video_segmentation_to_image_labels(
    video_segmentation_annotation:VideoSegmentationAnnotation,
    ai_label_config_dict,
    skip_labels:str,
    is_annotation:bool = True
    ):
    
    annotator_id = video_segmentation_annotation.annotator_id
    flanks = video_segmentation_annotation.value
    date = video_segmentation_annotation.date

    if is_annotation:
        _class_selector = "annotation_class"
    else:
        _class_selector = "prediction_class"

    image_labels = defaultdict(list)
    for flank in flanks:
        if flank.name in skip_labels:
            continue
        ai_label_config = ai_label_config_dict[flank.name]
        ai_label_config = AiLabelConfig(**ai_label_config)
        _class = ai_label_config.dict()[_class_selector]
        assert flank.value in ai_label_config.choices

        for i in range(flank.start, flank.stop):
            image_labels[i].append(_class(
                source = "web_annotation_flanks",
                annotator_id=annotator_id,
                date = date,
                choices = ai_label_config.choices,
                name = flank.name,
                value = flank.value
            ))
    return image_labels


def image_label_dict_to_updates(video_key, image_label_dict, db_interventions):
    intervention = get_intervention(video_key, db_interventions)
    updates = []
    for n, annotations in image_label_dict.items():
        image_id = intervention.frames[n]
        _set = {}
        for annotation in annotations:
            _set[f"annotations.{annotation.name}"] = annotation.dict()
        
        
        _update = ({
            "_id": image_id,
            f"annotations.{annotation.name}.source": {"$in": ["web_annotation_flanks", None]}
            },
            {"$set": _set})
        updates.append(_update)

    return updates   


def get_image_updates_for_video_segmentation(
        video_key, 
        db_interventions,
        ai_label_config_dict,
        skip_labels = [
            # "withdrawal",
            # "resection",
            "polyp_1",
            "polyp_2",
            "polyp_3",
            "new_polyp_1",
            "new_polyp_2",
            "new_polyp_3",
            ]
    ):

    intervention = get_intervention(video_key, db_interventions)
    db_images = db_interventions.database.Images
    db_instances = db_interventions.database.VideoInstances
    intervention = polyps_to_instances(intervention, db_interventions, db_images, db_instances)
    video_segmentation_annotation = intervention.video_segments_annotation


    image_labels = defaultdict(list)
    for name, segmentation in video_segmentation_annotation.items():
        _image_labels = video_segmentation_to_image_labels(
            segmentation,
            ai_label_config_dict,
            is_annotation=True,
            skip_labels=skip_labels
        )

        for n_frame, value in _image_labels.items():
            image_labels[n_frame].extend(value)

    updates = image_label_dict_to_updates(video_key, image_labels, db_interventions)

    return updates