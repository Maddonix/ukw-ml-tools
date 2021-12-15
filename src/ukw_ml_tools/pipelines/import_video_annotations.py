from ..extern.requests import get_extern_annotations
from ..mongodb.data_import import (
    filter_extern_annotation_by_latest_date,
    get_video_segmentation_extern,
    generate_db_images_for_frame_extraction,
    get_image_updates_for_video_segmentation
)
from ..media.frame_extraction import extract_frames_for_video_segmentation
from pymongo.collection import Collection
from ..mongodb.get_objects import get_intervention
from ..mongodb.update import (
    update_video_segments_annotation,
    insert_frame, 
    add_frames_to_framedict,
)
from ..mongodb.base import filter_existing_frames_by_intervention
from ..classes.config import LabelSettings

def get_new_annotation_videos(url:str, auth: str, db_interventions:Collection):
    annotated = get_extern_annotations(url, auth)
    print("Annotated Videos:", len(annotated))
    annotated_new = filter_extern_annotation_by_latest_date(annotated, db_interventions)
    print(len(annotated_new), "Videos have new annotations")
    return annotated_new

def import_new_video_annotation(annotated_video, label_group_settings:LabelSettings, url, auth, db_interventions):
    extern_latest_annotation = annotated_video.date
    video_key = annotated_video.video_key

    video_segmentation = get_video_segmentation_extern(
        video_key, extern_latest_annotation,
        label_group_settings.default_values,
        url,
        auth,
        db_interventions
    )
    
    r = update_video_segments_annotation(video_key, video_segmentation, db_interventions)
    return r
    

def extract_and_generate_frames(annotated_video, base_dir_frames, url, auth, db_interventions, db_images):
    video_key = annotated_video.video_key
    paths = extract_frames_for_video_segmentation(
        video_key,
        url,
        auth,
        db_interventions,
        base_dir_frames
    )
    paths = filter_existing_frames_by_intervention(video_key, paths, db_interventions)
    images = generate_db_images_for_frame_extraction(video_key, paths, db_interventions)
    inserted_ids = {}
    for image in images:
        r = insert_frame(image, db_images)
        inserted_ids[image.metadata.frame_number] = r.inserted_id
    add_frames_to_framedict(video_key, inserted_ids, db_interventions)
    return inserted_ids

def set_image_label_updates_from_video_segmentations(
        annotated_video,
        ai_config_dict,
        db_interventions,
        db_images,
        set_in_db = True
    ):
    """
    Executes Updates and sets
    Returns Tuples like ({"_id": ObjectId("asd"), {"$set": {"annotations.name": Annotation}}})
    """
    video_key = annotated_video.video_key
    timestamp = annotated_video.date
    updates = get_image_updates_for_video_segmentation(
        video_key, db_interventions, ai_config_dict
    )
    if set_in_db:
        for update in updates:
            db_images.update_one(update[0], update[1])
    
    return updates