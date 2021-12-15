from .import_videos import import_new_extern_interventions
from .import_video_annotations import (
    get_new_annotation_videos,
    import_new_video_annotation,
    extract_and_generate_frames,
    set_image_label_updates_from_video_segmentations
)
from .check_extracted_frames import (
    get_video_keys_of_not_extracted_frames,
    update_frames_extracted
)
from ..classes.config import Configuration, AiLabelConfig
from datetime import datetime as dt
import time
from .set_compund_labels import set_compound_labels

def daily_service_pipeline(cfg:Configuration, ai_config_dict, ai_config:AiLabelConfig):
    db_images, db_interventions, db_test_data, db_train_data = cfg.get_databases()
    url, auth = cfg.get_extern_tuple()
    label_settings = cfg.label_settings
    base_dir_frames = cfg.base_paths.frames



    log = {}
    log["new_intervention_ids"] = import_new_extern_interventions(url, auth, db_interventions)
    
    annotated_new = get_new_annotation_videos(url, auth, db_interventions)

    # Import new flank annotations to videos
    print("Import new annotated videos")
    for annotated_video in annotated_new:
        r = import_new_video_annotation(annotated_video, label_settings, url, auth, db_interventions)


    # Extract Frames and Generate Frame Entries for DB
    print("Extract Frames for new annotations")
    log["extracted_frames"] = {}
    for annotated_video in annotated_new:
        log[annotated_video.video_key] = extract_and_generate_frames(
            annotated_video, base_dir_frames, url, auth, db_interventions, db_images
        )

    # Set Image Labels
    print("Set new annotations")
    for annotated_video in annotated_new:
        set_image_label_updates_from_video_segmentations(
            annotated_video,
            ai_config_dict,
            db_interventions,
            db_images,
            set_in_db = True
        )

    # Set Colo Segmentation Labels
    print("Calculate and set compound labels: Colo Segmentation")
    name = "colo_segmentation"
    exclude_initially = [
        "outside",
        "blurry"
    ]
    base_annotation_dict = {
        "source": "calculation",
        "annotator_id": 0,
        "date": dt.now(),
        "name": name,
        "choices": ai_config.choices,
    }
    set_compound_labels(name, base_annotation_dict, ai_config, db_images)

    
    

    time.sleep(60*10*60*1.5)
    video_keys = get_video_keys_of_not_extracted_frames(db_images)
    
    for video_key in video_keys:
        update_frames_extracted(video_key, db_interventions, db_images)

    return log