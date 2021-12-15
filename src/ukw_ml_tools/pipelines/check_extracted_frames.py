from a_ukw_ml_tools.mongodb.validation import check_intervention_frames_extracted
from a_ukw_ml_tools.mongodb.update import update_many_frame_is_extracted
from ..mongodb.get_objects import get_intervention

def get_video_keys_of_not_extracted_frames(db_images):
    """
    Matches metadata.is_extracted: False and metadata.path: {"$exists": True}
    """
    
    r = db_images.aggregate([
        {"$match": {"metadata.is_extracted": False, "metadata.path": {"$exists": True}}},
        {
            "$group": {
                "_id": "$video_key"
            }
        }
    ])
    return [_["_id"] for _ in r]

def update_frames_extracted(video_key, db_interventions, db_images):
    intervention = get_intervention(video_key, db_interventions)

    is_extracted_dict = check_intervention_frames_extracted(video_key, db_images, db_interventions)
    ids_extracted = [intervention.frames[i] for i, value in is_extracted_dict.items() if value]
    ids_not_extracted = [intervention.frames[i] for i, value in is_extracted_dict.items() if not value]

    update_many_frame_is_extracted(True, ids_extracted, db_images)
    update_many_frame_is_extracted(False, ids_not_extracted, db_images)