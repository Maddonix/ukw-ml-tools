from bson import ObjectId
from typing import List
from .get_objects import get_images_by_id_list, get_intervention
from tqdm import tqdm

def video_keys_unique(db_interventions):
    video_keys = db_interventions.find({"video_key": {"$exists": True}}, {"video_key": 1})
    video_keys = [_ for _ in video_keys]
    _keys = []
    _duplicates = []

    for _ in video_keys:
        if _["video_key"] in _keys:
            _duplicates.append(_["video_key"])
        else:
            _keys.append(_["video_key"])
        
    duplicates = [_ for _ in video_keys if _["video_key"] in _duplicates]

    return duplicates
    
def ids_extern_unique(db_interventions):
    ids_extern = db_interventions.find({"ids_extern": {"$exists": True}}, {"ids_extern": 1})
    ids_extern = [_ for _ in ids_extern]
    _keys = []
    _duplicates = []

    for _ in ids_extern:
        if _["ids_extern"] in _keys:
            _duplicates.append(_["ids_extern"])
        else:
            _keys.append(_["ids_extern"])
        
    duplicates = [_ for _ in ids_extern if _["ids_extern"] in _duplicates]

    return duplicates

def check_intervention_frames_extracted(video_key, db_images, db_interventions):
    intervention = get_intervention(video_key, db_interventions)
    id_list = [_id for n, _id in intervention.frames.items()]
    images = get_images_by_id_list(id_list, db_images)

    is_extracted_dict = {}
    for image in images:
        if image.metadata.path:
            is_extracted_dict[image.metadata.frame_number] = image.metadata.path.exists()
        else: 
            is_extracted_dict[image.metadata.frame_number] = False
    return is_extracted_dict