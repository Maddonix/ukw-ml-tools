import os
from ukw_ml_tools.mongodb.base import get_base_image_dict_from_intervention
import shutil
from ..classes.image import Image

def get_freeze_dir_intern(intervention, base_dir):
    _ = str(intervention.id)
    target_dir = base_dir.joinpath(_)
    
    if not target_dir.exists():
        os.mkdir(target_dir)
        os.system(f"chgrp -R lux_tomcat {target_dir.as_posix()}")

    return target_dir

def get_freeze_paths_cwd(intervention, base_dir):
    target_dir = base_dir.joinpath(intervention.get_cwd_hex())
    if not target_dir.exists():
        return False
    img_paths = [_ for _ in target_dir.iterdir() if _.suffix==".jpg" and _.name.startswith("O")]
    return img_paths

def import_cwd_freezes(intervention, freeze_base_dir, cwd_base_dir, db_images, db_interventions):
    dir_intern = get_freeze_dir_intern(intervention, freeze_base_dir)
    img_paths = get_freeze_paths_cwd(intervention, cwd_base_dir)

    if not intervention.freezes == {}:
        return False
    if not img_paths:
        return False


    img_paths_intern = [dir_intern.joinpath(_.name) for _ in img_paths]

    base_img_dict = get_base_image_dict_from_intervention(intervention)
    base_img_dict["metadata"] = {
        "is_frame": False,
        "is_extracted": True
    }

    img_ids = {}

    for i, path in enumerate(img_paths):
        shutil.copy(path.as_posix(), img_paths_intern[i].as_posix())
        _ = base_img_dict.copy()
        _["metadata"]["path"] = img_paths_intern[i]
        img = Image(**_)
        img_id = db_images.insert_one(img.to_dict()).inserted_id
        img_ids[str(i)] = img_id

    db_interventions.update_one({"_id": intervention.id}, {"$set": {"freezes": img_ids}})    

    return img_ids
    
