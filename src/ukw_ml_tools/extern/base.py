from datetime import datetime as dt

def path_to_timestamp(path, time_format, origin=None):
    name = path.name
    if origin:
        name = name.replace(origin, "")
    try:
        time_str = name.split("_") [-3:-1]
        time_str = "_".join(time_str)
        timestamp = dt.strptime(time_str, time_format)
    except:
        time_str = name.split("_") [-4:-2]
        time_str = "_".join(time_str)
        timestamp = dt.strptime(time_str, time_format)    

    return timestamp

def video_extern_exists_intern(video_id, db_interventions) -> bool:
    _ = db_interventions.find_one({"id_extern": video_id})
    if _: return True
    else: return False

def video_extern_get_intern_id(video_id, db_interventions):
    _ = db_interventions.find_one({"id_extern": video_id})
    if _: return _["_id"]

