import os
from ..mongodb.get_objects import get_intervention
import requests
from ..extern.requests import extract_video_intervals

# Base
def get_frame_name(i, prefix, suffix):
    name = "{:0>7d}".format(i)
    if prefix:
        name = prefix+name
    if suffix:
        name = name+suffix
    return name

def get_frame_dir(video_key, base_dir_frames):
    frame_dir = base_dir_frames.joinpath(video_key)
    if not frame_dir.exists():
        os.mkdir(frame_dir)
        os.system(f"chgrp -R lux_tomcat {frame_dir.as_posix()}")
    return frame_dir


def extract_range_by_video_path(
    url, auth, video_path,
    target_dir, intervals, suffix, prefix=None,
    ):
    paths = {}
    skip_intervals = []
    if not target_dir.exists():
        os.mkdir(target_dir)
        os.system(f"chgrp -R lux_tomcat {target_dir.as_posix()}")
        

    for n_interval, interval in enumerate(intervals):
        _paths = {}
        for i in range(interval["start"], interval["end"]+1):
            _paths[i] = target_dir.joinpath(get_frame_name(i, prefix, suffix))
        all_exist = False
        for i, _path in _paths.items():
            if not _path.exists():
                all_exist = False
                break
        if all_exist:
            skip_intervals.append(n_interval)
        else:
            paths.update(_paths)
    
    intervals = [_ for i, _ in enumerate(intervals) if not i in skip_intervals]
    _r = {
        "videoPath": video_path.as_posix(),
        "intervals": intervals,
        "imageFolderPath": target_dir.as_posix()
    }
    if prefix:
        _r["prefix"]=prefix
    if suffix:
        _r["extension"]=suffix

    r = extract_video_intervals(_r, url, auth)
    return paths
    

# Pipelines
def extract_frames_for_video_segmentation(video_key, url, auth, db_interventions, base_dir_frames, suffix = ".jpg", prefix = None):
    intervention = get_intervention(video_key, db_interventions)
    video_path = intervention.metadata.path
    extract_dict = {
            "url": url,
            "auth": auth,
            "target_dir": get_frame_dir(video_key, base_dir_frames),
            "intervals": [],
            "prefix": prefix,
            "suffix": suffix,
            "video_path": video_path

        }

    for name, video_segmentation in intervention.video_segments_annotation.items():
        _intervals = [{"start": flank.start, "end": flank.stop} for flank in video_segmentation.value]
        extract_dict["intervals"].extend(_intervals)

    paths = extract_range_by_video_path(**extract_dict)
    return paths