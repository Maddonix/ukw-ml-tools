import requests
from ..classes.extern import ExternAnnotatedVideo, ExternVideoFlankAnnotation
from urllib3.exceptions import InsecureRequestWarning
from typing import List
from ..classes.extern import VideoExtern

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


def get_extern_annotations(url, auth):
    r = requests.get(f"{url}/GetVideosWithAnnotations", auth = auth, verify = False)
    assert r.status_code == 200
    r = [ExternAnnotatedVideo(**_) for _ in r.json()]

    return r

def get_extern_video_annotation(video_key, url, auth):
    r = requests.get(url+"/GetAnnotationsByVideoName/"+video_key, auth = auth, verify = False)
    assert r.status_code == 200
    annotations = [ExternVideoFlankAnnotation(**_) for _ in r.json()]

    return annotations


def extract_video_intervals(_json, url, auth):
    r = requests.post(url+"/PostVideoToImageIntervals", auth = auth,json=_json, verify=False)
    return r

def get_extern_interventions(url, auth) -> List[dict]:
    r = requests.get(f"{url}/GetVideosExtern", auth=auth, verify = False)
    assert r.status_code == 200
    videos = [VideoExtern(**_) for _ in r.json()]

    return videos