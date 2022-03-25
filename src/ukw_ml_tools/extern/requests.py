import requests
from ..classes.extern import ExternAnnotatedVideo, ExternVideoFlankAnnotation
from urllib3.exceptions import InsecureRequestWarning
from typing import List
from ..classes.extern import VideoExtern

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
URL_POST_ONTOLOGY = "/PostOntology"
URL_TEXT_TO_TOKEN = "/PostTerminologyTokens"
URL_TEXT_TO_XML = "/PostTerminologyXml"
FIELDNAME_TOKENS = "tokens"
FIELDNAME_TOKENS_PATHO = "patho"
FIELDNAME_TOKENS_REPORT = "report"
FIELDNAME_TOKENS_VALUE = "value"


def get_extern_annotations(url, auth):
    r = requests.get(f"{url}/GetVideosWithAnnotations", auth=auth, verify=False)
    assert r.status_code == 200
    r = [ExternAnnotatedVideo(**_) for _ in r.json()]

    return r


def get_extern_video_annotation(video_key, url, auth):
    r = requests.get(url+"/GetAnnotationsByVideoName/"+video_key, auth=auth, verify=False)
    assert r.status_code == 200
    annotations = [ExternVideoFlankAnnotation(**_) for _ in r.json()]

    return annotations

def get_smartie_data(url, auth):
    r = requests.get(url+"/GetSmartieVideoData", auth = auth, verify=False)
    assert r.status_code == 200

    r = r.json()
    return r

def extract_video_intervals(_json, url, auth):
    r = requests.post(url+"/PostVideoToImageIntervals", auth=auth, json=_json, verify=False)
    return r


def get_extern_interventions(url, auth) -> List[dict]:
    r = requests.get(f"{url}/GetVideosExtern", auth=auth, verify=False)
    assert r.status_code == 200
    videos = [VideoExtern(**_) for _ in r.json()]

    return videos


def post_ontology(url, auth, file, terminology_type):
    r = requests.post(
        url + URL_POST_ONTOLOGY,
        auth=auth,
        files={"file": file},
        data={"type": terminology_type},
        verify=False
    )

    return r.status_code


def text_to_tokens(url, auth, text, terminology_type):
    r = requests.post(
        url + URL_TEXT_TO_TOKEN,
        json={"text": text, "type": terminology_type},
        auth=auth,
        verify=False
    )
    assert r.status_code == 200
    tokens = r.json()
    return tokens


def text_to_xml(url, auth, text, terminology_type):
    r = requests.post(
        url + URL_TEXT_TO_XML,
        json={"text": text, "type": terminology_type},
        auth=auth,
        verify=False
    )
    return r
