import re
import pandas as pd
import requests
from pathlib import Path
from .fieldnames import *

class Terminology:
    """[summary]
    """
    def __init__(self, cfg: dict, terminology_path: Path = None, terminology_type: int = 0):
        if not terminology_path:
            self.terminology_path = cfg["path_terminology"]
        else:
            self.terminology_path = terminology_path
        self.url = cfg["url_endobox_extreme"]+":"+ cfg["port_webserver"] + "/data"
        self.auth = (cfg["user_webserver"], cfg["password_webserver"])
        self.cfg = cfg
        self.terminology_type = terminology_type
        
        self.terminology_df = pd.read_excel(self.terminology_path)


    def process_raw_text(self, text: str) -> str:
        """Removes \\r \\n and strips flanking spaces from text.\
            Returns text in lower case.

        Args:
            text (str): text to process

        Returns:
            str: processed text
        """
        text = re.sub(r"\r\n", " ", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"  ", " ", text)
        text = text.strip() 

        return text.lower()

    def post_ontology(self):
        with open(self.terminology_path, "rb") as f:

            r = requests.post(
                self.url + URL_POST_ONTOLOGY,
                auth = self.auth,
                files = {"file": f},
                data = {"type": self.terminology_type})

            return r.status_code
            # assert r.status_code == 201

    def terminology_attribute_id_to_name(self, att_id):
        return self.terminology_df[self.terminology_df["ID"] == att_id]["Attribut"].to_list()[0]

    def get_all_child_attribute_ids(att_id, terminology_df):
        return terminology_df[terminology_df["ID"].str.startswith(att_id)].ID.to_list()


    def get_terminology_result(self, text:str, terminology_type: int = None, return_tokens: bool = True, preprocess_text: bool=True):
        if not terminology_type:
            terminology_type = self.terminology_type
        if preprocess_text:
            text = self.process_raw_text(text)

        if return_tokens:
            r = requests.post(
                self.url + URL_TEXT_TO_TOKEN,
                json = {"text": text, "type": terminology_type},
                auth = self.auth
            )
            assert r.status_code == 200
            return r.json()

        else:
            r = requests.post(
                self.url + URL_TEXT_TO_XML,
                json = {"text": text, "type": terminology_type},
                auth = self.auth
            )

            assert r.status_code == 200
            return r


    