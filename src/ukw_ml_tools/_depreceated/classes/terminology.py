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
        self.concept_attributes = cfg["terminology_concept_attributes"]
        self.terminology_type = terminology_type
        
        self.terminology_df = pd.read_excel(self.terminology_path)


    def get_concepts(self):
        return [_ for _ in self.concept_attributes.keys()]

    def get_concept_query(self, concept): 
        assert concept in self.concept_attributes

        attributes = self.concept_attributes[concept]
        child_attributes = []
        for attribute in attributes:
            child_attributes.extend(self.get_all_child_attribute_ids(attribute))
        
        attributes.extend(child_attributes)

        query = {
            "$elemMatch": {"$in": attributes}
        }

        return query

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

    def get_all_child_attribute_ids(self, att_id, terminology_df = None):
        if not terminology_df:
            terminology_df = self.terminology_df
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
            tokens = r.json()
            tokens[FIELDNAME_TOKENS_VALUE] = [_[FIELDNAME_TOKENS_VALUE] for _ in tokens["tokens"] if not _["modifier"]]
            return tokens

        else:
            r = requests.post(
                self.url + URL_TEXT_TO_XML,
                json = {"text": text, "type": terminology_type},
                auth = self.auth
            )

            assert r.status_code == 200
            return r


    