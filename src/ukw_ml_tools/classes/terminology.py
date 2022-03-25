import re
import pandas as pd
import json
from ukw_ml_tools.extern.requests import post_ontology, text_to_tokens, text_to_xml
from typing import List, Any, Optional
from pydantic import BaseModel, Field

URL_POST_ONTOLOGY = "/PostOntology"
URL_TEXT_TO_TOKEN = "/PostTerminologyTokens"
URL_TEXT_TO_XML = "/PostTerminologyXml"
FIELDNAME_TOKENS = "tokens"
FIELDNAME_TOKENS_PATHO = "patho"
FIELDNAME_TOKENS_REPORT = "report"
FIELDNAME_TOKENS_VALUE = "value"


class TerminologyToken(BaseModel):
    group_id: int = Field(alias="groupId")
    covered_text: Optional[str] = Field(alias="coveredText")
    modifier: Optional[str]
    type: str
    related: Any

    def __hash__(self):
        return hash(repr(self))


class TerminologyResult(BaseModel):
    tokens: List[TerminologyToken]
    values_without_relation: List[Any] = Field(alias="valuesWithoutRelation")
    value: List[str]

    def __hash__(self):
        return hash(repr(self))


class Terminology:
    """[summary]
    """

    def __init__(self, cfg, terminology_type: int = 0):
        url, auth = cfg.get_extern_tuple()
        self.url = url
        self.auth = auth
        self.terminology_type = terminology_type

        self.terminology_dir = cfg.base_paths.terminology
        self.terminology_path = self.terminology_dir.joinpath(f"{terminology_type}.xlsx")
        self.terminology_df = pd.read_excel(self.terminology_path)
        with open(self.terminology_dir.joinpath("concepts.json"), "r") as f:
            self.concepts = json.load(f)

    def __hash__(self):
        return hash(repr(self))

    def get_concepts(self):
        return list(self.concepts.keys())

    def get_concept_query(self, concept):
        assert concept in self.concepts

        attributes = self.concepts[concept]
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
            r = post_ontology(self.url, self.auth, f, self.terminology_type)
        return r.status_code

    def terminology_attribute_id_to_name(self, att_id):
        return self.terminology_df[self.terminology_df["ID"] == att_id]["Attribut"].to_list()[0]

    def get_all_child_attribute_ids(self, att_id, terminology_df=None):
        if not terminology_df:
            terminology_df = self.terminology_df
        return terminology_df[terminology_df["ID"].str.startswith(att_id)].ID.to_list()

    def get_terminology_result(self, text: str, terminology_type: int = None, return_tokens: bool = True, preprocess_text: bool = True):
        if not terminology_type:
            terminology_type = self.terminology_type
        if preprocess_text:
            text = self.process_raw_text(text)

        if return_tokens:
            result = text_to_tokens(self.url, self.auth, text, terminology_type)
            result[FIELDNAME_TOKENS_VALUE] = [_[FIELDNAME_TOKENS_VALUE] for _ in result["tokens"] if not _["modifier"]]
            result = TerminologyResult(**result)
        else:
            result = text_to_xml(self.url, self.auth, text, terminology_type)
        return result
