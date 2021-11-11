from pathlib import Path
import requests
from typing import List

from ukw_ml_tools.classes.fieldnames import FIELDNAME_VIDEO_KEY, FIELDNAME_VIDEO_PATH
from ukw_ml_tools.classes.utils import get_intervention_db_template


class DbExtern:
    """[summary]
    """

    def __init__(self, cfg: dict):
        self.url = cfg["url_endobox_extreme"] + ":" + cfg["port_webserver"] + "/data"
        self.fields_extern = cfg["db_extern_fields"]
        self.user = cfg["user_webserver"]
        self._p = cfg["password_webserver"]
        self._auth = (self.user, self._p)
        self.lookup_extern_fieldtypes = {
            "int": int,
            "str": str,
        }

    def convert_intervention_external_to_internal(self, intervention):
        valid = self.validate_extern_intervention(intervention)
        if not valid:
            raise Exception

        intervention_internal = get_intervention_db_template(intervention_dict = intervention)

        for field, value in intervention.items():
            if field in self.fields_extern:
                if "map_value" in self.fields_extern[field]:
                    value = self.fields_extern[field]["map_value"][value]
                internal_field_name = self.fields_extern[field]["internal_field"]
                intervention_internal[internal_field_name] = value

        video_key = Path(intervention_internal[FIELDNAME_VIDEO_PATH]).name
        intervention_internal[FIELDNAME_VIDEO_KEY] = video_key

        return intervention_internal


    def get_extern_interventions(self) -> List[dict]:
        r = requests.get(f"{self.url}/GetVideosExtern", auth=self._auth)
        assert r.status_code == 200

        return r.json()


    # Validation
    def validate_field_extern(self, field, value) -> bool:
        if field not in self.fields_extern:
            return True
        
        field_config = self.fields_extern[field]
        target_type = field_config["type"]
        target_type = self.lookup_extern_fieldtypes[target_type]
        
        if not field_config["required"] and value == None:
            return True
        else:    
            _check_value = isinstance(value, target_type)
            return _check_value


    def validate_extern_intervention(self, intervention: dict) -> bool:
        '''Only validates fields mentioned in cfg["db_extern_fields"]'''
        valid = True

        for field, value in intervention.items():
            field_valid = self.validate_field_extern(field, value)
            if not field_valid:
                valid = False
        
        return valid
