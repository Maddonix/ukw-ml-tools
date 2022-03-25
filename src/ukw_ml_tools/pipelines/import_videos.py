from ..mongodb.data_import import get_new_extern_interventions
from ..classes.intervention import Intervention
from ..mongodb.update import insert_new_intervention
from typing import List, Tuple
from pymongo.collection import Collection


def import_new_extern_interventions(url: str, auth: Tuple[str], db_interventions) -> List:
    interventions, failed, duplicates = get_new_extern_interventions(url, auth, db_interventions)
    video_keys = [_ for _ in db_interventions.distinct("video_key")]
    inserted_ids = []
    for intervention in interventions:
        intervention = Intervention(**intervention.to_dict())
        assert not intervention.id
        assert intervention.video_key not in video_keys

        r = insert_new_intervention(intervention, db_interventions)
        inserted_ids.append(r.inserted_id)

    return inserted_ids
