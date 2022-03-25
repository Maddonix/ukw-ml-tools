from tqdm import tqdm
from ..mongodb.update import update_terminology_result


def apply_terminology(_type, terminology, db_interventions):
    """Only on Videos!

    Args:
        _type ([type]): [description]
    """
    if _type == "report":
        field = "intervention_report_text.text"
    elif "histo":
        field = "intervention_histo_text.text"
    else:
        raise Exception

    interventions = [_ for _ in db_interventions.aggregate([
        {
            "$match": {
                "video_key": {"$exists": True, "$nin": ["", None]},
                field: {"$exists": True, "$nin": [None]},
            }
        },
        {
            "$project":  {field: 1, "video_key": 1}
        }
    ])]
    _, __ = field.split(".")
    for intervention in tqdm(interventions):
        video_key = intervention["video_key"]
        result = terminology.get_terminology_result(intervention[_][__])
        update_terminology_result(result, video_key, _type, db_interventions)
