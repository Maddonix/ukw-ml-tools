from datetime import datetime as dt
from ..db.crud import get_intervention_text_match_query
from pymongo.collection import Collection

text_match_version = 0.0
text_match_keywords = {
    "polyp": [
        "polyp",
        "polypen",
        "polyposis",
        "polypös",
        "sessil",
        "sessiler",
        "gestielt",
        "gestiehlt",
        "adenom",
        "adenomatös",
        "nice",
        "paris",
        "polypensprosse",
        "polypektomie",
    ],
    "diverticulum": [
        "divertikel",
        "divertikulös",
        "divertikulitis",
        "diverticulös",
        "diverticulitis",
    ],
    "bbps": [
        "verschmutzt",
        "eingeschränkte sicht",
        "verschutzung",
        "sicht",
        "stuhl",
        "stuhlverschmutzung",
        "stuhlreste",
    ],
}


def get_text_prediction_template(_creation):
    _targets = list(text_match_keywords.keys())

    return {
        "version": text_match_version,
        "targets": _targets,
        "creation_date": _creation,
        "labels": {_: False for _ in _targets},
    }


def match_texts(db_collection: Collection):
    _creation = dt.now()

    matches = {_: [] for _ in text_match_keywords.keys()}
    all_matches = []
    for target, keyword_list in text_match_keywords.items():
        agg = get_intervention_text_match_query(keyword_list)
        r = db_collection.aggregate(agg)
        for intervention in r:
            matches[target].extend(
                _img["_id"] for _img in intervention["image_objects"]
            )
        all_matches.extend(matches[target])

    all_matches = list(set(all_matches))
    all_matches = {_: get_text_prediction_template(_creation) for _ in all_matches}

    for target, _ids in matches.items():
        for _id in _ids:
            all_matches[_id]["labels"][target] = True

    return all_matches
