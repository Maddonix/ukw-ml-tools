from pymongo.collection import Collection
import warnings


def get_count_query(feature_name):
    return {
        "$facet": {
            "data": [
                {"$group": {"_id": f"${feature_name}", "count": {"$sum": 1}}},
                {"$project": {"origin": "$_id", "count": "$count"}},
            ]
        }
    }


def get_predictions_without_annotations_query(prelabel_type: str, label: str):
    return {
        "$match": {
            f"labels.predictions.{prelabel_type}.labels.{label}": True,
            f"labels.annotation.{label}": {"$exists": False},
            "in_progress": False,
        }
    }


def get_images_to_prelabel(prelabel_type: str, version: float, db_collection: str, batchsize: int = 0):
    agg = [
        {
            '$match': {
                '$or': [
                    {
                        f'labels.predictions.{prelabel_type}': {
                            '$exists': False
                        }
                    }, {
                        f'labels.predictions.{prelabel_type}.version': {
                            '$lt': version
                        }
                    }
                ]
            }
        }
    ]
    if batchsize > 0:
        agg.append({
            "$limit": batchsize
        })

    return db_collection.aggregate(agg)


def get_number_of_images_in_progress_query():
    return [{"$match": {"in_progress": True}}, {"$count": "count"}]


def get_intervention_for_image_id(
    image_id, db_images: Collection, db_interventions: Collection
):
    r = db_images.find_one({"_id": image_id})

    if r:
        intervention_id = r["intervention_id"]
        intervention = db_interventions.find_one({"_id": intervention_id})

        return intervention

    else:
        warnings.warn(f"No image found for id {image_id}")


intervention_has_image_paths = {"$match": {"image_paths.1": {"$exists": True}}}


def get_intervention_text_match_query(
    keyword_list, intervention_type: str = "Koloskopie"
):
    text_query = {
        "image_paths.1": {"$exists": True},
        "intervention_type": intervention_type,
        "$text": {
            "$search": " ".join(keyword_list),
            "$language": "de",
            "$caseSensitive": False,
        },
    }

    agg = [
        {"$match": text_query},
        {
            "$lookup": {
                "from": "images",
                "localField": "image_ids",
                "foreignField": "_id",
                "as": "image_objects",
            }
        },
        {
            "$project": {
                "report": True,
                "image_objects": True,
                "intervention_date": True,
                "age": True,
                "gender": True,
                "origin": True,
                "intervention_type": True,
            }
        },
    ]

    return agg
