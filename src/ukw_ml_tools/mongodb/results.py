def get_intervention_labels(intervention_id, db_images, select = "predictions"):
    """
    Returns one record for each label (_id, name, value, intervention_id, framenumber)
    """
    aggregation = [
        {"$match": {"intervention_id": intervention_id}},
        {
            "$project": {
                select: {"$objectToArray": f"${select}"},
                "video_key": "$video_key",
                "intervention_id": "$intervention_id",
                "frame_number": "$metadata.frame_number"
            }
        },
        {
            "$unwind":  f"${select}"
        },
        {
            "$project": {
                "intervention_id": "$intervention_id",
                "video_key": "$video_key",
                "frame_number": "$frame_number",
                "_type": select,
                "name": f"${select}.v.name",
                "value": f"${select}.v.value",
                "raw": f"${select}.v.raw"
            }
        }
    ]

    r = db_images.aggregate(aggregation)
    records = [_ for _ in r]
    return records


