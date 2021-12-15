
from a_ukw_ml_tools.mongodb.test_data import get_test_data_ids
from a_ukw_ml_tools.mongodb.train_data import get_train_data_query, image_list_to_train_data
from a_ukw_ml_tools.mongodb.base import filter_images_by_frame_diff

def create_new_train_data(name, ai_config, db_images, db_test_data):

    choices = ai_config.choices
    label_type = ai_config.label_type
    exclude_origins = ai_config.sampler_settings.exclude_origins
    test_sets = ai_config.sampler_settings.test_sets
    exclude_intervention_ids = []
    for test_set in test_sets:
        exclude_intervention_ids.extend(get_test_data_ids(test_set, db_test_data))

    aggregation = [
        get_train_data_query(name, exclude_intervention_ids, exclude_origins),
        # {
        #     "$project": {
        #         "paths": "$metadata.path",
        #         "labels": f"$annotations.{name}.value",
        #         "origins": "$origin",
        #         "intervention_ids": "$intervention_id",
        #         "choices": f"$annotations.{name}.choices"
        #     }
        # }
    ]

    
    cursor = db_images.aggregate(aggregation)
    images = filter_images_by_frame_diff(cursor, ai_config.sampler_settings.min_frame_diff)
    train_data = image_list_to_train_data(name, label_type, choices, images)

    return train_data