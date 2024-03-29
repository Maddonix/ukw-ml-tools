from tqdm import tqdm
from bson import ObjectId
from ..labels.multiclass_prediction import (
    mc_batch_result_raw_to_pred_values,
    mc_base_pred_dict,
    mc_prediction_to_object,
)
from ..classes.intervention import Intervention
from ..mongodb.get_objects import get_images_by_id_list, get_intervention
from ..datasets.base import get_intervention_dataloader


def mc_prediction_and_upload(model, ai_config, dataloader, db_images):
    base_prediction_dict = mc_base_pred_dict(ai_config)

    for imgs, ids in tqdm(dataloader):
        preds = model.forward(imgs.to(0))
        values, raw = mc_batch_result_raw_to_pred_values(preds)
        for i, value in enumerate(values):
            _pred = mc_prediction_to_object(value, raw[i], base_prediction_dict)
            db_images.update_one(
                {"_id": ObjectId(ids[i])},
                {"$set": {f"predictions.{ai_config.name}": _pred.dict()}},
            )


def mc_to_binary_labels(name, intervention: Intervention, db_images):
    frame_df = intervention.frame_df()
    frames = get_images_by_id_list(frame_df.id.to_list(), db_images)

    for frame in frames:
        if name in frame.predictions:
            predictions = frame.predictions[name].to_binary_predictions()
        _update = {
            f"predictions.{prediction.name}": prediction.dict()
            for prediction in predictions
        }
        db_images.update_one({"_id": frame.id}, {"$set": _update})


def predict_full_video(
    video_key,
    name, model,
    s_threshold,
    s_weight,
    s_rel_conv_len,
    ai_config,
    db_interventions,
    db_images
):

    intervention = get_intervention(video_key, db_interventions)
    dataloader = get_intervention_dataloader(
        video_key,
        ai_config,
        db_interventions,
        db_images,
        num_workers=13,
        batch_size=100
    )

    mc_prediction_and_upload(model, ai_config, dataloader, db_images)
    mc_to_binary_labels(name, intervention, db_images)
