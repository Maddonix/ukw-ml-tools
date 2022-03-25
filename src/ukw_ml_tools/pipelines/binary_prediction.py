from tqdm import tqdm
from bson import ObjectId
from ..labels.binary_prediction import (
    b_base_pred_dict,
    b_batch_result_raw_to_pred_values,
    b_prediction_to_object
)
from ..classes.intervention import Intervention
from ..mongodb.get_objects import get_images_by_id_list, get_intervention
from ..datasets.base import get_intervention_dataloader

def b_prediction_and_upload(model, ai_config, dataloader, db_images):
    base_prediction_dict = b_base_pred_dict(ai_config)
    for imgs, ids in tqdm(dataloader):
        preds = model.forward(imgs.to(0))
        values, raw = b_batch_result_raw_to_pred_values(preds)
        for i, value in enumerate(values):
            _pred = b_prediction_to_object(value, raw[i], base_prediction_dict)
            db_images.update_one(
                {"_id": ObjectId(ids[i])},
                {"$set": {f"predictions.{ai_config.name}": _pred.dict()}},
            )

