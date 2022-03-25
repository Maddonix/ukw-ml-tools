from ..mongodb.get_objects import get_intervention
from ..datasets.binary_image_classification_ds import BinaryImageClassificationDS
from torch.utils.data import DataLoader
import pandas as pd


def get_frame_df_dataloader(frame_df, ai_config, batch_size=200, num_workers=12, is_train=True, shuffle=True, swapaxes = None):
    dataset = BinaryImageClassificationDS(
        frame_df.path,
        frame_df.id,
        frame_df.crop,
        scaling=ai_config.ai_settings.image_scaling,
        training=is_train,
        swapaxes = swapaxes
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    return dataloader


def get_intervention_dataloader(video_key, ai_config, db_interventions, db_images, batch_size=200, num_workers=12, is_train=True, shuffle = True, swapaxes = None):
    intervention = get_intervention(video_key, db_interventions)
    frame_df = intervention.frame_df()
    cursor = db_images.find(
        {
            "_id": {"$in": frame_df.id.to_list()},
            "metadata.is_extracted": True
        },
        {"metadata": 1}
    )
    images = [{"id": str(_["_id"]), "path": _["metadata"]["path"], "crop": _["metadata"]["crop"]} for _ in cursor]
    frame_df = pd.DataFrame.from_records(images)

    dataloader = get_frame_df_dataloader(frame_df, ai_config, batch_size, num_workers, is_train, shuffle, swapaxes = swapaxes)

    return dataloader
