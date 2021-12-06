from .image import Image
from .db_images import DbImages
from typing import List
from .fieldnames import *
from ..models.utils import load_model
from ..datasets.utils import load_binary_image_classification_dataset
from torch.utils.data import DataLoader
from datetime import datetime as dt
import torch
from ..db.conversions import binary_classification_pred_to_db_value
from tqdm.auto import tqdm
from pathlib import Path


class AiPredictor:
    """[summary]
    """

    def __init__(self,
        ai_name,
        cfg,
        db_images: DbImages,
    ):
        self.ai_name = ai_name
        self.specs = cfg["models"][ai_name]
        self.cfg = cfg
        self.version = self.specs[FIELDNAME_AI_VERSION]
        self.image_scaling = self.specs[FIELDNAME_IMAGE_SCALING]
        self.base_paths_models = Path(cfg["base_path_models"])

    def load_model(self, specs = None, _eval: bool = True):
        self.model = None
        if not specs:
            specs = self.specs
        
        print("Loading in Prediction Mode!")
        self.model = load_model(self.ai_name, specs[FIELDNAME_AI_VERSION], _eval, self.base_paths_models)
        if self.model:
            return True

    def load_dataset(self, image_list: List[dict]):
        self.dataset = None
        paths = [_[FIELDNAME_IMAGE_PATH] for _ in image_list]
        ids = [str(_["_id"]) for _ in image_list]

        self.dataset = load_binary_image_classification_dataset(paths, ids, self.image_scaling, training = False)

        if self.dataset:
            return True

    def load_dataloader(
            self,
            dataset = None,
            batch_size: int = 100,
            num_workers = 0 ################ Change?
        ):
        if not dataset:
            dataset = self.dataset

        self.dataloader = None
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, sampler=None,
            batch_sampler=None, num_workers=num_workers, collate_fn=None,
            pin_memory=False, drop_last=True, timeout=0,
            worker_init_fn=None,prefetch_factor=2,
            persistent_workers=False
        )

        if self.dataloader:
            return True


    def predict_images(self, targets: List[str] = None, model = None, dataloader = None, specs = None):
        if not dataloader:
            dataloader = self.dataloader

        if not model:
            model = self.model

        if not specs:
            specs = self.specs

        if not targets:
            targets = [self.ai_name]
        
        creation_date = dt.now()
        print(targets)
        predicted_batches = []
        for batch_images, batch_labels in tqdm(dataloader):
            if torch.cuda.is_available():
                _pred = model(batch_images.to(0))#.to(0))
            else:
                _pred = model(batch_images)
            
            update = binary_classification_pred_to_db_value(
                _pred,
                batch_labels,
                version = specs[FIELDNAME_AI_VERSION],
                targets = targets,
                creation_date=creation_date)
            
            predicted_batches.append(update)
        
        return targets, predicted_batches

        