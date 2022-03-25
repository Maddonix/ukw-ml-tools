import torch
from torchvision import models
from pathlib import Path
import torch.nn as nn
from pytorch_lightning import LightningModule
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from ukw_ml_tools.metrics.base import calculate_metrics
import torch.optim as optim
from bson import ObjectId

EPOCHS = 100
FREEZE_EXTRACTOR = True
# LR_CLASSIFIER = 0.3
EPOCHS = 50
WEIGHT_DECAY = 0.001
FINE_TUNED_CKPT_DIR = Path("fine_tuned_checkpoints")
METRICS_ON_STEP = False

def predict_batch(batch, model):
    imgs, img_ids = batch
    img_ids = [ObjectId(_) for _ in img_ids]
    prediction_updates = {_id: [] for _id in img_ids}
    preds = model(imgs.to(0))
    preds = model.sigm(preds).to("cpu").numpy()
    pred_labels = np.argwhere(preds >0.2)
    for _pred in pred_labels:
        img_id = img_ids[_pred[0]]
        _value = preds[_pred[0],_pred[1]]
        _label = model.labels[_pred[1]]
        prediction_updates[img_id].append((_label, _value))
    prediction_updates = {key: value for key, value in prediction_updates.items() if value}

    return prediction_updates

def load_model(checkpoint_path):
    model = Net.load_from_checkpoint(checkpoint_path)
    model.to(0)
    model.eval()
    model.freeze()
    return model

class Net(LightningModule):
    def __init__(self, labels = ["ASD"], lr=6e-3, weight_decay = WEIGHT_DECAY, pos_weight = 2, model_type = "EfficientNetB4"):
        # super(FineTunedBarlow, self).__init__()
        super().__init__()
        self.save_hyperparameters()        

        model_type = "RegNetX800MF"
        self.model_type = model_type

        self.labels = labels
        self.n_classes = len(labels)
        self.val_preds = []
        self.val_targets = []
        self.pos_weight = 2 #pos_weight
    

        if model_type == "EfficientNetB4":
            self.model = models.efficientnet_b4(pretrained=True)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, len(labels))

        elif model_type == "RegNetX800MF":
            self.model = models.regnet_x_800mf(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, len(labels))

        print(self.model)

        self.weight_decay = WEIGHT_DECAY#weight_decay
        self.lr =0.03 #lr
        self.sigm = nn.Sigmoid()
        # self.criterion = nn.BCELoss()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.pos_weight]*len(self.labels)))

    

    def forward(self, x):
        x = self.model(x)
        return x # self.sigm(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train/loss", loss, on_step = METRICS_ON_STEP, on_epoch=True, prog_bar=True)
        
        preds = np.array(self.sigm(y_pred).cpu() > 0.5, dtype=float)     

        return {"loss": loss, "preds": preds, "targets": y}


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        preds = np.array(self.sigm(y_pred).cpu() > 0.5, dtype=float)
        self.val_preds.append(preds)
        self.val_targets.append(y.cpu().numpy())

        return {"loss": loss, "preds": preds, "targets": y}

    def validation_epoch_end(self, outputs):
        self.val_preds = np.concatenate([_ for _ in self.val_preds])
        self.val_targets = np.concatenate([_ for _ in self.val_targets])

        metrics=calculate_metrics(self.val_preds, self.val_targets, threshold = 0.5)
        for key, value in metrics.items():
            value = value.tolist()
            if isinstance(value, list):
                for i, _value in enumerate(value):
                    name = "val/"+f"{key}/{self.labels[i]}"
                    self.log(name, _value, on_epoch=True, on_step=METRICS_ON_STEP, prog_bar=False)
            else:
                name = "val/"+f"{key}"
                self.log(name, value, on_epoch=True, on_step=METRICS_ON_STEP, prog_bar=True)
        
        self.val_preds = []
        self.val_targets = []        


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.SGD(self.parameters(), self.lr, momentum=0.5, weight_decay=self.weight_decay)
        # optimizer = torch.optim.Adam(
        #     params=self.parameters(),
        #     lr=self.lr,
        #     weight_decay=self.weight_decay,
        # )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 20, verbose = True)

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": scheduler,
        #     "monitor": "val/loss",
        # }
        # optimizer = torch.optim.Adam(
        #     params=self.parameters(),
        #     lr=self.lr,
        #     weight_decay=self.weight_decay,
        # )
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, verbose = True)

        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     factor=0.5,
        #     patience=10,
        #     threshold=0.00001,
        #     threshold_mode="rel",
        #     cooldown=0,
        #     min_lr=0,
        #     eps=1e-08,
        #     verbose=True,
        # )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }