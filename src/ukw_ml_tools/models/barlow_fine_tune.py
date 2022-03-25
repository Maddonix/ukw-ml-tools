import torch
from torchvision.models.resnet import resnet50
from pathlib import Path
import torch.nn as nn
from pytorch_lightning import LightningModule
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from ..metrics.base import calculate_metrics
import torch.optim as optim

EPOCHS = 100
FREEZE_EXTRACTOR = True
LR_CLASSIFIER = 0.3
EPOCHS = 50
WEIGHT_DECAY = 1e-6
FINE_TUNED_CKPT_DIR = Path("fine_tuned_checkpoints")
METRICS_ON_STEP = False


class FineTunedBarlow(LightningModule):
    def __init__(self, checkpoint_path = None, labels = ["ASD"], epochs = EPOCHS, weight_decay = WEIGHT_DECAY, train = False):
        # super(FineTunedBarlow, self).__init__()
        super().__init__()
        self.save_hyperparameters()
        model = resnet50()
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        else: state_dict = {}


        self.labels = labels
        self.n_classes = len(labels)

        model.fc = nn.Linear(
            in_features = model.fc.in_features,
            out_features = self.n_classes,
        )
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        
        if "model" in state_dict:
            state_dict = state_dict["model"]
            state_dict = {key.replace("module.backbone.", ""): value for key, value in state_dict["model"].items() if "backbone" in key}
        state_dict = {key.replace("backbone.", ""): value for key, value in state_dict.items() if "backbone" in key}

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        # print(state_dict)
        if train:
            assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
        
        if FREEZE_EXTRACTOR:
            model.requires_grad_(False)
            model.fc.requires_grad_(True)

        classifier_parameters, model_parameters = [], []
        for name, param in model.named_parameters():
            if name in {'fc.weight', 'fc.bias'}:
                classifier_parameters.append(param)
            else:
                model_parameters.append(param)

        self.param_groups = [dict(params=classifier_parameters, lr=LR_CLASSIFIER)]
        # print(self.param_groups)
        # if args.weights == 'finetune':
            # param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
        self.model = model
        self.val_preds = []
        self.val_targets = []


        self.epochs = epochs
        self.weight_decay = weight_decay
        self.sigm = nn.Sigmoid()
        # self.criterion = nn.BCELoss()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([5]*len(self.labels)))


    def forward(self, x):
        # with torch.no_grad():
        #     representations = self.feature_extractor(x)
        #     representations = torch.flatten(representations, 1)

        x = self.model(x)
        return x # self.sigm(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train/loss", loss, on_step = METRICS_ON_STEP, on_epoch=True, prog_bar=True)

        # metrics=calculate_metrics(y_pred.cpu(), y.cpu(), threshold = 0.5)
        # for key, value in metrics.items():
        #     value = value.tolist()
        #     if isinstance(value, list):
        #         for i,_value in enumerate(value):
        #             name = "train/"+f"{key}/{self.labels[i]}"
        #             self.log(name, _value, on_epoch=True, on_step=METRICS_ON_STEP, prog_bar=True)
        #     else:
        #         name = "val/"+f"{key}"
        #         self.log(name, value, on_epoch=True, on_step=METRICS_ON_STEP, prog_bar=True)

        
        preds = np.array(self.sigm(y_pred).cpu() > 0.5, dtype=float)     

        return {"loss": loss, "preds": preds, "targets": y}


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        # metrics=calculate_metrics(y_pred.cpu(), y.cpu(), threshold = 0.5)
        # for key, value in metrics.items():
        #     value = value.tolist()
        #     if isinstance(value, list):
        #         for i, _value in enumerate(value):
        #             name = "val/"+f"{key}/{self.labels[i]}"
        #             self.log(name, _value, on_epoch=True, on_step=METRICS_ON_STEP, prog_bar=True)
        #     else:
        #         name = "val/"+f"{key}"
        #         self.log(name, value, on_epoch=True, on_step=METRICS_ON_STEP, prog_bar=True)

        preds = np.array(self.sigm(y_pred).cpu() > 0.5, dtype=float)
        self.val_preds.append(preds)
        self.val_targets.append(y.cpu().numpy())

        return {"loss": loss, "preds": preds, "targets": y}

    def validation_epoch_end(self, outputs):
        self.val_preds = np.concatenate([_ for _ in self.val_preds])
        self.val_targets = np.concatenate([_ for _ in self.val_targets])
        # print(self.val_preds.shape)
        # print(self.val_targets.shape)

        # val_prec = precision_score(y_true=self.val_targets, y_pred=self.val_preds, average=None, zero_division = 0)
        # val_rec = recall_score(y_true=self.val_targets, y_pred=self.val_preds, average=None, zero_division = 0)
        # val_f1 = f1_score(y_true=self.val_targets, y_pred=self.val_preds, average=None, zero_division = 0)
        # print(val_prec)
        # print(val_rec)
        # print(val_f1)

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
        # optimizer = optim.SGD(self.param_groups, 0.0003, momentum=0.9, weight_decay=self.weight_decay)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": scheduler,
        #     "monitor": "val/loss",
        # }
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=0.0015,
            weight_decay=self.weight_decay,
        )
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, verbose = True)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=0.00001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }