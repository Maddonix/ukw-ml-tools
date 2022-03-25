from typing import Any
from typing import List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torchvision import models
from torch import sigmoid
from torch.hub import load_state_dict_from_url

class BinaryResnext(LightningModule):
    def __init__(self, num_classes, freeze_extractor, val_loss_weights=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_extractor = freeze_extractor

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        # self.model = models.resnext50_32x4d(pretrained=True)
        # self.model = models.resnet18(pretrained=True)
        
        self.model = models.efficientnet_b4(pretrained=True)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, 1)

        if val_loss_weights:
            val_loss_weights = torch.FloatTensor(val_loss_weights).to(0)
        if self.freeze_extractor:
            for param in self.model.parameters():
                param.requires_grad = False

        # num_ftrs = self.model.fc.in_features
        # FIXME -> two outputs are better because more is always better #CAPITALISM + Softmax
        # self.model.fc = nn.Linear(num_ftrs, 1)

        # loss function
        self.loss = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        print("Model Setup Complete!")

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        inputs, labels = batch
        # inputs=inputs.double()
        output = self.forward(inputs)
        loss = self.criterion(output, labels.unsqueeze(1).type_as(output))

        preds = sigmoid(output.detach()).squeeze(dim = -1)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        preds = preds.bool()
        # preds = preds.unsqueeze(1)

        return loss, preds, labels

    def training_step(self, batch: Any, batch_idx: int):
        inputs, labels = batch
        output = self.forward(inputs)
        loss = self.loss(output, labels.unsqueeze(1).type_as(output))
        preds = sigmoid(output.detach()).squeeze(dim = -1)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        preds = preds.bool()
        # preds = preds.unsqueeze(1)
        

        acc = self.train_accuracy(preds, labels)

        # log train metrics
        self.log("train/total_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": labels}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        use_optimizer = "adam"

        if use_optimizer == "adam":
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            from madgrad import MADGRAD

            optimizer = MADGRAD(
                params=self.parameters(),
                lr=self.hparams.lr,
                weight_decay=1e-12,
            )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }
