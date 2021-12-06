from typing import Any
from typing import List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import sigmoid
from torchmetrics.classification.accuracy import Accuracy
from torchvision import models


class GoogleNet(LightningModule):
    def __init__(self, num_classes, freeze_extractor, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_extractor = freeze_extractor

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        # self.model = models.resnext50_32x4d(pretrained=True)
        self.model = models.googlenet(pretrained=False)

        if self.freeze_extractor:
            print("Transfer learning with a fixed ConvNet feature extractor")
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print("Transfer learning with a full ConvNet finetuning")

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        self.model.aux1.fc2 = nn.Linear(self.model.aux1.fc2.in_features, self.num_classes)
        self.model.aux2.fc2 = nn.Linear(self.model.aux2.fc2.in_features, self.num_classes)

        # loss function
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([5]))
        self.loss1 = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([3]))
        self.loss2 = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.5]))
        self.discount = 0.3

        self.criterion = nn.BCEWithLogitsLoss()  # nn.NLLLoss

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()


    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        inputs, labels = batch
        output = self.forward(inputs)

        preds = sigmoid(output).squeeze(dim = -1)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0

        loss = self.criterion(output.squeeze(dim = -1), torch.tensor(labels).type_as(output))

        return loss, preds, labels

    def training_step(self, batch: Any, batch_idx: int):
        self.model.train()
        inputs, labels = batch
        output = self.forward(inputs)
        o = output.logits
        o1 = output.aux_logits1
        o2 = output.aux_logits2
        
        preds = sigmoid(o).squeeze(dim = -1)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0

        preds1 = sigmoid(o1).squeeze(dim = -1)
        preds1[preds1 >= 0.5] = 1
        preds1[preds1 < 0.5] = 0

        preds2 = sigmoid(o2).squeeze(dim = -1)
        preds2[preds2 >= 0.5] = 1
        preds2[preds2 < 0.5] = 0
        
        loss = self.loss(o.squeeze(dim = -1), torch.tensor(labels).type_as(o))
        loss1 = self.loss1(o1.squeeze(dim = -1), torch.tensor(labels).type_as(o1))
        loss2 = self.loss2(o2.squeeze(dim = -1), torch.tensor(labels).type_as(o2))
        total_loss = self.discount * loss + self.discount * loss1 + self.discount * loss2 


        acc = self.train_accuracy(preds, labels)

        # log train metrics
        self.log("train/total_loss", total_loss, on_step = False, on_epoch = True, prog_bar = True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": total_loss, "preds": preds, "targets": labels}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        self.model.eval()
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        self.model.eval()
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
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=3,
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
