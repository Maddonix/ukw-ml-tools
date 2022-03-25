## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## Torchvision
import torchvision

# PyTorch Lightning
import pytorch_lightning as pl
from datetime import datetime as dt
from ukw_ml_tools.models.utils import instantiate
from torchmetrics.classification.accuracy import Accuracy

class SimCLR(pl.LightningModule):

    def __init__(self, hidden_dim = 128, lr = 3e-3, temperature = 0.07, weight_decay = 1e-4, max_epochs=500, num_classes = None):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(pretrained=False,num_classes=4*hidden_dim)  # Output of last linear layer
        # self.convnet = torchvision.models.regnet_x_1_6gf(pretrained = False, num_classes=4*hidden_dim)
        
        if num_classes:
            ll = num_classes
        else:
            ll = hidden_dim
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=False), #originally inplace=True
            nn.Linear(4*hidden_dim, ll)
        )

        if num_classes:
            self.setup_for_finetune(num_classes, lr)
        else:
            self.fine_tune = False

    def configure_optimizers(self):
        if self.fine_tune:
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )

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
        else:
            optimizer = optim.AdamW(self.parameters(),
                                    lr=self.hparams.lr,
                                    weight_decay=self.hparams.weight_decay)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=self.hparams.max_epochs,
                                                                eta_min=self.hparams.lr/50)
            return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode='train'):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode+'_loss', nll)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
        self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

        return nll

    def forward(self, x):
        return self.convnet(x)

    def training_step(self, batch, batch_idx):
        if self.fine_tune:
            inputs, labels = batch
            labels = labels.long()
            output = self.forward(inputs)

            preds = self.softmax(output).exp().argmax(dim=-1)
            loss = self.loss(output, labels)
            acc = self.train_accuracy(preds, labels)

            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

            return {"loss": loss, "preds": preds, "targets": labels}

        else:
            return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        if self.fine_tune:
            inputs, labels = batch
            labels = labels.long()
            output = self.forward(inputs)
            preds = self.softmax(output).exp().argmax(dim=-1)
            loss = self.criterion(output, labels)
            acc = self.val_accuracy(preds, labels)

            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

            return {"loss": loss, "preds": preds, "targets": labels}
            
        else:
            self.info_nce_loss(batch, mode='val')

    def setup_for_finetune(self, num_classes, lr=3e-3):
        self.convnet.fc[-1] = nn.Linear(self.convnet.fc[-1].in_features, num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.softmax = nn.LogSoftmax(dim=-1)

        self.hparams.lr = lr
        self.hparams.weight_decay = 0.00005
        self.fine_tune=True