import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
import MinkowskiEngine as ME

from mos4d.models.MinkowskiEngine.customminkunet import CustomMinkUNet
from mos4d.models.loss import MOSLoss
from mos4d.models.metrics import ClassificationMetrics

#* dataset
class DummyDataset(Dataset):

    ...

    def __getitem__(self, i):
        filename = self.filenames[i]
        pcd = o3d.io.read_point_cloud(filename)
        quantized_coords, feats = ME.utils.sparse_quantize(
            np.array(pcd.points, dtype=np.float32),
            np.array(pcd.colors, dtype=np.float32),
            quantization_size=self.voxel_size,
        )
        random_labels = torch.zeros(len(feats))
        return {
            "coordinates": quantized_coords,
            "features": feats,
            "labels": random_labels,
        }




#* network
class DummyNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, 32, 3, dimension=D),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(32, 64, 3, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolutionTranspose(64, 32, 3, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(32, out_channels, kernel_size=1, dimension=D),
        )

    def forward(self, x):
        return self.net(x)




#*lightning module
class MinkowskiSegmentationModule(LightningModule):
    r"""
    Segmentation Module for MinkowskiEngine.
    """

    def __init__(
        self,
        model,
        optimizer_name="SGD",
        lr=1e-3,
        weight_decay=1e-5,
        voxel_size=0.05,
        batch_size=12,
        val_batch_size=6,
        train_num_workers=4,
        val_num_workers=2,
    ):
        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)

        self.criterion = nn.CrossEntropyLoss()

    def train_dataloader(self):
        return DataLoader(
            DummyDataset("train", voxel_size=self.voxel_size),
            batch_size=self.batch_size,
            collate_fn=minkowski_collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            DummyDataset("val", voxel_size=self.voxel_size),
            batch_size=self.val_batch_size,
            collate_fn=minkowski_collate_fn,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        stensor = ME.SparseTensor(
            coordinates=batch["coordinates"], features=batch["features"]
        )
        # Must clear cache at regular interval
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
        return self.criterion(self(stensor).F, batch["labels"].long())

    def validation_step(self, batch, batch_idx):
        stensor = ME.SparseTensor(
            coordinates=batch["coordinates"], features=batch["features"]
        )
        return self.criterion(self(stensor).F, batch["labels"].long())

    def configure_optimizers(self):
        return SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)




if __name__=="__main__":
    pl_module = MinkowskiSegmentationModule(DummyNetwork(3, 20, D=3), lr=args.lr)
    trainer = Trainer(max_epochs=args.max_epochs, gpus=num_devices, accelerator="ddp")
    trainer.fit(pl_module)