import os
import yaml
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import MinkowskiEngine as ME
from pytorch_lightning import loggers 


import sys
sys.path.append("src") 
from mos4d.models.MinkowskiEngine.customminkunet import CustomMinkUNet
from mos4d.models.loss import MOSLoss
from mos4d.models.metrics import ClassificationMetrics

#* dataset
class DummyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        dataset="/share/sgb/semantic_kitti/dataset/sequences/04/velodyne"
        self.filenames=[os.path.join(dataset,file) for file in os.listdir(dataset)]
        # print(self.filenames)
        self.voxel_size = 0.1
        
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        filename = self.filenames[i]
        pcd = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        # print(pcd.shape,pcd)
        pcd_xyz=pcd[:,:3]
        pcd_features=pcd[:,3:]
        quantized_coords, feats = ME.utils.sparse_quantize(
            np.array(pcd_xyz, dtype=np.float32),
            np.array(pcd_features, dtype=np.float32),
            quantization_size=self.voxel_size,
        )
        random_labels = torch.zeros(len(feats))
        return {
            "coordinates": quantized_coords,
            "features": feats,
            "labels": random_labels,
        }

def collate_fn(batch):
    # start=time.time()
    # coordinates = []
    # features = []
    # labels = []
    # for b in batch:
    #     coordinate=b["coordinates"]
    #     feature=b["features"]
    #     label=b["labels"]
    #     coordinates.append(coordinate)
    #     features.append(feature)
    #     labels.append(label)
    # print(time.time()-start)


    start=time.time()
    coordinates_batch,features_batch,labels_batch=ME.utils.sparse_collate([b["coordinates"] for b in batch],[b["features"] for b in batch],[b["labels"] for b in batch],dtype=torch.float32)

    # print(time.time()-start)
    
    return {'coordinates': coordinates_batch, 'features': features_batch, 'labels': labels_batch}






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
        self.model=model
        self.lr=lr
        self.weight_decay=weight_decay

    def train_dataloader(self):
        return DataLoader(
            DummyDataset(),
            batch_size=10,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            timeout=0,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            DummyDataset(),
            batch_size=10,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            timeout=0,
            num_workers=4,
        )

    def forward(self, x):
        return self.model(x)
    
    def get_loss(self,pred,labels):
        return self.criterion(pred.F, labels.long())

    def training_step(self, batch, batch_idx):
        print(len(batch))
        stensor = ME.SparseTensor(
            coordinates=batch["coordinates"], features=batch["features"]
        )
        pred=self(stensor)
        # loss=self.criterion(pred.F, batch["labels"].long())
        loss=self.get_loss(pred,batch["labels"])
        self.log("train_loss", loss.item(),on_step=True,on_epoch=True, sync_dist=True)

        # Must clear cache at regular interval
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        stensor = ME.SparseTensor(
            coordinates=batch["coordinates"], features=batch["features"]
        )
        pred=self(stensor)
        # loss=self.criterion(pred.F, batch["labels"].long())
        return self.criterion(self(stensor).F, batch["labels"].long())

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)




if __name__=="__main__":
    # tb_logger = loggers.TensorBoardLogger(
    #     log_dir, name=cfg["EXPERIMENT"]["ID"], default_hp_metric=False
    # )
    pl_module = MinkowskiSegmentationModule(DummyNetwork(1, 20, D=3))
    trainer = Trainer(max_epochs=100, gpus=2, accelerator="ddp",logger=True,log_every_n_steps=2)
    trainer.fit(pl_module)

    # train_set=DummyDataset()
    # train_loader=DataLoader(train_set, batch_size=4,collate_fn=collate_fn)
    # for batch in train_loader:
    #     coordinates, features, labels=batch
