#!/usr/bin/env python3
# @file      predict_confidences.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import click
from pytorch_lightning import Trainer
import torch
import torch.nn.functional as F

import sys

sys.path.append("src")
import mos4d.datasets.datasets as datasets
import mos4d.models.models as models
import os


@click.command()
### Add your options here
@click.option(
    "--weights",
    "-w",
    type=str,
    help="path to checkpoint file (.ckpt) to do inference.",
    required=True,
)
@click.option(
    "--sequence",
    "-seq",
    type=int,
    help="Run inference on a specific sequence. Otherwise, test split from config is used.",
    default=None,
    multiple=True,
)
@click.option(
    "--dt",
    "-dt",
    type=float,
    help="Desired temporal resolution of predictions.",
    default=None,
)
@click.option("--poses", "-poses", type=str, default=None, help="Specify which poses to use.")
@click.option(
    "--transform",
    "-transform",
    type=bool,
    default=None,
    help="Transform point clouds to common viewpoint.",
)
def main(weights, sequence, dt, poses, transform):
    cfg = torch.load(weights)["hyper_parameters"]

    if poses:
        cfg["DATA"]["POSES"] = poses

    if transform != None:
        cfg["DATA"]["TRANSFORM"] = transform
        if not transform:
            cfg["DATA"]["POSES"] = "no_poses"

    #! seq以参数的形式传入，不走config
    if sequence:
        cfg["DATA"]["SPLIT"]["TEST"] = list(sequence)

    if dt:
        cfg["MODEL"]["DELTA_T_PREDICTION"] = dt

    #! 自己设置推理的batch_size，不走config
    cfg["TRAIN"]["BATCH_SIZE"] = 6

    #! 设置推理GPU数
    cfg["TRAIN"]["N_GPUS"] = 1

    # Load data and model
    cfg["DATA"]["SPLIT"]["TRAIN"] = cfg["DATA"]["SPLIT"]["TEST"]
    cfg["DATA"]["SPLIT"]["VAL"] = cfg["DATA"]["SPLIT"]["TEST"]
    data = datasets.KittiSequentialModule(cfg)
    data.setup()

    model = models.MOSNet.load_from_checkpoint(weights, hparams=cfg)

    # Setup trainer
    trainer = Trainer(gpus=cfg["TRAIN"]["N_GPUS"], logger=False)

    # Infer!
    trainer.predict(model, data.test_dataloader())


if __name__ == "__main__":
    os.environ['SWITCH'] = 'run'  # run debug
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    os.environ['DATA'] = '/share/sgb/semantic_kitti/dataset/sequences'
    os.environ['GROUND'] = "/share/sgb/kitti-ground"
    main()
