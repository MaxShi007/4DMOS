from pytorch_lightning import Trainer
import MinkowskiEngine as ME
import torch
from icecream import ic
import numpy as np
data0 = [
    [2, 0, 2.1, 0, 0],
    [0, 1, 1.4, 3, 0],
    [0, 0, 4.0, 0, 0]
]

data1=[
    [1.0, 3.7, 0, 0, 0],
    [0, 1, 1.4, 0, 8],
    [0, 0, 4.0, 9.1, 0],
    [0, 0, 4.0, 9.1, 0]
]


def to_sparse_coo(data):
    # An intuitive way to extract coordinates and features
    coords, feats = [], []
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            if val != 0:
                coords.append([i, j])
                feats.append([val])
    return torch.IntTensor(coords), torch.FloatTensor(feats)

coords0,feats0=to_sparse_coo(data0)
ic(coords0,feats0)
coords1,feats1=to_sparse_coo(data1)
ic(coords1,feats1)
coords,feats=ME.utils.sparse_collate([coords0,coords1],[feats0,feats1])
ic(coords,feats,coords.shape,feats.shape)

sinput = ME.SparseTensor(
    features=feats, # Convert to a tensor
    coordinates=coords,  # coordinates must be defined in a integer grid. If the scale
    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE  # when used with continuous coordinates, average features in the same coordinate
)
ic(sinput,sinput.coordinate_manager)

A=ME.SparseTensor(coordinates=coords0,features=feats0)
B=ME.SparseTensor(coordinates=coords1,features=feats1,coordinate_manager=A.coordinate_manager)

C=B/A
print(C)