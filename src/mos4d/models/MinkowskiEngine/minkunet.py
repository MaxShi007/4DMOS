# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch
import torch.nn as nn
from torch.optim import SGD

import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from mos4d.models.MinkowskiEngine.resnet import ResNetBase


class MinkUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    def m_space_n_time(self, m, n):
        return [m, m, m, n]

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution( # 1->8
            in_channels,
            self.inplanes,
            kernel_size=self.m_space_n_time(5, 1),
            dimension=D,
        )

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution( # 8->8
            self.inplanes,
            self.inplanes,
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0]) # 8->8

        self.conv2p2s2 = ME.MinkowskiConvolution( # 8->8
            self.inplanes,
            self.inplanes,
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1]) # 8->16

        self.conv3p4s2 = ME.MinkowskiConvolution( # 16->16
            self.inplanes,
            self.inplanes,
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2]) #16->32

        self.conv4p8s2 = ME.MinkowskiConvolution( # 32->32
            self.inplanes,
            self.inplanes,
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3]) #32->64

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose( #64->64
            self.inplanes,
            self.PLANES[4],
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4]) #96->64
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose( #64->32
            self.inplanes,
            self.PLANES[5],
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5]) #48->32
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose( #32-16
            self.inplanes,
            self.PLANES[6],
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6]) #24->16
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(# 16-8
            self.inplanes,
            self.PLANES[7],
            kernel_size=self.m_space_n_time(2, 1),
            stride=self.m_space_n_time(2, 1),
            dimension=D,
        )
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7]) #16->16

        self.final = ME.MinkowskiConvolution( #16->3
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D,
        )
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.conv0p1s1(x)# 1-8
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1) #8-8 /2
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out) #8-8

        out = self.conv2p2s2(out_b1p2) #8-8 /2
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out) #8-16

        out = self.conv3p4s2(out_b2p4) #16-16 /2
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out) #16-32

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8) #32-32 /2
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out) #32-64

        # tensor_stride=8
        out = self.convtr4p16s2(out) #64-64 *2
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8) #64-96
        out = self.block5(out) #96-64

        # tensor_stride=4
        out = self.convtr5p8s2(out)#64-32 *2
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4) #32-48
        out = self.block6(out)#48-32

        # tensor_stride=2
        out = self.convtr6p4s2(out) #32-16 *2
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2) #16-24
        out = self.block7(out)#24-16

        # tensor_stride=1
        out = self.convtr7p2s2(out) #16-8 *2
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1) #8-16
        out = self.block8(out)# 16-8

        return self.final(out)#8-3


class MinkUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet18(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet50(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet101(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class MinkUNet14A(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet14B(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet14C(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class MinkUNet14D(MinkUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet18A(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet18B(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet18D(MinkUNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet34A(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


if __name__ == "__main__":
    from tests.python.common import data_loader

    # loss and network
    criterion = nn.CrossEntropyLoss()
    net = MinkUNet14A(in_channels=3, out_channels=5, D=2)
    print(net)

    # a data loader must return a tuple of coords, features, and labels.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    optimizer = SGD(net.parameters(), lr=1e-2)

    for i in range(10):
        optimizer.zero_grad()

        # Get new data
        coords, feat, label = data_loader(is_classification=False)
        input = ME.SparseTensor(feat, coordinates=coords, device=device)
        label = label.to(device)

        # Forward
        output = net(input)

        # Loss
        loss = criterion(output.F, label)
        print("Iteration: ", i, ", Loss: ", loss.item())

        # Gradient
        loss.backward()
        optimizer.step()

    # Saving and loading a network
    torch.save(net.state_dict(), "test.pth")
    net.load_state_dict(torch.load("test.pth"))
