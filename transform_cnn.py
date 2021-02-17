# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

import torch

from e2cnn import gspaces
from e2cnn import nn as nn_e2




class SteerCNN(nn.Module):

    def __init__(self, n_classes=6):
        super(SteerCNN, self).__init__()

        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.Rot2dOnR2(N=4)

        # the input image is a scalar field, corresponding to the trivial representation
        input_type  = nn_e2.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = input_type
        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = nn_e2.FieldType(self.r2_act, 24* [self.r2_act.regular_repr])
        self.block1 = nn_e2.SequentialModule(
            nn_e2.R2Conv(input_type, out_type, kernel_size=7, padding=3, bias=False),
            nn_e2.InnerBatchNorm(out_type),
            nn_e2.ReLU(out_type, inplace=True)
        )

        self.pool1 = nn_e2.PointwiseAvgPool(out_type, 4)

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        #out_type = nn_e2.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
        self.block2 = nn_e2.SequentialModule(
            nn_e2.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False),
            nn_e2.InnerBatchNorm(out_type),
            nn_e2.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn_e2.SequentialModule(
            nn_e2.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0),
            nn_e2.PointwiseAvgPool(out_type, 4),
            nn_e2.GroupPooling(out_type)
        )
        # PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=7)


        # number of output channels
        c = 24*13*13 #self.gpool.out_type.size

        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, n_classes),
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn_e2.GeometricTensor(input, self.input_type)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)


        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor
       
        # classify with the final fully connected layers)
        x = x.view(-1, 24*13*13)
        x = self.fully_net(x)


        return x



class VA(nn.Module):
    """The layer for transforming the skeleton to the observed viewpoints"""
    def __init__(self,num_classes = 60,steer=1):
        super(VA, self).__init__()
        self.num_classes = num_classes
        self.steer = steer

        if self.steer:
            self.steer_model = SteerCNN().to('cuda')
            self.classifier = models.resnet50(pretrained=True)
            num_ftrs = self.classifier.fc.in_features
            self.classifier.fc = nn.Linear(num_ftrs, self.num_classes)



        else:

            self.conv1 = nn.Conv2d(3, 128, kernel_size=5, stride=2,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(128)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(128, 128, kernel_size=5, stride=2,
                                   bias=False)
            self.bn2 = nn.BatchNorm2d(128)
            self.relu2 = nn.ReLU(inplace=True)
            self.avepool = nn.MaxPool2d(7)

            self.fc = nn.Linear(6272, 6)

            self.classifier = models.resnet50(pretrained=True)
            self.init_weight()

    def forward(self, x1, maxmin):
        #x1 32x3x224x224

        if self.steer:
            trans = self.steer_model(x1)

        else:
            x = self.conv1(x1)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.avepool(x)

            x = x.view(x.size(0), -1)
            trans = self.fc(x) # x: 32x6272->32x6 (trans)


        temp1 = trans.cpu()
        x = _transform(x1, trans, maxmin)

        temp = x.cpu()
        x = self.classifier(x) # 32x60 actionclass
        return x, temp.data.numpy(), temp1.data.numpy()

    def init_weight(self):
        for layer in [self.conv1, self.conv2]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                if 'bias' in name:
                    param.data.zero_()
        for layer in [self.bn1, self.bn2]:
            layer.weight.data.fill_(1)
            layer.bias.data.fill_(0)
            layer.momentum = 0.99
            layer.eps = 1e-3

        self.fc.bias.data.zero_()
        self.fc.weight.data.zero_()

        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_ftrs, self.num_classes)

# get transformation matrix
def _trans_rot(trans, rot):
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = Variable(rot.data.new(rot.size()[:1] + (1,)).zero_())
    ones = Variable(rot.data.new(rot.size()[:1] + (1,)).fill_(1))

    r1 = torch.stack((ones, zeros, zeros),dim=-1)
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)
    rx = torch.cat((r1, rx2, rx3), dim = 1)

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)

    rot = rz.matmul(ry).matmul(rx)


    rt1 = torch.stack((ones, zeros, zeros, trans[:,0:1]), dim = -1)
    rt2 = torch.stack((zeros, ones, zeros, trans[:,1:2]), dim = -1)
    rt3 = torch.stack((zeros, zeros, ones, trans[:,2:3]), dim = -1)
    trans = torch.cat((rt1, rt2, rt3), dim = 1)

    return trans, rot

# transform skeleton
def _transform(x, mat, maxmin):
    #x 32x3x224x224 #mat32x6
    rot = mat[:,0:3]
    trans = mat[:,3:6]

    x = x.contiguous().view(-1, x.size()[1] , x.size()[2] * x.size()[3])

    max_val, min_val = maxmin[:,0], maxmin[:,1]
    max_val, min_val = max_val.contiguous().view(-1,1), min_val.contiguous().view(-1,1)
    max_val, min_val = max_val.repeat(1,3), min_val.repeat(1,3)
    trans, rot = _trans_rot(trans, rot) # get transformation matrix
    # trans->32X 3x4 rot->32x3x3

    x1 = torch.matmul(rot,x) # x: 32x3x50176 rot:32x3x3 -> x1: 32x3x50176;
    min_val1 = torch.cat((min_val, Variable(min_val.data.new(min_val.size()[0], 1).fill_(1))), dim=-1)
    min_val1 = min_val1.unsqueeze(-1)
    min_val1 = torch.matmul(trans, min_val1)

    min_val = torch.div( torch.add(torch.matmul(rot, min_val1).squeeze(-1), - min_val), torch.add(max_val, - min_val))

    min_val = min_val.mul_(255)
    x = torch.add(x1, min_val.unsqueeze(-1)) #x1: 32x3x50176 -> #x 32x3x224x224

    x = x.contiguous().view(-1,3, 224,224)

    return x

# transform only skeleton
def _transform_skel(x, mat, maxmin):
    #x 32x3x224x224 #mat32x6
    # x 32x3x25x78
    # rgb_ske = np.transpose(rgb_ske, [1, 0, 2])
    # rgb_ske = np.transpose(rgb_ske, [2, 1, 0])

    rot = mat[:,0:3]
    trans = mat[:,3:6]

    # x = x.contiguous().view(-1, x.size()[1] , x.size()[2] * x.size()[3])

    max_val, min_val = maxmin[:,0], maxmin[:,1]
    max_val, min_val = max_val.contiguous().view(-1,1), min_val.contiguous().view(-1,1)
    max_val, min_val = max_val.repeat(1,3), min_val.repeat(1,3)
    trans, rot = _trans_rot(trans, rot) # get transformation matrix
    # trans->32X 3x4 rot->32x3x3

    x1 = torch.matmul(rot,x) # x: 32x3x50176 rot:32x3x3 -> x1: 32x3x50176;
    min_val1 = torch.cat((min_val, Variable(min_val.data.new(min_val.size()[0], 1).fill_(1))), dim=-1)
    min_val1 = min_val1.unsqueeze(-1)
    min_val1 = torch.matmul(trans, min_val1)

    min_val = torch.div( torch.add(torch.matmul(rot, min_val1).squeeze(-1), - min_val), torch.add(max_val, - min_val))

    min_val = min_val.mul_(255)
    x = torch.add(x1, min_val.unsqueeze(-1)) #x1: 32x3x50176 -> #x 32x3x224x224



    return x



