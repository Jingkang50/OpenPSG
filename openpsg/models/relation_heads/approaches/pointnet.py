# ---------------------------------------------------------------
# pointnet.py
# Set-up time: 2020/10/6 23:24
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STN2d(nn.Module):
    def __init__(self, init_c=32):
        super(STN2d, self).__init__()
        self.init_c = init_c
        self.conv1 = torch.nn.Conv1d(2, init_c, 1)
        self.conv2 = torch.nn.Conv1d(init_c, init_c * 2, 1)
        self.conv3 = torch.nn.Conv1d(init_c * 2, init_c * 4, 1)
        self.fc1 = nn.Linear(init_c * 4, init_c * 2)
        self.fc2 = nn.Linear(init_c * 2, init_c)
        self.fc3 = nn.Linear(init_c, 4)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(init_c)
        self.bn2 = nn.BatchNorm1d(init_c * 2)
        self.bn3 = nn.BatchNorm1d(init_c * 4)
        self.bn4 = nn.BatchNorm1d(init_c * 2)
        self.bn5 = nn.BatchNorm1d(init_c)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batchsize, -1)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(2).view(1, -1).repeat(batchsize, 1).to(x)
        x = x + iden
        x = x.view(-1, 2, 2)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k).view(1, -1).repeat(batchsize, 1).to(x)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    def __init__(self,
                 init_c=32,
                 out_c=1024,
                 global_feat=True,
                 feature_transform=False):
        super(PointNetFeat, self).__init__()
        self.stn = STN2d()
        self.init_c = init_c
        self.out_c = out_c
        self.conv1 = torch.nn.Conv1d(2, init_c, 1)
        self.conv2 = torch.nn.Conv1d(init_c, init_c * 2, 1)
        self.conv3 = torch.nn.Conv1d(init_c * 2, init_c * 4, 1)
        self.bn1 = nn.BatchNorm1d(init_c)
        self.bn2 = nn.BatchNorm1d(init_c * 2)
        self.bn3 = nn.BatchNorm1d(init_c * 4)
        self.global_feat = global_feat
        if self.global_feat and out_c != init_c * 4:
            self.fc = nn.Linear(init_c * 4, out_c)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batchsize, -1)
        if self.global_feat:
            if hasattr(self, 'fc'):
                x = self.fc(x)
            return x, trans, trans_feat
        else:
            x = x.view(batchsize, -1, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetFeat(global_feat=True,
                                 feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetFeat(global_feat=False,
                                 feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        x = x / torch.max(torch.sqrt(torch.sum(x**2, dim=1, keepdim=True)),
                          dim=-1,
                          keepdim=True)[0]
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :].to(trans)
    # bug: pytorch-1.4: do not support torch.norm on GPU directly
    diff = torch.bmm(trans, trans.transpose(2, 1)) - I
    loss = torch.mean(torch.norm(diff.cpu(), dim=(1, 2)).to(trans))
    #loss = torch.mean(torch.norm(, dim=(1, 2)))
    return loss
