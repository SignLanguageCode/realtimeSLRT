import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from Net.Rep import *
from scipy.sparse import coo_matrix

#USE_CUDA = torch.cuda.is_available()
#device = torch.device("cuda" if USE_CUDA else "cpu")


def build_graph4(num_joints=4, video_len=85, undirection=True):
    assert num_joints == 4
    edge = []
    for i in range(video_len):
        edge.append([i * 4, i * 4 + 1])
        edge.append([i * 4, i * 4 + 2])
        edge.append([i * 4, i * 4 + 3])
        #edge.append([i * 4+2, i * 4 + 3])
    for i in range(video_len - 1):
        for j in range(num_joints):
            edge.append([i * 4 + j, (i + 1) * 4 + j])
    edge_rever = []
    if undirection:
        for e in edge:
            edge_rever.append([e[1], e[0]])
    edge.extend(edge_rever)
    edge = np.array(edge)
    # row =np.array( edge[:,0])
    # col = np.array( edge[:,1])
    # data = np.ones_like(row)
    # print(row, col)
    # matrix = coo_matrix((data,(row,col)), shape=[video_len*num_joints,video_len*num_joints]).toarray()
    return edge.transpose(1, 0)


class GCN_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3, 1), deploy=False):
        super(GCN_Block, self).__init__()
        self.conv = RepBlock(in_channel, out_channel, kernel_size=1, stride=1, padding=0, deploy=deploy)
        self.gcn1 = GCNConv(out_channel, out_channel)
        self.gcn2 = GraphConv(out_channel, out_channel)
        self.temporal = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=(1, 0)),
            nn.BatchNorm2d(out_channel)
        )
        if in_channel != out_channel:
            self.res = RepBlock(in_channel, out_channel, kernel_size=1, stride=1, deploy=deploy)
        else:
            self.res = lambda x: x
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge):
        edge = edge.to(x.device)
        resx = self.res(x)
        x = self.conv(x)
        batch, channel, video_len, num_joint = x.shape
        x = x.view(batch, channel, -1).permute(0, 2, 1).contiguous()
        x = self.gcn1(x, edge)
        x = x.permute(0, 2, 1).view(batch, channel, video_len, num_joint).contiguous()
        x = self.temporal(x)
        x = x.view(batch, channel, -1).permute(0, 2, 1).contiguous()
        x = self.gcn2(x, edge)
        x = x.permute(0, 2, 1).view(batch, channel, video_len, num_joint).contiguous()
        return self.relu(x + resx)

class GraphNet(nn.Module):
    def __init__(self, in_channel=256, channels=[256, 256, 512, 512], num_joints=4, video_len=85, deploy=False):
        super(GraphNet, self).__init__()
        self.num_joints = num_joints
        self.video_len = video_len
        self.deploy = deploy
        edge = build_graph4(num_joints=self.num_joints, video_len=self.video_len)
        self.edge = torch.tensor(edge, dtype=torch.long)
        self.gcn1 = GCN_Block(in_channel, channels[0], kernel_size=(3, 1), deploy=deploy)
        self.gcn2 = GCN_Block(channels[0], channels[1], kernel_size=(3, 1), deploy=deploy)
        self.gcn3 = GCN_Block(channels[1], channels[2], kernel_size=(3, 1), deploy=deploy)
        self.gcn4 = GCN_Block(channels[2], channels[3], kernel_size=(3, 1), deploy=deploy)

    def forward(self, x):

        np.save("dev_layer0.npy", x.detach().cpu().numpy())
        x = self.gcn1(x, self.edge)
        np.save("dev_layer1.npy",x.detach().cpu().numpy())
        x = self.gcn2(x, self.edge)
        np.save("dev_layer2.npy", x.detach().cpu().numpy())
        x = self.gcn3(x, self.edge)
        np.save("dev_layer3.npy", x.detach().cpu().numpy())
        x = self.gcn4(x, self.edge)
        np.save("dev_layer4.npy", x.detach().cpu().numpy())
        return x


if __name__ == '__main__':
    num_joints = 4
    video_len = 3
    edge = build_graph4(num_joints=num_joints, video_len=video_len, undirection=True)
    print(edge)
    #print(edge.shape)
    x = torch.rand([1, 256, video_len, num_joints])
    strat = time.time()
    gnet = GraphNet(in_channel=256)
    x = gnet(x, torch.tensor(edge, dtype=torch.long))
    print(x.shape, time.time()-strat)
