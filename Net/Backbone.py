import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
from torchsummary import summary
from utils.RepVGGBlock import *
from utils.Densenet import *

 

class RepVGG(nn.Module):
    def __init__(self, num_blocks, in_channel=1, num_classes=512, num_joints=14, deploy=False,
                 width_multiplier=[1,1,1,1], override_groups_map=None):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.in_channel = in_channel
        self.num_joints = num_joints
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=self.in_channel, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveMaxPool2d(output_size=2)
        self.linear = nn.Linear(int(512*2*2 * width_multiplier[3]), num_classes)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(int(128 * width_multiplier[1]), 64, 3, stride=2, padding=1,output_padding=1),
            nn.Conv2d(64, self.num_joints, 1)
        )
        self.deconv2 = nn.Sequential(
            # nn.Conv2d(5 * growthRate + nChannels, 32, 3, stride=2, padding=1),
            nn.ConvTranspose2d(int(256 * width_multiplier[2]), 128, 3, stride=2, padding=1, output_padding=0),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(64, num_joints, 1)
        )

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        heat_map = self.deconv1(x)
        x = self.stage3(x)
        heat_map += self.deconv2(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x, heat_map


class RepVGG_loc(nn.Module):
    def __init__(self, num_blocks, in_channel=1, num_classes=512, deploy=False,
                 width_multiplier=[1,1,1,1], override_groups_map=None):
        super(RepVGG_loc, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.in_channel = in_channel
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=self.in_channel, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveMaxPool2d(output_size=1)
        self.linear = nn.Linear(int(512*1*1 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def create_RepVGG_A0(deploy=False, num_classes=250):
    return RepVGG(num_blocks=[4, 6, 8, 2], num_classes=num_classes,
                  width_multiplier=[1,1,1,1], override_groups_map=None, deploy=deploy, num_joints=14)


 
def test_RepVGG():
    model = RepVGG(num_blocks=[2, 4, 14, 1], in_channel=3, num_classes=512,
                   num_joints=14, deploy=False, width_multiplier=[1, 1, 1, 1], override_groups_map=None)
    x = torch.ones([160, 3, 200, 200])
    x,h  = model(x)
    print(x.shape, h.shape)


def test_RepLoc():
    x = torch.rand([80,3,30,30])
    model = RepVGG_loc(num_blocks=[2, 4, 8, 1],
                       in_channel=3,
                       num_classes=512,
                       deploy=False,
                       width_multiplier=[0.5, 0.5, 0.5, 1])
    x = model(x)
    print(x.shape)


if __name__ == '__main__':
    test_RepVGG()