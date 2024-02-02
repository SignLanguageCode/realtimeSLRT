import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint
from torchsummary import summary
from config import *


class ConvBn3x1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,1), stride=1, padding=(1, 0),deploy=False):
        super(ConvBn3x1, self).__init__()
        assert kernel_size==(3,1)
        self.deploy = deploy
        if deploy:
            self.conv_reparam = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=(3,1),
                                          stride=stride, padding=padding, bias=True)
        else:
            self.convbn = nn.Sequential()
            self.convbn.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                     kernel_size=kernel_size, stride=stride, padding=padding,
                                                     bias=False))
            self.convbn.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    def forward(self, x):
        if self.deploy:
            return self.conv_reparam(x)
        else:
            return self.convbn(x)

    def _pad_3x1_to_3x3_tensor(self, kernel3x1):
        if kernel3x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel3x1, [0, 2, 0, 0])

    def _fuse_bn_tensor3x1(self):
        kernel = self.convbn.conv.weight
        kernel = self._pad_3x1_to_3x3_tensor(kernel)
        running_mean = self.convbn.bn.running_mean
        running_var = self.convbn.bn.running_var
        gamma = self.convbn.bn.weight
        beta = self.convbn.bn.bias
        eps = self.convbn.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return (kernel * t)[:,:,:,:1], beta - running_mean * gamma / std


class ConvBn(nn.Module):
    def __init__(self, deploy, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBn, self).__init__()
        self.deploy = deploy
        if deploy:
            self.conv_reparam = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, bias=True)
        else:
            self.convbn = nn.Sequential()
            self.convbn.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                     kernel_size=kernel_size, stride=stride, padding=padding,
                                                     bias=False))
            self.convbn.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    def forward(self, x):
        if self.deploy:
            return self.conv_reparam(x)
        else:
            return self.convbn(x)

    def _fuse_bn_tensor(self):
        kernel = self.convbn.conv.weight
        running_mean = self.convbn.bn.running_mean
        running_var = self.convbn.bn.running_var
        gamma = self.convbn.bn.weight
        beta = self.convbn.bn.bias
        eps = self.convbn.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, deploy=False):
        super(RepBlock, self).__init__()
        self.deploy = deploy
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = ConvBn(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding,deploy=deploy)
            self.rbr_1x1 = ConvBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                  stride=stride, padding=0, deploy=deploy)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernelx, biasx = self.rbr_dense._fuse_bn_tensor()
        kernel1x1, bias1x1 = self.rbr_1x1._fuse_bn_tensor()
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        if self.kernel_size == 3:
            return kernelx + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, biasx + bias1x1 + biasid
        else:
            return kernelx + kernel1x1 + kernelid, biasx + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        assert isinstance(branch, nn.BatchNorm2d)
        if not hasattr(self, 'id_tensor'):
            if self.kernel_size == 3:
                input_dim = self.in_channels
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            else:
                input_dim = self.in_channels
                kernel_value = np.zeros((self.in_channels, input_dim, 1, 1), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 0, 0] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
        kernel = self.id_tensor
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu(), bias.detach().cpu()


class RepVGG(nn.Module):
    def __init__(self, num_blocks, in_channel=1, num_classes=512, num_joints=14, deploy=False, width_multiplier=[1,1,1,1]):
        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.in_channel = in_channel
        self.num_joints = num_joints

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepBlock(in_channels=self.in_channel, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy)
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
            blocks.append(RepBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1,  deploy=self.deploy))
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
    def __init__(self, num_blocks, in_channel=1, num_classes=512, deploy=False, width_multiplier=[1,1,1,1]):
        super(RepVGG_loc, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.in_channel = in_channel
        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepBlock(in_channels=self.in_channel, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy)
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
            blocks.append(RepBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        #print(x.shape)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def convert(old_model, new_model):
    all_weight = {}
    for name, module in old_model.named_modules():
        if hasattr(module, '_fuse_bn_tensor'):
            kernel, bias = module._fuse_bn_tensor()
            all_weight[name + 'conv_reparam.weight'] = kernel
            all_weight[name + 'conv_reparam.bias'] = bias
    new_model.load_state_dict(all_weight)
    return new_model

def convert_rep(old_model, new_model):
    all_weight= {}
    for name, module in old_model.named_modules():

        if hasattr(module, 'repvgg_convert'):
            kernel, bias = module.repvgg_convert()
            all_weight[name + '.rbr_reparam.weight'] = kernel
            all_weight[name + '.rbr_reparam.bias'] = bias
        elif isinstance(module, torch.nn.Linear):
            all_weight[name + '.weight'] = module.weight.detach().cpu()
            all_weight[name + '.bias'] = module.bias.detach().cpu()
        elif "deconv" in name and "." in name:
            all_weight[name + '.weight'] = module.weight.detach().cpu()
            all_weight[name + '.bias'] = module.bias.detach().cpu()
    new_model.load_state_dict(all_weight)
    return new_model

def test_ConvBn():
    x = torch.rand([1, 3, 4, 4])
    net_bef = ConvBn(deploy=False, in_channels=3, out_channels=5, kernel_size=1, stride=1, padding=1)
    net_aft = ConvBn(deploy=True, in_channels=3, out_channels=5, kernel_size=1, stride=1, padding=1)
    net_aft = convert(net_bef, net_aft)
    net_bef.eval()
    net_aft.eval()
    y = net_bef(x)
    y1 = net_aft(x)
    print(y - y1)
    print(torch.sum(torch.pow(y1 - y, 2)))

def test_Rep():
    x = torch.rand([100, 3,100, 100])

    net_bef = RepBlock(deploy=False, in_channels=3, out_channels=256, kernel_size=1, stride=1, padding=0)
    net_aft = RepBlock(deploy=True, in_channels=3, out_channels=256, kernel_size=1, stride=1, padding=0)
    net_aft = convert_rep(net_bef, net_aft)
    net_bef.eval()
    net_aft.eval()
    y = net_bef(x)
    y1 = net_aft(x)
    print(y - y1)
    print(torch.sum(torch.pow(y1 - y, 2)))


def test_RepVGG():
    x = torch.rand([10, 3,100, 100])
    net_bef = RepVGG(deploy=False, in_channel=3, num_blocks=[1,2,3,1], num_classes=512, width_multiplier=[1,1,1,1])
    net_aft = RepVGG(deploy=True, in_channel=3, num_blocks=[1,2,3,1], num_classes=512, width_multiplier=[1,1,1,1])
    net_aft = convert_rep(net_bef, net_aft)
    net_bef.eval()
    net_aft.eval()
    y,h = net_bef(x)
    y1,h1 = net_aft(x)
    print(y - y1)
    print(torch.sum(torch.pow(y1 - y, 2)))
    print(torch.sum(torch.pow(h - h1, 2)))

def test_RepVGGLoc():
    x = torch.rand(10, 1, 40,40)
    net = RepVGG_loc(deploy=False, num_classes=256, num_blocks=[2, 4, 8, 1],  in_channel=1, width_multiplier=[0.5, 0.5, 0.5, 1])
    x = net(x)
    print(x.shape)

if __name__ == '__main__':
    test_ConvBn()

