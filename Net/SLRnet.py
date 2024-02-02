import torch
import torch.nn as nn
import numpy
import time
from config import *
from utils.tools import *
from Net.Rep import *
from Net.GCN import *

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
'''
# cut_hw = [[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[20,20],[20,20],[5,5],[5,5],[5,5]]
cut_hw = [[15, 15], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [20, 20], [20, 20], [0, 0],
          [0, 0], [0, 0]]

face = torch.zeros([batch, channel, 30, 30]).to(device)
lhand = torch.zeros([batch, channel, 40, 40]).to(device)
rhand = torch.zeros([batch, channel, 40, 40]).to(device)
'''
cut_hw = [[20, 20], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [30, 30], [30, 30], [0, 0],
          [0, 0], [0, 0]]
# cut_hw = [[15, 15], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [20, 20], [20, 20], [0, 0],
#           [0, 0], [0, 0]]

class SLRNet(nn.Module):
    def __init__(self, deploy, cfg):
        super(SLRNet, self).__init__()

        self.deploy = deploy

        self.vis_feature_dim = cfg.VIDEO_FEATURE_DIM
        self.num_class = cfg.VOC_SZIE
        self.image_size = cfg.IMAGE_SIZE
        self.heatmap_size = cfg.HEATMAP_SIZE
        self.video_len = cfg.MAX_VIDEOLEN
        self.num_joints = cfg.NUM_JOINTS
        self.in_channel = cfg.IN_CHANNEL

        self.scale = self.image_size[0] /self.heatmap_size[0]

        self.bodyNet = RepVGG(in_channel=self.in_channel, num_classes=self.vis_feature_dim,
                              num_joints=self.num_joints, num_blocks=[4, 6, 8, 2],
                              deploy=self.deploy, width_multiplier=[1, 1, 1, 1])

        self.faceNet = RepVGG_loc(deploy=self.deploy, num_classes=self.vis_feature_dim, num_blocks=[2, 2, 2, 1],
                                  in_channel=self.in_channel, width_multiplier=[0.5, 0.5, 0.5, 1])
        self.lhandNet = RepVGG_loc(deploy=self.deploy, num_classes=self.vis_feature_dim, num_blocks=[2, 4, 8, 1],
                                   in_channel=self.in_channel, width_multiplier=[0.5, 0.5, 0.5, 1])
        self.rhandNet = RepVGG_loc(deploy=self.deploy, num_classes=self.vis_feature_dim, num_blocks=[2, 4, 8, 1],
                                   in_channel=self.in_channel, width_multiplier=[0.5, 0.5, 0.5, 1])

        self.gnet = GraphNet(deploy=self.deploy, channels=[256, 256, 512, 512],
                             num_joints=4, video_len=self.video_len, in_channel=256)

        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(512*4, self.num_class)

    def get_loc_area(self, images, coordinate):
        # images = [batch, channel, image_h, image_w]
        # coordinate = [batch, num_joints, 2]
        _, channel, image_h, image_w = images.shape
        batch, num_joints, _= np.array(coordinate).shape
        local_area=[]
        for i in range(batch):
            image = images[i]
            point = coordinate[i]* self.scale
            # image=[channel, image_h, image_w]
            for j in range(num_joints):
                cut_h, cut_w = cut_hw[j]
                if cut_h == 0:
                    continue
                p = point[j]
                x, y = max(cut_h, int(p[0])), max(cut_w, int(p[1]))
                x, y = min(image_h - cut_h, x), min(image_w - cut_w, y)
                cut = image[:,y - cut_w:y + cut_w, x - cut_h:x + cut_h]
                local_area.append(cut)
        face = torch.zeros([batch, channel, 40, 40]).to(device)
        lhand = torch.zeros([batch, channel, 60, 60]).to(device)
        rhand = torch.zeros([batch, channel, 60, 60]).to(device)
        # face = torch.zeros([batch, channel, 30, 30]).to(device)
        # lhand = torch.zeros([batch, channel, 40, 40]).to(device)
        # rhand = torch.zeros([batch, channel, 40, 40]).to(device)
        for i in range(0, len(local_area), 3):
            face[i//3], lhand[i//3], rhand[i//3] = local_area[i:i+3]
        return face, lhand, rhand

    def forward(self, x):
        batch, video_len, channel, image_h, image_w = x.shape
        x = x.view(-1, channel, image_h, image_w)
        start_time = time.time()
        x_whole, heat_map = self.bodyNet(x)

        # target = [batch_size, num_joints, height, width]
        # preds = [batch*videolen, num_joints, 2]
        # x = [batch*videolen, chnnel, image_h, imgae_w]

        preds, _ = get_max_preds(heat_map.cpu().detach().numpy())
        face, lhand, rhand = self.get_loc_area(x, preds)

        print(time.time() - start_time,"test...")

        face = self.faceNet(face)
        lhand = self.lhandNet(lhand)
        rhand = self.rhandNet(rhand)
        x = torch.stack([x_whole, face, lhand, rhand], dim=1).to(device)
        x = x.view(batch, video_len, -1, self.vis_feature_dim).permute(0,3, 1, 2).contiguous()
        x = self.gnet(x)
        batch, channel, video_len, num_joints = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, video_len, -1)
        #x = self.dropout(x)
        x= self.out(x).log_softmax(2)
        return x, heat_map



def convert_slr(old_model, new_model):
    all_weight = {}
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


if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = CLR_config()
    x = torch.rand([1, 200, 1, 200, 200])
    net = SLRNet(True, config)
    start_time = time.time()
    x,h = net(x)

    print(time.time()-start_time, x.shape)