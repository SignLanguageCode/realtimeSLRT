import torch
import torch.nn as nn
import numpy
import time
import random
from config import *
from Net.BackBone import *
#USE_CUDA = torch.cuda.is_available()
#device = torch.device("cuda" if USE_CUDA else "cpu")


class SeqDecode(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(SeqDecode, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer =n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_size*2, input_size)
        self.dropout = nn.Dropout(0.5, inplace=True)
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                nn.init.orthogonal_(m.weight_ih_l0.data)
                nn.init.orthogonal_(m.weight_ih_l0.data)
                m.bias_ih_l0.data.fill_(0)
                m.bias_hh_l0.data.fill_(0)

    def forward(self, x, h, c, encoder_output):
        self.rnn.flatten_parameters()
        # hidden = [num_layers=2 * num_directions=1, batch=2, hidden_size=4096]
        # encoder_output = [batch=2, step=24, hidden_size=4096]
        # x = [batch]
        x = x.unsqueeze(1)
        # x embedded =[batch, step=1, hidden_size=4096]
        embed = self.dropout(self.embedding(x))

        x, (h, c) = self.rnn(embed, (h, c))
        x = self.out(torch.cat((x, embed), dim=2)).squeeze(1)
        return x, h, c

class SLRNet(nn.Module):
    def __init__(self, deploy, cfg):
        super(SLRNet, self).__init__()
        self.vis_feature_dim = cfg.VIDEO_FEATURE_DIM
        self.num_class = cfg.VOC_SZIE
        self.max_seq = cfg.MAX_SENTENCELEN
        self.RepVGG = create_RepVGG_A0(deploy, num_classes=self.vis_feature_dim)
        self.rnn = nn.LSTM(input_size=self.vis_feature_dim, hidden_size=self.vis_feature_dim//2, num_layers=1,
                           batch_first=True, bidirectional=False)
        self.SeqDecode = SeqDecode(self.num_class, self.vis_feature_dim//2, n_layers=1)

    def forward(self, x):
        batch, video_len, channel, image_h, image_w = x.shape
        x = x.view(-1, channel, image_h, image_w)
        x = self.RepVGG(x)
        x = x.view(batch, video_len, -1)
        x, (h,c) = self.rnn(x)

        w = torch.full(size=(batch,), fill_value=(self.num_class-2), dtype=torch.long).to(device)
        outputs = torch.zeros(self.max_seq, batch, self.num_class).to(device)
        for t in range(self.max_seq):
            w, h, c = self.SeqDecode(w, h, c, x)
            outputs[t] = w
            w = w.max(1)[1].detach()
        return outputs.permute(1, 0, 2)


if __name__ == '__main__':
    config = CLR_config()
    x = torch.rand([2,80,1,200,200])
    net = SLRNet(True, config)
    start_time = time.time()

    x = net(x)
    print(x.shape)

    print(time.time()-start_time, x.shape)