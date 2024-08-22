from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class buffer2keyframe(torch.nn.Module):
    def __init__(self, opt):
        super(buffer2keyframe, self).__init__()
        self._1x1conv = nn.Conv2d(opt.buffer_size, 1, 1)
        self.bufferSize = opt.buffer_size
        self.buffers = list()

        fill_weights(self._1x1conv)

    def forward(self, buffers):
        rec_key_f = torch.stack(buffers).permute(1, 0, 2, 3)  # T x N X (C X C) -> N X T X C X C
        flat_key_f = rec_key_f.reshape(rec_key_f.shape[0], len(buffers), -1)  # N X T X (C^2)

        trans_key_f = torch.transpose(flat_key_f, 1, 2)  # N, (C*C) x T
        flat_key_f = flat_key_f.matmul(trans_key_f)  # N x T x T
        flat_key_f = nn.Softmax(dim=-1)(flat_key_f)
        weighted_key_f = torch.matmul(rec_key_f.permute(0, 2, 3, 1), flat_key_f.permute(0, 2, 1).unsqueeze(1)).permute(0, 3, 1, 2)  # # N, T x (C x C)

        # 1x1 conv
        key_f = self._1x1conv(weighted_key_f)  # N x 1 x (C x C)
        key_f = nn.Softmax(dim=-1)(key_f)

        return key_f


class Conv_Encoder(nn.Module):
    def __init__(self):
        super(Conv_Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.maxpool = nn.MaxPool2d((38, 68))
        self.avgpool = nn.AvgPool2d((38, 68))

        fill_weights(self.encoder)

    def forward(self, x):
        x = self.encoder(x)
        x_maxpool = self.maxpool(x).squeeze(-1)
        x_avgpool = self.avgpool(x).squeeze(-1)

        return x_maxpool + x_avgpool



def fill_weights(layers):
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # pyre-ignore
    for module in layers.modules():
        if isinstance(module, nn.Conv2d):
            # MSRA initial
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:  # pyre-ignore
                nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.Linear):
            # Xavier initial
            nn.init.kaiming_uniform_(module.weight, a=1)  # pyre-ignore
            if module.bias is not None:  # pyre-ignore
                nn.init.constant_(module.bias, 0)



class ChannelPool(nn.MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = F.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)