import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys

cur_dir = os.getcwd()
sys.path.append(os.path.join(cur_dir, "net"))
from resblock import Bottleneck, make_resblock
from gcn import GCN
from propunit import PropUnit


class AdaMatting(nn.Module):

    def __init__(self, in_channel):
        super(AdaMatting, self).__init__()

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        encoder_inplanes = 64
        self.encoder_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoder_resblock1, encoder_inplanes = make_resblock(encoder_inplanes, 64, blocks=3, stride=2, block=Bottleneck)
        self.encoder_resblock2, encoder_inplanes = make_resblock(encoder_inplanes, 128, blocks=4, stride=2, block=Bottleneck)
        self.encoder_resblock3, encoder_inplanes = make_resblock(encoder_inplanes, 256, blocks=6, stride=2, block=Bottleneck)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        # Shortcuts
        self.shortcut_shallow = GCN(64, 64)
        self.shortcut_middle = GCN(64 * Bottleneck.expansion, 64 * Bottleneck.expansion)
        self.shortcut_deep = GCN(128 * Bottleneck.expansion, 128 * Bottleneck.expansion)

        # T-decoder
        self.t_decoder_upscale1 = nn.Sequential(
            nn.Conv2d(256 * Bottleneck.expansion, 512 * 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2)
        )
        self.t_decoder_upscale2 = nn.Sequential(
            nn.Conv2d(512, 256 * 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2)
        )
        self.t_decoder_upscale3 = nn.Sequential(
            nn.Conv2d(256, 64 * 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2)
        )
        self.t_decoder_upscale4 = nn.Sequential(
            nn.Conv2d(64, 3 * (2 ** 2), kernel_size=1, stride=1, padding=0, bias=True),
            nn.PixelShuffle(2)
        )

        # A-deocder
        self.a_decoder_upscale1 = nn.Sequential(
            nn.Conv2d(256 * Bottleneck.expansion, 512 * 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2)
        )
        self.a_decoder_upscale2 = nn.Sequential(
            nn.Conv2d(512, 256 * 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2)
        )
        self.a_decoder_upscale3 = nn.Sequential(
            nn.Conv2d(256, 64 * 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2)
        )
        self.a_decoder_upscale4 = nn.Sequential(
            nn.Conv2d(64, 1 * (2 ** 2), kernel_size=1, stride=1, padding=0, bias=True),
            nn.PixelShuffle(2)
        )

        # Propagation unit
        self.propunit = PropUnit(
            input_dim=4 + 1 + 1,
            hidden_dim=[1],
            kernel_size=(1, 1),
            num_layers=1,
            seq_len=3,
            bias=True)

        # Task uncertainty loss
        self.log_sigma_t_sqr = nn.Parameter(torch.log(torch.Tensor([16.0])))
        self.log_sigma_a_sqr = nn.Parameter(torch.log(torch.Tensor([16.0])))


    def forward(self, x):
        raw = x
        x = self.encoder_conv(x) # 64
        encoder_shallow = self.encoder_maxpool(x) # 64

        encoder_middle = self.encoder_resblock1(encoder_shallow) # 256
        encoder_deep = self.encoder_resblock2(encoder_middle) # 512
        encoder_result = self.encoder_resblock3(encoder_deep) # 1024

        shortcut_deep = self.shortcut_deep(encoder_deep) # 512
        shortcut_middle = self.shortcut_middle(encoder_middle) # 256
        shortcut_shallow = self.shortcut_shallow(encoder_shallow) # 64

        t_decoder_deep = self.t_decoder_upscale1(encoder_result) + shortcut_deep # 512
        t_decoder_middle = self.t_decoder_upscale2(t_decoder_deep) + shortcut_middle # 256
        t_decoder_shallow = self.t_decoder_upscale3(t_decoder_middle) # 64
        trimap_adaption = self.t_decoder_upscale4(t_decoder_shallow) # 3
        t_argmax = trimap_adaption.argmax(dim=1)

        a_decoder_deep = self.a_decoder_upscale1(encoder_result) # 512
        a_decoder_middle = self.a_decoder_upscale2(a_decoder_deep) + shortcut_middle # 256
        a_decoder_shallow = self.a_decoder_upscale3(a_decoder_middle) + shortcut_shallow # 64
        a_decoder = self.a_decoder_upscale4(a_decoder_shallow) # 1

        propunit_input = torch.cat((raw, torch.unsqueeze(t_argmax, dim=1).float(), a_decoder), dim=1)
        # propunit_input = torch.cat((raw, trimap_adaption, a_decoder), dim=1)
        alpha_estimation = self.propunit(propunit_input)

        return trimap_adaption, t_argmax, alpha_estimation, self.log_sigma_t_sqr, self.log_sigma_a_sqr
