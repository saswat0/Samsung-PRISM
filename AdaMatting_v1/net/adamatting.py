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
from BR import BR

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
        self.encoder_resblock2, encoder_inplanes = make_resblock(encoder_inplanes, 128, blocks=3, stride=2, block=Bottleneck)
        self.encoder_resblock3, encoder_inplanes = make_resblock(encoder_inplanes, 256, blocks=3, stride=2, block=Bottleneck)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        #Boundary Refinement
        self.br1 = BR(64)
        self.br2 = BR(64 * Bottleneck.expansion)
        self.br3 = BR(128 * Bottleneck.expansion)

        #  RES boundary Shortcuts
        shortcut_inplanes = 64
        self.shortcut_shallow_intial, shortcut_inplanes = make_resblock(shortcut_inplanes, 256, blocks=1, stride=2, block=Bottleneck)
        self.shortcut_shallow = self.br1(self.shortcut_shallow_intial)
        self.shortcut_middle_initial, shortcut_inplanes = make_resblock(shortcut_inplanes, 256, blocks=1, stride=2, block=Bottleneck)
        self.shortcut_shallow = self.br2(self.shortcut_middle_initial)
        self.shortcut_deep_initial, shortcut_inplanes = make_resblock(shortcut_inplanes, 256, blocks=1, stride=2, block=Bottleneck)
        self.shortcut_deep = self.br3(self.shortcut_deep_initial)

        # Boundary GCN Shortcuts
        # self.shortcut_shallow_intial = GCN(64, 64)
        # self.shortcut_shallow = self.br1(self.shortcut_shallow_intial)
        # self.shortcut_middle_initial = GCN(64 * Bottleneck.expansion, 64 * Bottleneck.expansion)
        # self.shortcut_middle = self.br2(self.shortcut_middle_initial)
        # self.shortcut_deep_initial = GCN(128 * Bottleneck.expansion, 128 * Bottleneck.expansion)
        # self.shortcut_deep = self.br3(self.shortcut_deep_initial)

        # Original shortcuts
        # self.shortcut_shallow = GCN(64, 64)
        # self.shortcut_middle = GCN(64 * Bottleneck.expansion, 64 * Bottleneck.expansion)
        # self.shortcut_deep = GCN(128 * Bottleneck.expansion, 128 * Bottleneck.expansion)
        # Separate two middle shortcuts
        # self.shortcut_shallow = self.shortcut_block(64, 64)
        # self.shortcut_middle_a = self.shortcut_block(64 * Bottleneck.expansion, 64 * Bottleneck.expansion)
        # self.shortcut_middle_t = self.shortcut_block(64 * Bottleneck.expansion, 64 * Bottleneck.expansion)
        # self.shortcut_deep = self.shortcut_block(128 * Bottleneck.expansion, 128 * Bottleneck.expansion)

        # T-decoder
        self.t_decoder_upscale1 = nn.Sequential(
            self.decoder_unit(256 * Bottleneck.expansion, 512 * 4),
            self.decoder_unit(512 * 4, 512 * 4),
            nn.PixelShuffle(2)
        )
        self.t_decoder_upscale2 = nn.Sequential(
            self.decoder_unit(512, 256 * 4),
            self.decoder_unit(256 * 4, 256 * 4),
            nn.PixelShuffle(2)
        )
        self.t_decoder_upscale3 = nn.Sequential(
            self.decoder_unit(256, 64 * 4),
            self.decoder_unit(64 * 4, 64 * 4),
            nn.PixelShuffle(2)
        )
        self.t_decoder_upscale4 = nn.Sequential(
            self.decoder_unit(64, 3 * (2 ** 2)),
            self.decoder_unit(3 * (2 ** 2), 3 * (2 ** 2)),
            nn.PixelShuffle(2)
        )

        # A-deocder
        self.a_decoder_upscale1 = nn.Sequential(
            self.decoder_unit(256 * Bottleneck.expansion, 512 * 4),
            self.decoder_unit(512 * 4, 512 * 4),
            nn.PixelShuffle(2)
        )
        self.a_decoder_upscale2 = nn.Sequential(
            self.decoder_unit(512, 256 * 4),
            self.decoder_unit(256 * 4, 256 * 4),
            nn.PixelShuffle(2)
        )
        self.a_decoder_upscale3 = nn.Sequential(
            self.decoder_unit(256, 64 * 4),
            self.decoder_unit(64 * 4, 64 * 4),
            nn.PixelShuffle(2)
        )
        self.a_decoder_upscale4 = nn.Sequential(
            self.decoder_unit(64, 1 * (2 ** 2)),
            self.decoder_unit(1 * (2 ** 2), 1 * (2 ** 2)),
            nn.PixelShuffle(2)
        )

        # Propagation unit
        # self.propunit = PropUnit(
        #     input_dim=4 + 1 + 1,
        #     hidden_dim=[1],
        #     kernel_size=(3, 3),
        #     num_layers=3,
        #     seq_len=3,
        #     bias=True)
        self.prop_unit = nn.Sequential(
            nn.Conv2d(3 + 1 + 1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True),
        )

        # Task uncertainty loss parameters
        self.log_sigma_t_sqr = nn.Parameter(torch.log(torch.Tensor([16.0])))
        self.log_sigma_a_sqr = nn.Parameter(torch.log(torch.Tensor([16.0])))


    def forward(self, x):
        raw = x.clone()[:, 0:3, :, :]
        x = self.encoder_conv(x) # 64
        encoder_shallow = self.encoder_maxpool(x) # 64

        encoder_middle = self.encoder_resblock1(encoder_shallow) # 256
        encoder_deep = self.encoder_resblock2(encoder_middle) # 512
        encoder_result = self.encoder_resblock3(encoder_deep) # 1024

        t_decoder = self.t_decoder_upscale1(encoder_result) + self.shortcut_deep(encoder_deep) # 512
        t_decoder = self.t_decoder_upscale2(t_decoder) + self.shortcut_middle_t(encoder_middle) # 256
        t_decoder = self.t_decoder_upscale3(t_decoder) # 64
        t_decoder = self.t_decoder_upscale4(t_decoder) # 3
        t_argmax = t_decoder.argmax(dim=1)

        a_decoder = self.a_decoder_upscale1(encoder_result) # 512
        a_decoder = self.a_decoder_upscale2(a_decoder) + self.shortcut_middle_a(encoder_middle) # 256
        a_decoder = self.a_decoder_upscale3(a_decoder) + self.shortcut_shallow(encoder_shallow) # 64
        a_decoder = self.a_decoder_upscale4(a_decoder) # 1
        
        alpha_estimation = torch.cat((raw, torch.unsqueeze(t_argmax, dim=1).float() / 2, a_decoder), dim=1)
        # alpha_estimation = torch.cat((raw, t_decoder, a_decoder), dim=1)
        # alpha_estimation = self.propunit(alpha_estimation)
        alpha_estimation = self.prop_unit(alpha_estimation)

        return t_decoder, t_argmax, alpha_estimation, self.log_sigma_t_sqr, self.log_sigma_a_sqr
        # return t_decoder, t_argmax, a_decoder, self.log_sigma_t_sqr, self.log_sigma_a_sqr
    

    def decoder_unit(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def shortcut_block(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
