# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import yaml
from attrdict import AttrDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')

import cv2
import math
import torchvision
from torchvision.models.optical_flow import raft_large
from einops import rearrange

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

from mamba_ssm import Mamba, Mamba2


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


# https://github.com/HRNet/HRNet-Image-Classification/blob/master/lib/models/cls_hrnet.py
class HRNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(HRNet, self).__init__()

        self._frames_in  = cfg['frames_in']
        self._frames_out = cfg['frames_out']
        self._out_scales = cfg['out_scales']
        self._stem_strides  = cfg['MODEL']['EXTRA']['STEM']['STRIDES']
        self._stem_inplanes = cfg['MODEL']['EXTRA']['STEM']['INPLANES']

        self.conv1 = nn.Conv2d(3*self._frames_in, self._stem_inplanes, kernel_size=3, stride=self._stem_strides[0], padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(self._stem_inplanes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(self._stem_inplanes, self._stem_inplanes, kernel_size=3, stride=self._stem_strides[1], padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(self._stem_inplanes, momentum=BN_MOMENTUM)
        self.relu  = nn.ReLU(inplace=True)

        self.stage1_cfg = cfg['MODEL']['EXTRA']['STAGE1']
        num_channels    = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, self._stem_inplanes, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        self.num_deconvs       = cfg['MODEL']['EXTRA']['DECONV']['NUM_DECONVS']
        self.deconv_config     = cfg['MODEL']['EXTRA']['DECONV']
        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

        self.deconv_layers = self._make_deconv_layers(cfg, pre_stage_channels[0])
        self.final_layers  = self._make_final_layers(cfg, pre_stage_channels)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_final_layers(self, cfg, channels):
        kernel_size = cfg['MODEL']['EXTRA']['FINAL_CONV_KERNEL']
        layers      = []
        for scale in self._out_scales:
            layers.append( nn.Conv2d(in_channels=channels[scale], out_channels=self._frames_out, kernel_size=kernel_size) )
        return nn.ModuleList(layers)

    def _make_deconv_layers(self, cfg, input_channels):
        extra      = cfg.MODEL.EXTRA
        deconv_cfg = extra.DECONV

        deconv_layers = []
        for i in range(deconv_cfg.NUM_DECONVS): 
            output_channels                        = input_channels
            deconv_kernel, padding, output_padding = self._get_deconv_cfg(deconv_cfg.KERNEL_SIZE[i])

            layers = []
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False),
                nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ))
            deconv_layers.append(nn.Sequential(*layers))
            input_channels = output_channels

        return nn.ModuleList(deconv_layers)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        """_summary_

        Args:
            x (tensor): [B, N, C, H, W]

        Returns:
            _type_: _description_
        """
        B,N,C,H,W = x.shape
        # Permute to bring frames and channels together
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # Shape: [B, C, N, H, W]

        # Reshape to combine frames into the channel dimension
        x = x.view(B, N * C, H, W) # Shape: [B, N*C, H, W]
        

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        y_out = {}
        for scale in self._out_scales:
            x = y_list[scale]
            for i in range(self.num_deconvs):
                x = self.deconv_layers[i][scale](x)
            y = self.final_layers[scale](x)
            y_out[scale] = post_process_heatmap(y)
        return y_out[0][0], y_out[0][1]

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def post_process_heatmap(heatmap):
    """
    Post-process a heatmap to produce two 1D distributions over height and width.

    Args:
    - heatmap: Tensor of shape [b, n, h, w] (batch, n frames, height, width).

    Returns:
    - vertical_heatmap: Tensor of shape [b, h], softmaxed along the height.
    - horizontal_heatmap: Tensor of shape [b, w], softmaxed along the width.
    """
    heatmap = heatmap[:,-1:,:] # [b,1,h,w]

    vertical_heatmap = heatmap.max(dim=-1)[0].squeeze(dim=1)  # Max along width (W)
    horizontal_heatmap = heatmap.max(dim=-2)[0].squeeze(dim=1)  # Max along height (H)
    # # Min-max normalization for vertical and horizontal heatmaps
    # vertical_heatmap = (vertical_heatmap - vertical_heatmap.min()) / (vertical_heatmap.max() - vertical_heatmap.min() + 1e-8)
    # horizontal_heatmap = (horizontal_heatmap - horizontal_heatmap.min()) / (horizontal_heatmap.max() - horizontal_heatmap.min() + 1e-8)

    # vertical_heatmap = torch.softmax(vertical_heatmap, dim=-1)
    # horizontal_heatmap = torch.softmax(horizontal_heatmap, dim=-1)

    return horizontal_heatmap, vertical_heatmap




class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(288, 512), patch_size=(8, 8), kernel_size=1, in_chans=2, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class MambaBlock(nn.Module):
    def __init__(self, d_model, num_mamba_layers=2, dropout_rate=0.1):
        """
        Args:
            d_model: Dimension of the input.
            num_mamba_layers: Number of Mamba layers in the block.
            dropout_rate: Dropout rate to apply after each Mamba layer.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                Mamba(d_model=d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout_rate)
            )
            for _ in range(num_mamba_layers)
        ])

    def forward(self, x):
        """
        Forward pass for the Mamba block.
        Args:
            x: Input tensor of shape [B, N, M], where B is batch size, N is sequence length, and M is feature size.
        """
        for layer in self.layers:
            x = layer(x)  # Apply each Mamba layer with LayerNorm and Dropout

        return x


class Mamba_Model(nn.Module):
    def __init__(self, num_frames, embed_dim=192, num_mamba_layers=6, kernel_size=1, drop_rate=0.2):
        super(Mamba_Model, self).__init__()
        self.embed_dim = embed_dim
        self.optical_flow_model = raft_large(weights=torchvision.models.optical_flow.Raft_Large_Weights.DEFAULT, progress=False).eval()
        self.optical_flow_model.requires_grad_(False)  # Freeze RAFT parameters
        self.patch_embed = PatchEmbed(embed_dim=self.embed_dim, in_chans=2)
        self.num_patches = self.patch_embed.num_patches

        self.flow_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, (num_frames-1) // kernel_size, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.block = MambaBlock(d_model=embed_dim, num_mamba_layers=num_mamba_layers, dropout_rate=drop_rate)

    
    def forward(self, x):
        """_summary_

        Args:
            x (tensor): input tensor is in shape [B,N,C,H,W] which represents a series of frames from a video 

        Returns:
            _type_: _description_
        """
        B, T, _, H, W = x.shape

        # first step is to extract optical flow
        flows = []
        for i in range(T - 1):
            frame1 = x[:, i, :, :, :]  # [B, C, H, W]
            frame2 = x[:, i + 1, :, :, :]  # [B, C, H, W]
            # Compute optical flow
            list_of_flows = self.optical_flow_model(frame1, frame2)  # The RAFT model outputs lists of predicted flows where each entry is a (N, 2, H, W) batch of predicted flows that corresponds to a given “iteration” in the model. 
            predicted_flows = list_of_flows[-1]  # Output: [B, 2, H, W]
            flows.append(predicted_flows)

        # Stack optical flows
        optical_flow = torch.stack(flows, dim=1)  # [B, T-1, 2, H, W]
        optical_flow_normalized = (optical_flow - optical_flow.mean()) / optical_flow.std()  # Normalize optical flow to zero mean, unit variance
        # pad optical flow to match temporal dimension 
        # padding = torch.zeros(optical_flow.shape[0], 1, optical_flow.shape[2], optical_flow.shape[3], optical_flow.shape[4], device=optical_flow.device)  # [B, 1, 2, H, W]
        # padded_optical_flow = torch.cat([optical_flow_normalized, padding], dim=1)  # [B, T, 2, H, W]
        # # concat both optical flow with original RGB frames
        # combined_input = torch.cat([x, padded_optical_flow], dim=2)  # Shape: [B, T, 5, H, W]
        combined_input = optical_flow_normalized
        
        combined_input = rearrange(combined_input, "b t c h w -> b c t h w")
  
        # project it on to 1D dimensions
        x = self.patch_embed(combined_input) # [B, C, T-1, H, W] M means embed dimensions, T means frames
        x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        x = x + self.pos_embed

        # concat flow and cls token
        flow_token = self.flow_token.expand(x.shape[0], -1, -1) 
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.concat((x, flow_token), dim=1)
        x = torch.concat((x, cls_token), dim=1)

        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T-1) # n means number of patches, m means featre dimension 
        x = x + self.temporal_pos_embedding
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T-1)
        x = self.pos_drop(x)
        x = self.layer_norm(x)
        x = self.block(x)
        cls_flow_token = x[:, :2, :]

        return cls_flow_token



class TwoStreamHRNet(nn.Module):
    def __init__(self, num_frames, device, cfg_spatial, embed_dim=192, num_mamba_layers=6, dropout_rate=0.1):
        """
        Args:
            cfg_spatial: Configuration dictionary for the spatial stream.
            cfg_temporal: Configuration dictionary for the temporal stream.
        """
        super(TwoStreamHRNet, self).__init__()
        self.device = device
        self.spatial_stream = HRNet(cfg_spatial)  # Spatial stream
        self.temporal_stream = Mamba_Model(num_frames, embed_dim=embed_dim, num_mamba_layers=num_mamba_layers, drop_rate=dropout_rate)  # Temporal stream
        # self.cls_project = nn.Linear(embed_dim, 4)
        # Projection layers for generating optical flow heatmaps
        self.flow_to_horizontal = nn.Sequential(
            nn.Linear(embed_dim, cfg_spatial['inp_width']),  # Project to horizontal heatmap
            nn.LayerNorm(cfg_spatial['inp_width']),          # Normalize output
        )

        self.flow_to_vertical = nn.Sequential(
            nn.Linear(embed_dim, cfg_spatial['inp_height']),  # Project to vertical heatmap
            nn.LayerNorm(cfg_spatial['inp_height']),          # Normalize output
        )
        self.softmax = nn.Softmax(dim=1)
        # self.state_weights = torch.tensor([0.0, 0.0, 0.5, 1.0], device=self.device)  # Adjust weights as needed

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize linear layers with Xavier initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                # Initialize Conv2D layers with Kaiming initialization
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm weights and biases are initialized to 1 and 0 respectively
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm weights are initialized to 1, biases to 0
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize all nn.Parameter instances
        for name, param in self.named_parameters():
            if "weight" in name and param.ndim > 1:  # For weight parameters (multi-dimensional)
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))  # Use Kaiming initialization for weights
            elif "bias" in name:  # For bias parameters (1-dimensional)
                fan_in = param.size(0)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(param, -bound, bound)
            elif "token" in name or "embed" in name:  # For learnable tokens and embeddings
                nn.init.normal_(param, mean=0.0, std=0.02)  # Use a small normal distribution
    


    def forward(self, x):
        """
        Args:
            spatial_input: Input for the spatial stream (e.g., RGB frames).
            temporal_input: Input for the temporal stream (e.g., optical flow).

        Returns:
            Output from the fused two-stream network.
        """
        # Process spatial stream
        spatial_features = self.spatial_stream(x)  # Outputs multi-scale features
        horizontal_heatmap, vertical_heatmap = spatial_features # outputs [B, W], [B, H]

        # Process temporal stream
        temporal_features = self.temporal_stream(x)  # outputs [B, 2, M] where the first is the classification token, second is flow tokens
        # cls_score = self.cls_project(temporal_features[:, :1, :]).squeeze(dim=1) #[B, 4]
        # cls_prob = self.softmax(cls_score)
        # confidence_score = (cls_prob * self.state_weights).sum(dim=-1, keepdim=True)  # [B, 1]

        # flow token which represents optical flow map
        flow_token = temporal_features[:, 1:2, :] #[B, 1, M]
        # Project flow token into horizontal and vertical flow maps
        horizontal_flow_map = self.flow_to_horizontal(flow_token.squeeze(1))  # [B, W]
        vertical_flow_map = self.flow_to_vertical(flow_token.squeeze(1))      # [B, H]

        final_horizontal_heatmap = horizontal_heatmap+horizontal_flow_map # [B, W]
        final_vertical_heatmap = vertical_heatmap+vertical_flow_map        # [B, H]

        final_horizontal_heatmap = self.softmax(final_horizontal_heatmap)
        final_vertical_heatmap = self.softmax(final_vertical_heatmap)

        return (final_horizontal_heatmap, final_vertical_heatmap), None




def build_two_streams_model(args):
    wasb_config_path = '/home/s224705071/github/PhysicsInformedDeformableAttentionNetwork/src/config/two_stream_network.yaml'
    with open(wasb_config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg['frames_in'] = args.num_frames
        cfg['frames_out'] = 1
    wasb_cfg = AttrDict(cfg)
    model = TwoStreamHRNet(args.num_frames, args.device, wasb_cfg, embed_dim=256, num_mamba_layers=6).to(args.device)

    return model


if __name__=='__main__':
    from config.config import parse_configs
    from data_process.dataloader import create_occlusion_train_val_dataloader
    from model.model_utils import get_num_parameters
    from losses_metrics.losses import Heatmap_Ball_Detection_Loss, focal_loss

    configs = parse_configs()
    configs.device = 'cuda'
    configs.num_frames = 5
    configs.img_size = (288, 512)
    configs.dataset_choice = 'tennis'

    heatmap_loss = Heatmap_Ball_Detection_Loss()

    train_dataloader, val_dataloader, train_sampler = create_occlusion_train_val_dataloader(configs) 
    batch_data, (masked_frameids, labels, visibility, _) = next(iter(train_dataloader)) # batch data will be in shape [B, N, C, H, W]
    print(torch.unique(batch_data))
    batch_data = batch_data.float().to(configs.device)
    labels = labels.float().to(configs.device)
    visibility = visibility.to(configs.device)

    model = build_two_streams_model(configs)
    print(f"model num params is {get_num_parameters(model)}")
    (horizontal_heatmap, vertical_heatmap), cls_score = model(batch_data)
    loss = heatmap_loss((horizontal_heatmap, vertical_heatmap), labels)
    print(f"heatmap loss is {loss}")
    # print(f"cls loss is {focal_loss(cls_score, visibility)}")
    # print(y[0].shape, y[1].shape)
    print(horizontal_heatmap.shape, vertical_heatmap.shape)
    print(torch.unique(vertical_heatmap))


