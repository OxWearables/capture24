import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):
    r""" Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer('kernel', kernel[None, None, :].repeat((channels, 1, 1)))

    def forward(self, x):
        return F.conv1d(x, self.kernel, stride=self.stride, padding=self.padding, groups=x.shape[1])


class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:
          bn-relu-conv-bn-relu-conv
         /                         \
        x --------------------------(+)->
    """

    def __init__(
        self, in_channels, out_channels,
        kernel_size=3, stride=1, padding=1,
    ):
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size, stride, padding,
                               bias=False, padding_mode='circular')
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size, stride, padding,
                               bias=False, padding_mode='circular')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))

        x = self.conv2(x)
        x = x + identity

        return x


class Resnet(nn.Module):
    r""" The general form of the architecture can be described as follows:

    x->[conv-[ResBlock]^m-bn-relu-down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-relu-conv-bn-relu-conv
           /                         \                      /                         \
    x->conv --------------------------(+)-bn-relu-down->conv --------------------------(+)-bn-relu-down-> ...

    """

    def __init__(self,
                 n_channels, outsize,
                 n_filters_list,
                 kernel_size_list,
                 n_resblocks_list,
                 resblock_kernel_size_list,
                 downfactor_list,
                 downorder_list,
                 drop1, drop2,
                 fc_size,
                 is_cnnlstm=False):
        super(Resnet, self).__init__()

        self.is_cnnlstm = is_cnnlstm

        # Broadcast if single number provided instead of list
        if isinstance(kernel_size_list, int):
            kernel_size_list = [kernel_size_list] * len(downfactor_list)

        if isinstance(resblock_kernel_size_list, int):
            resblock_kernel_size_list = [resblock_kernel_size_list] * len(downfactor_list)

        if isinstance(n_resblocks_list, int):
            n_resblocks_list = [n_resblocks_list] * len(downfactor_list)

        cfg = zip(n_filters_list,
                  kernel_size_list,
                  n_resblocks_list,
                  resblock_kernel_size_list,
                  downfactor_list,
                  downorder_list)

        resnet = nn.Sequential()

        # Input channel dropout
        resnet.add_module('input_dropout', nn.Dropout2d(drop1))

        # Main layers
        in_channels = n_channels
        for i, layer_params in enumerate(cfg):
            out_channels, kernel_size, n_resblocks, resblock_kernel_size, downfactor, downorder = layer_params
            resnet.add_module(f'layer{i+1}', Resnet.make_layer(in_channels, out_channels,
                                                               kernel_size, n_resblocks, resblock_kernel_size,
                                                               downfactor, downorder))
            in_channels = out_channels

        if not is_cnnlstm:
            # Fully-connected layer
            resnet.add_module('fc', nn.Sequential(nn.Dropout2d(drop2),
                                                  nn.Conv1d(in_channels, fc_size, 1, 1, 0, bias=False),
                                                  nn.ReLU(True)))
            # Final linear layer
            resnet.add_module('final', nn.Conv1d(fc_size, outsize, 1, 1, 0, bias=False))

        else:
            self.lstm = nn.LSTM(in_channels, int(fc_size // 2), batch_first=True, bidirectional=True)
            self.final = nn.Linear(fc_size, outsize, bias=False)

        self.resnet = resnet

    @staticmethod
    def make_layer(in_channels, out_channels,
                   kernel_size, n_resblocks, resblock_kernel_size,
                   downfactor, downorder):
        r""" Basic layer in Resnets:

        x->[conv-[ResBlock]^m-bn-relu-down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        assert kernel_size % 2, "Only odd number for conv_kernel_size supported"
        assert resblock_kernel_size % 2, "Only odd number for resblock_kernel_size supported"

        padding = int((kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [nn.Conv1d(in_channels, out_channels,
                             kernel_size, 1, padding,
                             bias=False, padding_mode='circular')]

        for _ in range(n_resblocks):
            modules.append(ResBlock(out_channels, out_channels,
                                    resblock_kernel_size, 1, resblock_padding))

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x):
        if self.is_cnnlstm:
            bsize, seqlen, _, _ = x.shape
            x = x.view(-1, x.shape[2], x.shape[3])  # merge batch and sequence axes
            feats = self.resnet(x).view(bsize, seqlen, -1)
            lstm_feats, (h_n, c_n) = self.lstm(feats)
            lstm_feats = lstm_feats.reshape(-1, lstm_feats.shape[2])  # merge batch and sequence axes
            y = self.final(lstm_feats).view(bsize, seqlen, -1)

        else:
            y = self.resnet(x).view(x.shape[0], -1)

        return y
