"""
The recognizer can be any off-the-shelf model,
model's first layer (or more layers) can be treated as
encoder(s) for the whole model,
and the model's last classifier head is the decoder,
other parts of recognizer just keeps the same.
If recognizer is pretrained, the classifier head should be train again
in your new datasets, namely finetune.
"""

import copy
import torch
import torch.nn as nn
from torch import Tensor
from model.modules.srb import PixelShuffleBlock
from model.utils.tokenizer import CTCTokenizer


class CRNNRecognizer(nn.Module):

    def __init__(
        self,
        in_channels: int = 64,
        num_class: int = 37,
        mid_linear: int = 64,
        rnn_hidden: int = 256,
        scale: int = 2,
    ):
        super(CRNNRecognizer, self).__init__()

        # up sample to 32, 128
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * scale ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

        self.cnn_layer = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # 16, 64
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # 8, 32
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # 4, 32
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            # 2 ,32
            nn.MaxPool2d(kernel_size=(2, 1)),
            # 1, 31
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.ReLU(),
        )

        cnn_out_size = (1, 31)
        cnn_out_channels = 512

        mid_channels = cnn_out_size[0] * cnn_out_channels

        self.map_to_seq = nn.Linear(mid_channels, mid_linear)
        self.mid_linear0 = nn.Linear(mid_linear, 3 * mid_linear, bias=False)
        self.mid_batchnorm = nn.BatchNorm1d(3 * mid_linear)
        self.mid_linear1 = nn.Linear(3 * mid_linear, mid_linear)

        self.rnn1 = nn.LSTM(mid_linear, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.linear0 = nn.Linear(2 * rnn_hidden, 4 * rnn_hidden)
        self.linear1 = nn.Linear(4 * rnn_hidden, 2 * rnn_hidden)
        self.dense = nn.Linear(2 * rnn_hidden, num_class)

        self.loss_fn = nn.CTCLoss(CTCTokenizer._BLANK_ID_, zero_infinity=True)

    def forward(self, x: Tensor):
        x = self.preprocess(x)
        x = self.cnn_layer(x)

        batch, channel, height, width = x.size()
        x = x.view(batch, channel * height, width)
        x = x.permute(2, 0, 1)

        x = self.map_to_seq(x)
        x = self.mid_linear0(x)
        x = x.permute(1, 2, 0)
        x = self.mid_batchnorm(x)
        x = x.permute(2, 0, 1)
        x = self.mid_linear1(x)

        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)

        x = self.linear0(x)
        x = self.linear1(x)
        x = self.dense(x)

        x = torch.log_softmax(x, dim=2)

        return x


class CRNNBased(nn.Module):

    def __init__(
        self,
        img_channel=1,
        img_height=32,
        img_width=128,
        num_classes=37,
        seq_hidden=64,
        rnn_hidden=256,
        model_channels: int = 64,
        scale: int = 2,
        out_channels: int = 4,
        mlp_classifier: bool = False,
        side_rnn: bool = False,
        input_type: str = 'img',  # img | deep | shallow
    ):
        super(CRNNBased, self).__init__()

        self.input_type = input_type

        self.block8 = nn.Sequential(
            PixelShuffleBlock(model_channels, scale),
            nn.Conv2d(model_channels, out_channels, kernel_size=9, padding=4),
        )
        self.activate = torch.tanh
        self.normalize = lambda x: 2 * x - 1

        self.cnn, (output_channel, output_height, output_width) = self._get_cnn_layer(img_channel, img_height, img_width)

        self.map_to_seq = nn.Linear(output_channel * output_height, seq_hidden)

        self.rnn1 = nn.LSTM(seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.requires_side_rnn = side_rnn
        self.requires_mlp_classifier = mlp_classifier

        if self.requires_side_rnn:
            self.side_rnn1 = nn.LSTM(seq_hidden, rnn_hidden, bidirectional=True)
            self.side_rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)
        if self.requires_mlp_classifier:
            self.mlp = nn.Sequential(
                nn.Linear(2 * rnn_hidden, 4 * rnn_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(4 * rnn_hidden, 2 * rnn_hidden),
            )

        self.dense = nn.Linear(2 * rnn_hidden, num_classes)

    def _get_cnn_layer(self, img_channel, img_height, img_width):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]  # keep last 3 layers
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def _cnn_block(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i + 1]

            cnn.add_module(f'conv{i}', nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i]))
            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))
            relu = nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        if self.input_type == 'img':
            _cnn_block(0)
            cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        _cnn_block(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        _cnn_block(2)
        _cnn_block(3)
        cnn.add_module('pooling2', nn.MaxPool2d(kernel_size=(2, 1)))
        _cnn_block(4, batch_norm=True)
        _cnn_block(5, batch_norm=True)
        cnn.add_module('pooling3', nn.MaxPool2d(kernel_size=(2, 1)))
        _cnn_block(6)

        output_channel, output_height, output_width = channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def load_pretrain(self, file, origin: bool = False, map_location: str = None):
        state_dict = torch.load(file, map_location=map_location)
        self.load_state_dict(state_dict, strict=origin)

    def forward(self, enc_x):
        # input sr img
        # if self.input_type == 'img':
        #     enc_x = self.block8(enc_x)
        #     enc_x = self.activate(enc_x)
        #     enc_x = enc_x[:, :3, :, :].mean(dim=1, keepdim=True)
        #     enc_x = self.normalize(enc_x)
        # input deep feat
        # if self.input_type == 'deep':
        #     enc_x = self.block8(enc_x)
        conv = self.cnn(enc_x)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)
        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        if self.requires_side_rnn:
            side_rec, _ = self.side_rnn1(seq)
            side_rec, _ = self.side_rnn2(side_rec)
            recurrent = 0.5 * recurrent + 0.5 * side_rec

        if self.requires_mlp_classifier:
            recurrent = self.mlp(recurrent)

        output = self.dense(recurrent)
        output = torch.log_softmax(output, dim=2)
        return output
