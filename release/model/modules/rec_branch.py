import torch
import torch.nn as nn


class CRNNBased(nn.Module):

    def __init__(
        self,
        img_channel,
        img_height,
        img_width,
        num_classes,
        seq_hidden=64,
        rnn_hidden=256,
    ):
        super(CRNNBased, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._get_cnn_layer(img_channel, img_height, img_width)

        self.map_to_seq = nn.Linear(output_channel * output_height, seq_hidden)

        self.rnn1 = nn.LSTM(seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

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

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def load_pretrain(self, file, origin: bool = False, map_location: str = None):
        state_dict = torch.load(file, map_location=map_location)
        self.load_state_dict(state_dict, strict=origin)

    def forward(self, images):
        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)
        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        output = torch.log_softmax(output, dim=2)
        return output


class RecWrapper(nn.Module):

    def __init__(self, module: nn.Module) -> None:
        super().__init__()

        self.module = module

    def load_pretrain(self, file: str, origin: bool = False, map_location: str = None):
        state_dicts = torch.load(file, map_location=map_location)
        self.module.load_state_dict(state_dicts)

    def forward(self, x):
        return self.module.forward(x)
