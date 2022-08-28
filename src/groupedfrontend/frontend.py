from typing import Union

import numpy as np
import torch
import torch.nn as nn

from groupedfrontend.filterbanks import GroupedFilterbank
from groupedfrontend.compression import PCEN, LogTBN


class GroupedFrontend(nn.Module):
    """
    Grouped frontend based on the EfficientLEAF frontend.
    This is a NON-learnable front-end that takes an audio waveform
    as input and outputs a learnable spectral representation.
    Initially approximates the computation of standard mel-filterbanks.

    A detailed technical description of EfficientLEAF
    is presented in Section 3 of https://arxiv.org/abs/2101.08596 .

    :param n_filters: number of filters
    :param min_freq: minimum frequency
                     (used for the mel filterbank initialization)
    :param max_freq: maximum frequency
                     (used for the mel filterbank initialization)
    :param sample_rate: sample rate
                        (used for the mel filterbank initialization)
    :param window_len: kernel/filter size of the convolutions in ms
    :param window_stride: stride used for the pooling convolution in ms
    :param conv_win_factor: factor is multiplied with the kernel/filter size
                            (filterbank)
    :param stride_factor: factor is multiplied with the kernel/filter stride
                          (filterbank)
    :param compression: compression function used: 'pcen', 'logtbn'
                        or a torch module (default: 'logtbn')
    """
    def __init__(
            self,
            n_filters: int=40,
            num_groups: int=4,
            min_freq: float=60.0,
            max_freq: float=7800.0,
            sample_rate: int=16000,
            window_len: float=25.,
            window_stride: float=10.,
            conv_win_factor: float=4.77,
            stride_factor: float=1.,
            compression: Union[str, torch.nn.Module]=None,
            filter_type: str = 'gabor',
            init_filter: str = 'mel'
    ):
        super(GroupedFrontend, self).__init__()

        # convert window sizes from milliseconds to samples
        window_size = int(sample_rate * window_len / 1000)
        window_size += 1 - (window_size % 2)  # make odd
        window_stride = int(sample_rate * window_stride / 1000)

        self.filterbank = GroupedFilterbank(
            n_filters,
            num_groups,
            min_freq,
            max_freq,
            sample_rate,
            pool_size=window_size,
            pool_stride=window_stride,
            conv_win_factor=conv_win_factor,
            stride_factor=stride_factor,
            filter_type=filter_type,
            init_filter=init_filter
        )

        if compression == 'pcen':
            self.compression = PCEN(
                n_filters,
                s=0.04,
                alpha=0.96,
                delta=2,
                r=0.5,
                eps=1e-12,
                learn_logs=False,
                clamp=1e-5
            )
        elif compression == 'logtbn':
            self.compression = LogTBN(
                n_filters,
                a=5,
                trainable=True,
                per_band=True,
                median_filter=True,
                append_filtered=True
            )
        elif isinstance(compression, torch.nn.Module):
            self.compression = compression
        elif compression == None:
            self.compression = None
        else:
            raise ValueError("unsupported value for compression argument")

    def forward(self, x: torch.tensor):
        while x.ndim < 3:
            x = x[:, np.newaxis]
        x = self.filterbank(x)
        if self.compression:
            x = self.compression(x)

        return x
