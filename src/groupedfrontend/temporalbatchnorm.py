from typing import Optional

import torch
import torch.nn as nn

class TemporalBatchNorm(nn.Module):
    """
    Batch normalization of a (batch, channels, bands, time) tensor over all but
    the previous to last dimension (the frequency bands). If per_channel is
    true-ish, normalize each channel separately instead of joining them.

    :param num_bands: number of filters
    :param affine: learnable affine parameters
    :param per_channel: normalize each channel separately
    :param num_channels: number of input channels
    """
    def __init__(
            self,
            num_bands: int,
            affine: bool=True,
            per_channel: bool=False,
            num_channels: Optional[int]=None
    ):
        super(TemporalBatchNorm, self).__init__()
        num_features = num_bands * num_channels if per_channel else num_bands
        self.bn = nn.BatchNorm1d(num_features, affine=affine)
        self.per_channel = per_channel

    def forward(self, x):
        shape = x.shape
        if self.per_channel:
            # squash channels into the bands dimension
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        else:
            # squash channels into the batch dimension
            x = x.reshape((-1,) + x.shape[-2:])
        # pass through 1D batch normalization
        x = self.bn(x)
        # restore squashed dimensions
        return x.reshape(shape)
