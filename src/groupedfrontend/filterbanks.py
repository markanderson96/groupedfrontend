import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from groupedfrontend.filterparams import *


# Complex Conv with different Strides/Kernelsizes
class GroupedFilterbank(nn.Module):
    """
    Torch module that functions as a gabor/gammatone filterbank.
    Initializes n_filters center frequencies and bandwidths that are based on
    a selected filterbank. The parameters are used to calculate filters
    for a 1D convolution over the input signal. The squared modulus is taken
    from the results. To reduce the temporal resolution a gaussian lowpass
    filter is calculated from pooling_widths, which are used to perform a
    pooling operation. The center frequencies, bandwidths and pooling_widths
    are learnable parameters.

    The module splits the different filters into num_groups and calculates for
    each group a separate kernel size and stride, so at the end all groups can
    be merged to a single output. conv_win_factor and stride_factor are
    parameters that can be used to influence the kernel size and stride.

    :param n_filters: number of filters
    :param num_groups: number of groups
    :param min_freq: minimum frequency
                     (used for the mel filterbank initialization)
    :param max_freq: maximum frequency
                     (used for the mel filterbank initialization)
    :param sample_rate: sample rate
                        (used for the mel filterbank initialization)
    :param pool_size: size of the kernels/filters for pooling convolution
    :param pool_stride: stride of the pooling convolution
    :param pool_init: initial value for the gaussian lowpass function
    :param conv_win_factor: factor is multiplied with the kernel/filter size
    :param stride_factor: factor is multiplied with the kernel/filter stride
    :param init_filter: initial sacle for filterbank
    """
    def __init__(
            self,
            n_filters: int,
            num_groups: int,
            min_freq: float,
            max_freq: float,
            sample_rate: int,
            pool_size: int,
            pool_stride: int,
            pool_init: float = 0.4,
            conv_win_factor: float = 3,
            stride_factor: float = 1.,
            filter_type: str = 'gabor',
            init_filter: str = 'mel'
    ):
        super(GroupedFilterbank, self).__init__()
        # fixed inits
        self.num_groups = num_groups
        self.n_filters = n_filters
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.conv_win_factor = conv_win_factor
        self.stride_factor = stride_factor
        self.psbl_strides = [i for i in range(1, pool_stride+1)
                             if pool_stride % i == 0] #possilbe strides

        # Gabor or Gammatone
        self.filter_type = filter_type

        # parameter inits
        self.sample_rate = sample_rate
        self.center_freqs, self.bandwidths = filter_params(
            n_filters,
            min_freq,
            max_freq,
            sample_rate,
            filter_type=init_filter
        )

        self.pooling_widths = nn.Parameter(torch.full((n_filters,), float(pool_init)))

    def get_stride(self, cent_freq):
        '''
        Calculates the dynamic convolution and pooling stride,
        based on the max center frequency of the group.
        This ensures that the outputs for each group have the same dimensions.

        :param cent_freq: max center frequency
        '''
        stride = max(1, np.pi / cent_freq * self.stride_factor)
        stride = self.psbl_strides[
            np.searchsorted(
                self.psbl_strides,
                stride,
                side='right') - 1
        ]
        return stride, self.pool_stride // stride

    def forward(self, x):
        # constraint center frequencies and pooling widths
        bandwidths = self.bandwidths
        center_freqs = self.center_freqs

        # iterate over groups
        splits = (np.arange(self.num_groups + 1)
                  * self.n_filters
                  // self.num_groups)
        outputs = []
        for i, (a, b) in enumerate(zip(splits[:-1], splits[1:])):
            num_group_filters = b-a

            # calculate strides
            conv_stride, pool_stride = self.get_stride(
                torch.max(center_freqs[a:b].detach()).item()
            )

            # complex convolution
            ## compute filters
            kernel_size = int(
                max(bandwidths[a:b].detach()) * self.conv_win_factor
            )
            kernel_size += 1 - kernel_size % 2  # make odd if needed

            if self.filter_type == 'gabor':
                kernel = gabor_filters(
                    kernel_size,
                    center_freqs[a:b],
                    bandwidths[a:b],
                    sample_rate=self.sample_rate
                )
            elif self.filter_type == 'gammatone':
                kernel = gamma_filters(
                    kernel_size,
                    center_freqs[a:b],
                    bandwidths[a:b],
                    sample_rate=self.sample_rate
                )
            else:
                raise ValueError(f'{self.filter_type} not supported!')

            kernel = torch.cat((kernel.real, kernel.imag), dim=0).unsqueeze(1)
            ## convolve with filters
            output = F.conv1d(
                x,
                kernel,
                stride=conv_stride,
                padding=kernel_size//2
            )

            # compute squared modulus
            output = output ** 2
            output = (output[:, :num_group_filters]
                      + output[:, num_group_filters:])

            # pooling convolution
            ## compute filters
            window_size = int(self.pool_size / conv_stride + .5)
            window_size += 1 - window_size % 2  # make odd if needed
            sigma = (self.pooling_widths[a:b] / conv_stride
                     * self.pool_size / window_size)
            windows = gauss_windows(window_size, sigma).unsqueeze(1)
            ## apply temporal pooling
            output = F.conv1d(
                output,
                windows,
                stride=pool_stride,
                padding=window_size // 2,
                groups=num_group_filters
            )

            outputs.append(output)

        # combine outputs
        output = torch.cat(outputs, dim=1)

        return output
