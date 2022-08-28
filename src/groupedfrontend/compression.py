from typing import Optional

import torch
import torch.nn as nn

from groupedfrontend.temporalbatchnorm import TemporalBatchNorm
from groupedfrontend.log1p import Log1p


class PCEN(nn.Module):
    """
    Trainable PCEN (Per-Channel Energy Normalization) layer:
    .. math::
        Y = (\\frac{X}{(\\epsilon + M)^\\alpha} + \\delta)^r - \\delta^r
        M_t = (1 - s) M_{t - 1} + s X_t

    Args:
        num_bands: Number of frequency bands (previous to last input dimension)
        s: Initial value for :math:`s`
        alpha: Initial value for :math:`alpha`
        delta: Initial value for :math:`delta`
        r: Initial value for :math:`r`
        eps: Value for :math:`eps`
        learn_logs: If false-ish, instead of learning the logarithm of each
          parameter (as in the PCEN paper), learn the inverse of :math:`r` and
          all other parameters directly (as in the LEAF paper).
        clamp: If given, clamps the input to the given minimum value before
          applying PCEN.
    """
    def __init__(self, num_bands: int, s: float=0.025, alpha: float=1.,
                 delta: float=1., r: float=1., eps: float=1e-6,
                 learn_logs: bool=True, clamp: Optional[float]=None):
        super(PCEN, self).__init__()
        if learn_logs:
            # learns logarithm of each parameter
            s = np.log(s)
            alpha = np.log(alpha)
            delta = np.log(delta)
            r = np.log(r)
        else:
            # learns inverse of r, and all other parameters directly
            r = 1. / r
        self.learn_logs = learn_logs
        self.s = nn.Parameter(torch.full((num_bands,), float(s)))
        self.alpha = nn.Parameter(torch.full((num_bands,), float(alpha)))
        self.delta = nn.Parameter(torch.full((num_bands,), float(delta)))
        self.r = nn.Parameter(torch.full((num_bands,), float(r)))
        self.eps = torch.as_tensor(eps)
        self.clamp = clamp

    def forward(self, x):
        # clamp if needed
        if self.clamp is not None:
            x = x.clamp(min=self.clamp)

        # prepare parameters
        if self.learn_logs:
            # learns logarithm of each parameter
            s = self.s.exp()
            alpha = self.alpha.exp()
            delta = self.delta.exp()
            r = self.r.exp()
        else:
            # learns inverse of r, and all other parameters directly
            s = self.s
            alpha = self.alpha.clamp(max=1)
            delta = self.delta.clamp(min=0)  # unclamped in original LEAF impl.
            r = 1. / self.r.clamp(min=1)
        # broadcast over channel dimension
        alpha = alpha[:, np.newaxis]
        delta = delta[:, np.newaxis]
        r = r[:, np.newaxis]

        # compute smoother
        smoother = [x[..., 0]]  # initialize the smoother with the first frame
        for frame in range(1, x.shape[-1]):
            smoother.append((1 - s) * smoother[-1] + s * x[..., frame])
        smoother = torch.stack(smoother, -1)

        # stable reformulation due to Vincent Lostanlen; original formula was:
        # return (input / (self.eps + smoother)**alpha + delta)**r - delta**r
        smoother = torch.exp(-alpha * (torch.log(self.eps) +
                                       torch.log1p(smoother / self.eps)))
        return (x * smoother + delta)**r - delta**r


# Log1p + Median filter + TBN (temporal batch normalization)
# compression function
class LogTBN(nn.Module):
    """
    Calculates the Log1p of the input signal, optionally subtracts the median
    over time, and finally applies batch normalization over time.

    :param num_bands: number of filters
    :param affine: learnable affine parameters for TBN
    :param a: value for 'a' for Log1p
    :param trainable: sets 'a' trainable for Log1p
    :param per_band: separate 'a' per band for Log1p
    :param median_filter: subtract the median of the signal over time
    :param append_filtered: if true-ish, append the median-filtered signal as
        an additional channel instead of subtracting the median in place
    """
    def __init__(
            self,
            num_bands: int,
            affine: bool=True,
            a: float=0,
            trainable: bool=False,
            per_band: bool=False,
            median_filter: bool=False,
            append_filtered: bool=False
    ):
        super(LogTBN, self).__init__()
        self.log1p = Log1p(
            a=a,
            trainable=trainable,
            per_band=per_band,
            num_bands=num_bands
        )
        self.TBN = TemporalBatchNorm(
            num_bands=num_bands,
            affine=affine,
            per_channel=append_filtered,
            num_channels=2 if append_filtered else 1
        )
        self.median_filter = median_filter
        self.append_filtered = append_filtered

    def forward(self, x):
        x = self.log1p(x)
        if self.median_filter:
            if self.append_filtered and x.ndim == 3:
                x = x[:, np.newaxis]  # add channel dimension
            m = x.median(-1, keepdim=True).values
            if self.append_filtered:
                x = torch.cat((x, x - m), dim=1)
            else:
                x = x - m
        x = self.TBN(x)
        return x
