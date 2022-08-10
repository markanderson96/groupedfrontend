import torch
import torch.nn as nn
import numpy as np

class Log1p(nn.Module):
    """
    Applies `log(1 + 10**a * x)`, with `a` fixed or trainable.
    If `per_band` and `num_bands` are given, learn `a` separately per band.

    :param a: value for 'a'
    :param trainable: sets 'a' trainable
    :param per_band: separate 'a' per band
    :param num_bands: number of filters
    """
    def __init__(self, a=0, trainable=False, per_band=False, num_bands=None):
        super(Log1p, self).__init__()
        if trainable:
            dtype = torch.get_default_dtype()
            if not per_band:
                a = torch.tensor(a, dtype=dtype)
            else:
                a = torch.full((num_bands,), a, dtype=dtype)
            a = nn.Parameter(a)
        self.a = a
        self.trainable = trainable
        self.per_band = per_band

    def forward(self, x):
        if self.trainable or self.a != 0:
            a = self.a[:, np.newaxis] if self.per_band else self.a
            x = 10 ** a * x
        return torch.log1p(x)

    def extra_repr(self):
        return 'trainable={}, per_band={}'.format(repr(self.trainable),
                                                  repr(self.per_band))
