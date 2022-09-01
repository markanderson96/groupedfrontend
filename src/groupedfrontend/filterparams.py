from typing import Optional, Union
import numpy as np
import torch


def filter_params(
    n_filters: int,
    min_freq: float,
    max_freq: float,
    sample_rate: int = 16000,
    filter_type: str = 'mel',
) -> (torch.Tensor, torch.Tensor):
    """
    Analytically calculates the center frequencies
    and sigmas of a selectable filter bank

    :param n_filters: number of filters for the filterbank
    :param min_freq: minimum cutoff for the frequencies
    :param max_freq: maximum cutoff for the frequencies
    :param sample_rate: sample rate of audio to filter
    :param filter_type: (default 'mel') type of filter
                        to calculate (mel/bark/linear)
    :return: center frequencies, sigmas both as tensors
    """
    if filter_type == 'linear':
        peaks_hz = torch.linspace(min_freq, max_freq, n_filters + 2)
    elif filter_type == 'bark':
        min_bark = 6 * np.arcsinh(min_freq / 600)
        max_bark = 6 * np.arcsinh(max_freq / 600)
        peaks_bark = torch.linspace(min_bark, max_bark, n_filters + 2)
        peaks_hz = 600 * torch.sinh(peaks_bark / 6)
    else:
        min_mel = 1127 * np.log1p(min_freq / 700.0)
        max_mel = 1127 * np.log1p(max_freq / 700.0)
        peaks_mel = torch.linspace(min_mel, max_mel, n_filters + 2)
        peaks_hz = 700 * (torch.expm1(peaks_mel / 1127))

    center_freqs = peaks_hz[1:-1] * (2 * np.pi / sample_rate)
    bandwidths = peaks_hz[2:] - peaks_hz[:-2]

    return center_freqs, bandwidths


def gabor_filters(
    size: int,
    center_freqs: torch.Tensor,
    bandwidth: torch.Tensor,
    sample_rate: int = 16000
) -> torch.Tensor:
    """
    Calculates a gabor function from given center frequencies
    and bandwidths that can be used as kernels/filters for an 1D convolution

    :param size: kernel/filter size
    :param center_freqs: center frequencies
    :param bandwidth: bandwidths of filters
    :param sample_rate (optional):
    :return: kernel/filter that can be used 1D convolution as tensor
    """
    time = torch.arange(
        -(size // 2),
        (size + 1) // 2,
        device=center_freqs.device
    )
    sigmas = (sample_rate / 2.) / bandwidth
    denominator = 1. / (np.sqrt(2 * np.pi) * sigmas)
    gaussian = torch.exp(torch.outer(1. / (2. * sigmas**2), -time**2))
    sinusoid = torch.exp(1j * torch.outer(center_freqs, time))
    return denominator[:, np.newaxis] * sinusoid * gaussian


def gamma_filters(
    size: int,
    center_freqs: torch.Tensor,
    bandwidth: torch.Tensor,
    order: int = 4,
    sample_rate: int = 16000
) -> torch.Tensor:
    """
    Calculates a gammatone filter function from given center frequencies
    and bandwidths that can be used as kernels/filters for an 1D convolution

    :param size: kernel/filter size
    :param center_freqs: center frequencies
    :param bandwidth: bandwidths of filters
    :param order: order of filter (default 4 to model human auditory system)
    :return: kernel/filter that can be used 1D convolution as tensor
    """
    time = torch.arange(size, device=center_freqs.device)
    bandwidth_w = np.pi * (bandwidth / sample_rate)
    envelope = (time**(order - 1)) * torch.exp(-torch.outer(bandwidth_w, time))
    carrier = torch.exp(1j * torch.outer(center_freqs, time))
    scale = 1. / torch.amax(envelope, dim=1)
    return scale[:, np.newaxis] * envelope * carrier


def gauss_windows(size: int, sigmas: torch.Tensor) -> torch.Tensor:
    """
    Calculates a gaussian lowpass function from given bandwidths
    that can be used as kernels/filters for an 1D convolution

    :param size: kernel/filter size
    :param sigmas: sigmas/bandwidths
    :return: kernel/filter that can be used 1D convolution as torch.Tensor
    """
    time = torch.arange(0, size, device=sigmas.device)
    numerator = time * (2 / (size - 1)) - 1
    return torch.exp(-0.5 * (numerator / sigmas[:, np.newaxis])**2)
