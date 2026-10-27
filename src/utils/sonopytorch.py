# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of wake-up-word-detection.
#
# wake-up-word-detection is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# wake-up-word-detection is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with wake-up-word-detection. If not, see https://www.gnu.org/licenses/

import torch
import math
from typing import Tuple
#from functools import lru_cache


def chop_array(arr: torch.Tensor, window_size: int, hop_size: int):
    """chop_array([1,2,3], 2, 1) -> [[1,2], [2,3]]"""
    audio_legth = arr.shape[0]
    chopped_array = torch.empty(
        ((audio_legth - window_size) // hop_size) + 1, window_size
        )
    idx = 0
    for i in range(window_size, audio_legth + 1, hop_size):
        chopped_array[idx, :] = arr[i - window_size:i]
        idx += 1
    return chopped_array


def numpy_linspace(left_bound: float, rigth_bound: float, steps: int):
    if steps <= 1:
        return torch.linspace(left_bound, rigth_bound, steps)
    else:
        return torch.linspace(left_bound, rigth_bound, steps + 1)[:-1]


def create_dct(n_mfcc: int, n_mels: int, norm: str="ortho") -> torch.Tensor:
    """Create a DCT transformation matrix with shape (``n_mels``, ``n_mfcc``),
    normalized depending on norm.
    .. devices:: CPU
    .. properties:: TorchScript
    Args:
        n_mfcc (int): Number of mfc coefficients to retain
        n_mels (int): Number of mel filterbanks
        norm (str or None): Norm to use (either "ortho" or None)
    Returns:
        Tensor: The transformation matrix, to be right-multiplied to
        row-wise data of size (``n_mels``, ``n_mfcc``).
    """

    if norm is not None and norm != "ortho":
        raise ValueError('norm must be either "ortho" or None')

    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = torch.arange(float(n_mels))
    k = torch.arange(float(n_mfcc)).unsqueeze(1)
    dct = torch.cos(math.pi / float(n_mels) * (n + 0.5) * k)  # size (n_mfcc, n_mels)

    if norm is None:
        dct *= 2.0
    else:
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
    return dct.t()


def hertz_to_mels(f: int) -> torch.Tensor:
    return 2595. * torch.log10(torch.tensor(1.) + (f / 700.))


def mel_to_hertz(mel: torch.Tensor) -> torch.Tensor:
    return 700. * (10**(mel / 2595.) - 1.)


def correct_grid(x: torch.Tensor) -> torch.Tensor:
    """Push forward duplicate points to prevent useless filters"""
    offset = 0
    correct_grid = []
    for prev, i in zip((x[0] - 1) + x, x):
        offset = max(0, offset + prev + 1 - i)
        correct_grid.append(i + offset)
    return torch.stack(correct_grid)

class MelSpectrogram(torch.nn.Module):

    def __init__(self, sample_rate: int, window_stride: Tuple[int] = (160, 80),
                 fft_size: int = 512, num_filt:int = 20, return_spectrogram: bool = False):
        super().__init__()
        self.power_spec = PowerSpectrogram(window_stride=window_stride, fft_size=fft_size)
        self.num_filt = num_filt
        self.sample_rate = int(sample_rate)
        self.fft_size = fft_size
        self.return_spectrogram = return_spectrogram

    #@lru_cache()  # Prevents recalculating when calling with same parameters
    def filterbanks(self, fft_len: int):
        """Makes a set of triangle filters focused on {num_filter} mel-spaced frequencies"""

        # Grid contains points for left center and right points of filter triangle
        # mels -> hertz -> fft indices
        grid_mels = torch.linspace(hertz_to_mels(0), hertz_to_mels(self.sample_rate), self.num_filt + 2)
        grid_hertz = mel_to_hertz(grid_mels)
        grid_indices = (grid_hertz * fft_len / self.sample_rate).long()
        #grid_indices = list(correct_grid(grid_indices))
        #grid_indices = torch.tensor(grid_indices)
        grid_indices = correct_grid(grid_indices)

        banks = torch.zeros([int(self.num_filt), int(fft_len)])

        for i, chopped in enumerate(chop_array(grid_indices, 3, 1)):
            left = chopped[0].long().item()
            middle = chopped[1].long().item()
            right = chopped[2].long().item()
            banks[i, left:middle] = numpy_linspace(0., 1., (middle - left))
            banks[i, middle:right] = numpy_linspace(1., 0., (right - middle))
        return banks

    def safe_log(self, x):
        """Prevents error on log(0) or log(-1)"""
        #return torch.log(torch.clip(x, torch.finfo(torch.float32).eps, None))
        return torch.log(torch.clip(x, 1.1920928955078125e-07, None))

    def forward(self, audio) -> Tuple[torch.Tensor, torch.Tensor]:

        # Get batch size
        bs = audio.shape[0]

        # Process Power Spectrogram
        spec = self.power_spec(audio)

        # Initilize result tensor
        mel_spec = torch.empty(bs, spec.shape[1], self.num_filt) # [bs, t, filters]

        # Iterate and get features
        for i in range(bs):
            mel_spec[i, :, :] = self.safe_log(
                torch.matmul(
                    spec[i, :, :].to(audio.device),
                    self.filterbanks(spec[i, :, :].shape[1]).T.to(audio.device)
                    )
                )
        return mel_spec, spec


class PowerSpectrogram(torch.nn.Module):

    def __init__(self, window_stride: Tuple[int] = (160, 80), fft_size: int = 512):
        super().__init__()
        self.window_stride = window_stride
        self.fft_size = fft_size
        self.window = torch.hann_window(window_stride[0])

    def forward(self, audio):

        # Get batch size
        bs = audio.shape[0]

        # Initilize result tensor
        time_bins = int(((audio.shape[1] - self.window_stride[0]) / self.window_stride[1]) + 1)
        frequency_bins = int((self.fft_size / 2) + 1)
        spec = torch.empty(bs, time_bins, frequency_bins) # [bs, t, f]

        # Iterate and get features
        for i in range(bs):
            frames = chop_array(audio[i, :], self.window_stride[0], self.window_stride[1])
            fft = torch.fft.rfft(frames*self.window, n=self.fft_size)
            spec[i, :, :] = (fft.real**2 + fft.imag**2) / self.fft_size
        
        return spec
        

class MFCC(torch.nn.Module):

    def __init__(self, sample_rate: int, window_stride: Tuple[int] = (160, 80), 
                 fft_size: int = 512, num_filt:int = 20,  num_coeffs: int = 13,
                 remove_zero_coefficient:bool = False, energy_deltas:bool = False):
        super().__init__()
        self.remove_zero_coefficient = remove_zero_coefficient
        self.energy_deltas = energy_deltas
        self.power_spec = MelSpectrogram(
            sample_rate=sample_rate,
            window_stride=window_stride,
            fft_size=fft_size,
            num_filt=num_filt,
            return_spectrogram=True
            )
        self.dct_mat = create_dct(num_coeffs, num_filt, norm='ortho')

    def safe_log(self, x):
        """Prevents error on log(0) or log(-1)"""
        #return torch.log(torch.clip(x, torch.finfo(float).eps, None))
        return torch.log(torch.clip(x, 1.1920928955078125e-07, None))

    def forward(self, audio):
        # Get mel-spectrogram
        mels_and_spec = self.power_spec(audio)
        mels = mels_and_spec[0]
        spec = mels_and_spec[1]
        mfccs = torch.matmul(
            mels.to(audio.device),
            self.dct_mat.to(audio.device)
            ) # apply the DCT transform
        if self.remove_zero_coefficient:
            mfccs =  mfccs[:, :, 1:]
        else:    
            mfccs[:, :, 0] = self.safe_log(torch.sum(spec, 2))  # Replace first band with log energies
            # claculate delta of log energies
            if self.energy_deltas:
                deltas = torch.diff(mfccs[:, :, 0])
                mfccs[:, :, 0] = torch.cat((deltas[:,0].unsqueeze(1), deltas), 1)
        return mfccs