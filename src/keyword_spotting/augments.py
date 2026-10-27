# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of EdgeWuW.
#
# EdgeWuW is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# EdgeWuW is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with EdgeWuW. If not, see https://www.gnu.org/licenses/

import math
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import torchaudio.transforms as T

from utils import dsp, vision_torch_augments

from torch.distributions.beta import Beta


class WhiteNoise(nn.Module):
    """Additive White Gaussian (Normal distribution) Noise"""

    def __init__(self, min_snr_db=20, max_snr_db=40, p=0.0):
        super().__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p

    def forward(self, x):

        if self.p < random.random():
            return x
        
        snr_db = np.random.uniform(self.min_snr_db, self.max_snr_db)
        awgn = torch.randn(x.shape).to(x.device)
        awgn_rms = awgn.std()
        x_rms = torch.sqrt(torch.mean(x**2))
        desired_awgn_rms = torch.sqrt(x_rms**2 / math.pow(10, snr_db/10))
        k = desired_awgn_rms / awgn_rms
        x = x + k*awgn
        return x/torch.max(torch.abs(x))


class Fade(nn.Module):
    """Fade."""

    def __init__(self, fade_in=0, fade_out=0, fade_shape:str= 'linear', p=0.0):
        super().__init__()
        self.p = p
        self.fade = T.Fade(
            fade_in_len=fade_in,
            fade_out_len=fade_out,
            fade_shape=fade_shape
            )

    def forward(self, x):

        if self.p < random.random():
            return x
        
        return self.fade(x)


class FrequencyMasking(torch.nn.Module):
    """Applies frequency masking to the input audio."""
    
    def __init__(self, n_fft=400, hop_length=100, freq_mask_param=30, n_freq_masks=2, p=0.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_mask_param = freq_mask_param
        self.n_freq_masks = n_freq_masks
        self.p = p

    def forward(self, audio):
        
        if self.p < random.random():
            return audio

        # Convert to spectrogram
        audio_spec = torch.stft(audio, self.n_fft, hop_length=self.hop_length, return_complex=True)

        # Separate magnitude and phase
        magnitude = audio_spec.abs()
        phase = audio_spec.angle()

        # Apply frequency masking on the magnitude
        for _ in range(self.n_freq_masks):
            freq_mask = T.FrequencyMasking(self.freq_mask_param)
            magnitude = freq_mask(magnitude)

        # Recombine magnitude and phase
        audio_spec = torch.polar(magnitude, phase)

        # Convert back to time
        audio = torch.istft(audio_spec, self.n_fft, hop_length=self.hop_length, length=audio.shape[-1])
        return audio


class TimeMasking(torch.nn.Module):
    """Applies time masking to the input audio."""
    
    def __init__(self, n_fft=400, hop_length=100, time_mask_param=40, n_time_masks=2, p=0.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.time_mask_param = time_mask_param
        self.n_time_masks = n_time_masks
        self.p = p

    def forward(self, audio):

        if self.p < random.random():
            return audio

        # Convert to spectrogram
        audio_spec = torch.stft(audio, self.n_fft, hop_length=self.hop_length, return_complex=True)

        # Separate magnitude and phase
        magnitude = audio_spec.abs()
        phase = audio_spec.angle()

        # Apply time masking on the magnitude
        for _ in range(self.n_time_masks):
            time_mask = T.TimeMasking(self.time_mask_param)
            magnitude = time_mask(magnitude)

        # Recombine magnitude and phase
        audio_spec = torch.polar(magnitude, phase)

        # Convert back to time
        audio = torch.istft(audio_spec, self.n_fft, hop_length=self.hop_length, length=audio.shape[-1])
        return audio


class TimeStretching(torch.nn.Module):
    """Applies time stretching to the input audio."""
    
    def __init__(self, n_fft=400, hop_length=100, target_size=16000, eps=0.0001, p=0.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_size = target_size
        self.eps = eps
        self.p = p


    def forward(self, audio):

        if self.p < random.random():
            return audio
        
        # Convert to spectrogram
        audio_spec = torch.stft(audio, self.n_fft, hop_length=self.hop_length, return_complex=True)

        # Apply time stretch transformation
        time_stretch = T.TimeStretch(hop_length=self.hop_length, n_freq=audio_spec.shape[1])
        stretched_spec = time_stretch(audio_spec, audio.shape[1] / self.target_size + self.eps)

        # Convert back to time
        audio = torch.istft(stretched_spec, self.n_fft, hop_length=self.hop_length)
        
                # Ensure the final audio is the correct length (trim or pad as necessary)
        if audio.shape[1] < self.target_size:
            # Pad the mixed audio if it's shorter than the target size
            padding = self.target_size - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        else:
            # Trim the mixed audio if it's longer than the target size
            audio = audio[:, :self.target_size]

        return audio


def to_soft_labels(labels, num_classes):
    """Converts class indices [batch] to one-hot targets [batch, num_classes]."""
    if labels.dim() == 1:
        return F.one_hot(labels.long(), num_classes).float()
    return labels.float()


def select_mix_indices(batch_size, perc_real_samples, p, device):
    """
    Selects the samples of a batch to be mixed: a fraction perc_real_samples of the
    batch is kept unmodified and each remaining sample is mixed with probability p.
    """
    n_candidates = batch_size - int(perc_real_samples * batch_size)
    candidates = torch.randperm(batch_size, device=device)[:n_candidates]
    return candidates[torch.rand(n_candidates, device=device) < p]


class MixUp(nn.Module):
    """
    MixUp (Zhang et al., 2018): replaces samples of the batch by x = lam*x1 + (1-lam)*x2
    and y = lam*y1 + (1-lam)*y2, with x2 a random sample of the batch and lam ~ Beta(alpha, alpha).
    Mixing the complex spectrograms gives the same result, as the STFT is linear.
    Returns the audio batch and soft targets [batch, num_classes].
    """
    def __init__(self, alpha=0.4, p=0.0, perc_real_samples=0.5, num_classes=2):
        super().__init__()
        self.alpha = alpha
        self.p = p
        self.perc_real_samples = perc_real_samples
        self.num_classes = num_classes

    def forward(self, x, labels):
        y = to_soft_labels(labels, self.num_classes)
        mix_idx = select_mix_indices(x.size(0), self.perc_real_samples, self.p, x.device)
        if mix_idx.numel() == 0:
            return x, y

        partner_idx = torch.randint(0, x.size(0), (mix_idx.numel(), ), device=x.device)
        lam = Beta(self.alpha, self.alpha).sample((mix_idx.numel(), )).to(x.device)

        mixed_x = x.clone()
        lam_x = lam.view(-1, *([1] * (x.dim() - 1)))
        mixed_x[mix_idx] = lam_x * x[mix_idx] + (1 - lam_x) * x[partner_idx]

        mixed_y = y.clone()
        mix_idx, partner_idx, lam = mix_idx.to(y.device), partner_idx.to(y.device), lam.to(y.device)
        mixed_y[mix_idx] = lam.unsqueeze(1) * y[mix_idx] + (1 - lam.unsqueeze(1)) * y[partner_idx]
        return mixed_x, mixed_y


class SpecMix(nn.Module):
    """
    SpecMix (Kim et al., 2021): replaces random frequency bands and time segments of the
    spectrogram of a sample by those of another random sample of the batch. The targets are
    mixed in proportion to the replaced area.
    Returns the audio batch and soft targets [batch, num_classes].
    """
    def __init__(self, n_fft=400, hop_length=100, freq_mask_param=30, time_mask_param=40,
                 n_freq_masks=2, n_time_masks=2, p=0.0, perc_real_samples=0.5, num_classes=2):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p
        self.perc_real_samples = perc_real_samples
        self.num_classes = num_classes

    def get_mask(self, n_freqs, n_frames, device):
        """Mask with ones in the regions taken from the other sample."""
        mask = torch.zeros(n_freqs, n_frames, device=device)
        for _ in range(self.n_freq_masks):
            width = random.randint(0, min(self.freq_mask_param, n_freqs))
            start = random.randint(0, n_freqs - width)
            mask[start:start + width, :] = 1.0
        for _ in range(self.n_time_masks):
            width = random.randint(0, min(self.time_mask_param, n_frames))
            start = random.randint(0, n_frames - width)
            mask[:, start:start + width] = 1.0
        return mask

    def forward(self, x, labels):
        y = to_soft_labels(labels, self.num_classes)
        mix_idx = select_mix_indices(x.size(0), self.perc_real_samples, self.p, x.device)
        if mix_idx.numel() == 0:
            return x, y

        partner_idx = torch.randint(0, x.size(0), (mix_idx.numel(), ), device=x.device)
        spec = torch.stft(x[mix_idx], self.n_fft, hop_length=self.hop_length, return_complex=True)
        partner_spec = torch.stft(x[partner_idx], self.n_fft, hop_length=self.hop_length, return_complex=True)

        masks = torch.stack([self.get_mask(spec.shape[1], spec.shape[2], x.device) for _ in range(mix_idx.numel())])
        spec = spec * (1 - masks) + partner_spec * masks

        mixed_x = x.clone()
        mixed_x[mix_idx] = torch.istft(spec, self.n_fft, hop_length=self.hop_length, length=x.shape[-1])

        # Proportion of each sample taken from the other one
        lam = masks.mean(dim=(1, 2)).to(y.device).unsqueeze(1)
        mixed_y = y.clone()
        mix_idx, partner_idx = mix_idx.to(y.device), partner_idx.to(y.device)
        mixed_y[mix_idx] = (1 - lam) * y[mix_idx] + lam * y[partner_idx]
        return mixed_x, mixed_y


class AudioRandAugment(nn.Module):
    """Applies RandAugment-like augmentations to audio after converting it to a spectrogram.
       These augmentations are the same as for images, treating the spectrogram as an image.
    """
    
    def __init__(self, n_fft=400, hop_length=100, num_ops=2, magnitude=9, p=0.0, target_size=16000, num_magnitude_bins=31):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.p = p
        self.target_size = target_size
        self.num_magnitude_bins = num_magnitude_bins

        # Define the available transformations
        self.augmentations = [
            self._rotate,
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y,
            self._adjust_brightness,
            self._adjust_contrast,
            self._adjust_saturation
        ]

    def _rotate(self, spec, magnitude):
        degrees = (magnitude / self.num_magnitude_bins) * 30  
        return vision_torch_augments.rotate(spec, degrees)

    def _shear_x(self, spec, magnitude):
        shear = (magnitude / self.num_magnitude_bins) * 20  
        return vision_torch_augments.affine(spec, angle=0, translate=[0, 0], scale=1.0, shear=[shear, 0])

    def _shear_y(self, spec, magnitude):
        shear = (magnitude / self.num_magnitude_bins) * 20  
        return vision_torch_augments.affine(spec, angle=0, translate=[0, 0], scale=1.0, shear=[0, shear])

    def _translate_x(self, spec, magnitude):
        max_translate = spec.size(2) // 3  
        translate = int((magnitude / self.num_magnitude_bins) * max_translate)
        return vision_torch_augments.affine(spec, angle=0, translate=[translate, 0], scale=1.0, shear=[0, 0])

    def _translate_y(self, spec, magnitude):
        max_translate = spec.size(1) // 3  
        translate = int((magnitude / self.num_magnitude_bins) * max_translate)
        return vision_torch_augments.affine(spec, angle=0, translate=[0, translate], scale=1.0, shear=[0, 0])

    def _adjust_brightness(self, spec, magnitude):
        factor = 1.0 + (magnitude / self.num_magnitude_bins) * 0.9  
        return vision_torch_augments.adjust_brightness(spec, factor)

    def _adjust_contrast(self, spec, magnitude):
        factor = 1.0 + (magnitude / self.num_magnitude_bins) * 0.9  
        return vision_torch_augments.adjust_contrast(spec, factor)

    def _adjust_saturation(self, spec, magnitude):
        factor = 1.0 + (magnitude / self.num_magnitude_bins) * 0.9  
        return vision_torch_augments.adjust_saturation(spec, factor)

    def forward(self, audio):
        if random.random() > self.p:
            return audio
        
        audio_spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        magnitude_spec = audio_spec.abs() 
        for _ in range(self.num_ops):
            op = random.choice(self.augmentations)  
            magnitude_spec = op(magnitude_spec, self.magnitude)  
        phase = audio_spec.angle()
        augmented_spec = torch.polar(magnitude_spec, phase)
        augmented_audio = torch.istft(augmented_spec, n_fft=self.n_fft, hop_length=self.hop_length)
        if augmented_audio.shape[1] < self.target_size:
            padding = self.target_size - augmented_audio.shape[1]
            augmented_audio = F.pad(augmented_audio, (0, padding))
        else:
            augmented_audio = augmented_audio[:, :self.target_size]

        return augmented_audio


class Convolve(nn.Module):
    """Convolve audio with Impulsive Response"""

    def __init__(self):
        super().__init__()

    def forward(self, x, rir):
        return dsp.reverberate(x, rir)


class RandomFilters(nn.Module):
    """Random Filters
    This module generates random filters to apply to audio signals.
    We generate a cocrete number of filters that 
    """

    def __init__(self, sample_rate=16000, gain_db_range=(-6, 6), num_bands=10):
        super().__init__()
        self.sample_rate = sample_rate
        self.gain_db_range = gain_db_range
        self.num_bands = num_bands

    def generate_filter(self):
        # Create a frequency response with higher resolution
        freqs = torch.fft.fftfreq(self.sample_rate)
        response = torch.ones_like(freqs)
        
        # Generate random gain points and interpolate
        random_points = torch.linspace(0, len(freqs) - 1, self.num_bands).long()
        random_gains_db = torch.tensor([random.uniform(*self.gain_db_range) for _ in range(self.num_bands)])
        random_gains_linear = 10 ** (random_gains_db / 20)  # Convert dB to linear scale
        
        # Interpolate the random gains to create a smooth response
        interpolated_gains = torch.interp(torch.arange(len(freqs)), random_points, random_gains_linear)
        response *= interpolated_gains
        
        return response

    def forward(self, audio):
        
        # Process batches of audio
        # Add a probability parameter to control when to apply the filter
        # Use an argument to control the probability of applying the filter
        
        if audio.dim() != 1:
            raise ValueError("Audio input should be a 1D tensor")
        
        # FFT of the audio signal
        audio_fft = torch.fft.fft(audio)
        
        # Generate the filter
        filter_response = self.generate_filter()
        
        # Apply the filter in the frequency domain
        filtered_audio_fft = audio_fft * filter_response
        
        # Inverse FFT to get back to time domain
        filtered_audio = torch.fft.ifft(filtered_audio_fft).real
        
        return filtered_audio


class FilterAugment(nn.Module):
    """FilterAugment
    Source: https://seunghan96.github.io/audio/ts/cl/(audio_paper27)FilterAugment/
    This applies linear or step filters to audio signals.
    """
    def __init__(self, db_range=(-6, 6), n_band=(3, 9), min_bw=6, filter_type="linear", p=0.0, n_fft=512, hop_length=160):
        super().__init__()
        # Probability to apply the filter
        self.p = p

        # Spectrogram parameters
        self.n_fft = n_fft
        self.hop_length = hop_length

        valid_filter_type = {"linear", "step"}
        if filter_type not in valid_filter_type:
            raise ValueError("Filter type must be one of %r." % valid_filter_type)
        
        # Check that low n_band is less than high n_band and bigger than 1
        if n_band[0] >= n_band[1] or n_band[0] <= 1:
            raise ValueError("n_band must be a tuple with the first element less than the second element and bigger than 1.")

        # Augment specific parameters
        self.db_range = db_range
        self.n_band = n_band # [low, high)
        self.min_bw = min_bw
        self.filter_type = filter_type

    def forward(self, x):
        if random.random() > self.p:
            return x
        
        length = x.shape[-1]
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        x_mag = torch.abs(x)
        x_phase = torch.angle(x)
        augment_filter = self.get_filter(x_mag)
        augment_filter = augment_filter.repeat(1, 1, x_mag.shape[-1])
        x = x_mag * augment_filter
        x = torch.polar(x, x_phase)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, length=length)
        # Apply peak normalization to the audio
        v,_ = torch.max(torch.abs(x), dim=1, keepdim=True)
        x = x / torch.clamp(v, min=1e-8)
        return x

    def get_filter(self, x):
        bs, freq_bins, _ = x.shape
        freq_bands = torch.randint(
            low=self.n_band[0],
            high=self.n_band[1],
            size=(1,)
            ).item()
        
        while freq_bins - freq_bands * self.min_bw + 1 < 0:
            self.min_bw -= 1
        
        # NOTE: in the source implementation, they work with mel-spectrograms
        # They pick filtered bands, here we work with linear spectrograms
        # We will pick random bands and apply the filter to the whole spectrogram
        # Initialize the bands array with minimum bandwidths
        bands = [self.min_bw] * (freq_bands)

        # Calculate the remaining bins after allocating minimum bandwidth to each band
        remaining_bins = freq_bins - sum(bands)
        
        # Ensure that a number of bands remain at minimum width by randomly choosing them
        num_min_bw_bands = max(1, freq_bands // 3)  # Adjust this ratio as necessary for more or fewer minimum bands
        min_bw_indices = random.sample(range(freq_bands), num_min_bw_bands)

        # Distribute any remaining bins randomly to bands not locked at min_bw
        while remaining_bins > 0:
            idx = random.choice([i for i in range(freq_bands) if i not in min_bw_indices])
            bands[idx] += 1
            remaining_bins -= 1

        band_bndry_freqs = torch.cumsum(torch.LongTensor(bands), dim=0)
        band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs))

        if self.filter_type == "step":
            band_factors = torch.rand((bs, freq_bands)).to(x) * (self.db_range[1] - self.db_range[0]) + self.db_range[0]
            band_factors = 10 ** (band_factors / 20)

            freq_filt = torch.ones((bs, freq_bins, 1)).to(x)
            for i in range(freq_bands):
                freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = band_factors[:, i].unsqueeze(-1).unsqueeze(-1)

        elif self.filter_type == "linear":
            band_factors = torch.rand((bs, freq_bands + 1)).to(x) * (self.db_range[1] - self.db_range[0]) + self.db_range[0]
            freq_filt = torch.ones((bs, freq_bins, 1)).to(x)
            for i in range(freq_bands):
                for j in range(bs):
                    freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                        torch.linspace(band_factors[j, i], band_factors[j, i+1],
                                    band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1)
            freq_filt = 10 ** (freq_filt / 20)
        return freq_filt
    

class FreqMixStyle(nn.Module):
    """Freq-MixStyle
    Source: ohttps://github.com/fschmid56/cpjku_dcase23/blob/main/helpers/utils.py
    This applies MixStyle augmentation to audio signals.
    """

    def __init__(self, p=0.0, alpha=0.4, freq_dim=-1):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = 1e-6
        
        # Frequency dimension depends on feature extraction
        # We will assume that the inputS are MFCCs with shape [batch_size, n_frames, n_mfcc]
        # Thus, the dimension that contains the frequency information is the last one
        self.freq_dim = freq_dim

    def forward(self, x, labels):
        if random.random() > self.p:
            return x
        
        # [bs, t_dim, f_dim]
        # The data augment is applied based on label information
        unique_labels = torch.unique(labels)

        mixed_x = torch.zeros_like(x)
        for label in unique_labels:
            class_idx = labels == label
            x_class = x[class_idx]

            if x_class.size(0) < 2:
                mixed_x[class_idx] = x_class
                continue

            f_mu = x_class.mean(dim=[1, 2], keepdim=True)
            f_var = x_class.var(dim=[1, 2], keepdim=True)
            f_sig = (f_var + self.eps).sqrt()
            f_mu, f_sig = f_mu.detach(), f_sig.detach()

            x_normed = (x_class - f_mu) / f_sig

            lmda = Beta(self.alpha, self.alpha).sample((x_class.size(0), 1, 1)).to(x.device)

            perm = torch.randperm(x_class.size(0)).to(x.device)
            f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]

            mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)
            sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)

            x_mixed = x_normed * sig_mix + mu_mix

            v, _ = torch.max(torch.abs(x_mixed), dim=1, keepdim=True)
            x_mixed = x_mixed / torch.clamp(v, min=1e-8)

            mixed_x[class_idx] = x_mixed

        return mixed_x
