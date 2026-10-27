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
import numpy as np

import torchaudio
from sonopy import mfcc_spec

import utils.sonopytorch as sonopytorch


class FeatureExtractor(torch.nn.Module):
    """
    Class to handle audio feature extraction
    """
    def __init__(self, device, window_size_s, fs=16000, features='mfccs_sonopy', transpose=False):
        super().__init__()
        
        self.device = device
        self.window_size = int(window_size_s * fs)
        self.fs = fs
        self.transpose = transpose
        self.features = features

        # Raw audio
        if (features == 'raw_audio'):
            self.output_shape = (self.window_size, )
            self.extract_features = self.__raw_audio

        # TORCHAUDIO
        # NOTE: if log is not applied, dynamic range is too big
        elif (features == 'spectrogram_torchaudio'):
            self.kwargs ={
                'n_fft':256,
                'win_length':256,
                'hop_length':128,
                'normalized':True,
            }
            self.extract_features = torchaudio.transforms.Spectrogram(
                **self.kwargs
            ).to(self.device)
            time_bins = self.__get_time_bins(self.kwargs['win_length'], self.kwargs['hop_length']) # 186
            frequency_bins = self.__get_frequency_bins(self.kwargs['n_fft']) # 129
            self.output_shape = (frequency_bins, time_bins + 2) # [freq_b, time_b]

        # log Mel-spectrogram
        # NOTE: log energies is what makes sense, otherwise
        # the differences in therms of enery are too big
        elif (features == 'melspectrogram_torchaudio'):
            self.n_mels = 40
            epsilon = 0.0001
            self.melkwargs = {
                'n_fft':512,
                'win_length':320,
                'hop_length':160,
                'f_min':0.0,
                'f_max':8000,
                'pad':0,
                'n_mels':self.n_mels,
            }
            time_bins = self.__get_time_bins(self.melkwargs['win_length'], self.melkwargs['hop_length']) # 149
            self.output_shape = (self.n_mels, time_bins + 2)
            mels = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.fs,
                **self.melkwargs
            ).to(self.device)
            self.extract_features = lambda x: torch.log(mels(x)+epsilon)

        elif (features == 'mfccs_torchaudio'):
            self.n_mels = 20
            self.n_coef = 13
            self.melkwargs = {
                'n_fft':2048,
                'win_length':1600,
                'hop_length':800,
                'f_min':0.0,
                'f_max':8000,
                'pad':0,
                'n_mels':self.n_mels,
            }
            self.extract_features = torchaudio.transforms.MFCC(
                sample_rate=self.fs,
                n_mfcc=self.n_coef,
                dct_type=2,
                norm='ortho',
                log_mels=False,
                melkwargs=self.melkwargs
            ).to(self.device)
            time_bins = self.__get_time_bins(self.melkwargs['win_length'], self.melkwargs['hop_length']) # 29
            self.output_shape = (self.n_coef, time_bins)

        # SONOPY
        elif (features == 'mfccs_sonopy'):
            self.window_s = 1600
            self.hop_s = 800
            self.n_coef = 13
            self.mel_fil = 20
            self.fft_size = 512
            self.extract_features = self.__mfccs_sonopy
            time_bins = self.__get_time_bins(self.window_s, self.hop_s) # 29
            self.output_shape = (time_bins, self.n_coef)
        
        # SONOPYTORCH
        elif features == "mfccs_sonopytorch_13_w20_h10":
            self.window_s = 320
            self.hop_s = 160
            self.n_coef = 13
            self.mel_fil = 20
            self.fft_size = 512
            self.extract_features = sonopytorch.MFCC(
                sample_rate=self.fs,
                window_stride=(self.window_s, self.hop_s),
                num_filt=self.mel_fil,
                fft_size=self.fft_size,
                num_coeffs=self.n_coef,
                energy_deltas=False,
                remove_zero_coefficient=False
            )
            time_bins = self.__get_time_bins(self.window_s, self.hop_s)
            self.output_shape = (time_bins, self.n_coef)


        elif features == "mfccs_sonopytorch_13_w30_h10":
            self.window_s = 480
            self.hop_s = 160
            self.n_coef = 13
            self.mel_fil = 20
            self.fft_size = 512
            self.extract_features = sonopytorch.MFCC(
                sample_rate=self.fs,
                window_stride=(self.window_s, self.hop_s),
                num_filt=self.mel_fil,
                fft_size=self.fft_size,
                num_coeffs=self.n_coef,
                energy_deltas=False,
                remove_zero_coefficient=False
            )
            time_bins = self.__get_time_bins(self.window_s, self.hop_s)
            self.output_shape = (time_bins, self.n_coef)

        elif features == "mfccs_sonopytorch_13_w100_h20":
            self.window_s = 1600
            self.hop_s = 320
            self.n_coef = 13
            self.mel_fil = 40
            self.fft_size = 512
            self.extract_features = sonopytorch.MFCC(
                sample_rate=self.fs,
                window_stride=(self.window_s, self.hop_s),
                num_filt=self.mel_fil,
                fft_size=self.fft_size,
                num_coeffs=self.n_coef,
                energy_deltas=False,
                remove_zero_coefficient=False
            )
            time_bins = self.__get_time_bins(self.window_s, self.hop_s)
            self.output_shape = (time_bins, self.n_coef)

        elif features == "mfccs_sonopytorch_13_w100_h50":
            self.window_s = 1600
            self.hop_s = 800
            self.n_coef = 13
            self.mel_fil = 40
            self.fft_size = 512
            self.extract_features = sonopytorch.MFCC(
                sample_rate=self.fs,
                window_stride=(self.window_s, self.hop_s),
                num_filt=self.mel_fil,
                fft_size=self.fft_size,
                num_coeffs=self.n_coef,
                energy_deltas=False,
                remove_zero_coefficient=False
            )
            time_bins = self.__get_time_bins(self.window_s, self.hop_s)
            self.output_shape = (time_bins, self.n_coef)

        elif features == "mfccs_sonopytorch_20_w100_h50":
            self.window_s = 1600
            self.hop_s = 800
            self.n_coef = 20
            self.mel_fil = 40
            self.fft_size = 512
            self.extract_features = sonopytorch.MFCC(
                sample_rate=self.fs,
                window_stride=(self.window_s, self.hop_s),
                num_filt=self.mel_fil,
                fft_size=self.fft_size,
                num_coeffs=self.n_coef,
                energy_deltas=False,
                remove_zero_coefficient=False
            )
            time_bins = self.__get_time_bins(self.window_s, self.hop_s)
            self.output_shape = (time_bins, self.n_coef)

        elif features == "mfccs_sonopytorch_40_w100_h50":
            self.window_s = 1600
            self.hop_s = 800
            self.n_coef = 40
            self.mel_fil = 40
            self.fft_size = 512
            self.extract_features = sonopytorch.MFCC(
                sample_rate=self.fs,
                window_stride=(self.window_s, self.hop_s),
                num_filt=self.mel_fil,
                fft_size=self.fft_size,
                num_coeffs=self.n_coef,
                energy_deltas=False,
                remove_zero_coefficient=False
            )
            time_bins = self.__get_time_bins(self.window_s, self.hop_s)
            self.output_shape = (time_bins, self.n_coef)

        elif features == "mfccs_sonopytorch_40_w30_h10":
            self.window_s = 480
            self.hop_s = 160
            self.n_coef = 40
            self.mel_fil = 40
            self.fft_size = 512
            self.extract_features = sonopytorch.MFCC(
                sample_rate=self.fs,
                window_stride=(self.window_s, self.hop_s),
                num_filt=self.mel_fil,
                fft_size=self.fft_size,
                num_coeffs=self.n_coef,
                energy_deltas=False,
                remove_zero_coefficient=False
            )
            time_bins = self.__get_time_bins(self.window_s, self.hop_s)
            self.output_shape = (time_bins, self.n_coef)

        elif features == "mfccs_sonopytorch_40_w20_h10":
            self.window_s = 360
            self.hop_s = 160
            self.n_coef = 40
            self.mel_fil = 40
            self.fft_size = 512
            self.extract_features = sonopytorch.MFCC(
                sample_rate=self.fs,
                window_stride=(self.window_s, self.hop_s),
                num_filt=self.mel_fil,
                fft_size=self.fft_size,
                num_coeffs=self.n_coef,
                energy_deltas=False,
                remove_zero_coefficient=False
            )
            time_bins = self.__get_time_bins(self.window_s, self.hop_s)
            self.output_shape = (time_bins, self.n_coef)

        if(self.transpose):
            self.output_shape = self.output_shape[::-1]

    def __mfccs_sonopy(self, signal):
        mfccs = mfcc_spec(signal,
                          self.fs,
                          (self.window_s, self.hop_s),
                          num_filt=self.mel_fil,
                          fft_size=self.fft_size,
                          num_coeffs=self.n_coef
                          )
        return mfccs

    def __raw_audio(self, signal):
        return signal

    def __get_time_bins(self, win, hop):
        return int((self.window_size - win)/hop + 1)

    def __get_frequency_bins(self, fft_size):
        return int((fft_size / 2) + 1)

    def forward(self, x):
        """
        Extracts audio features.
        Preconditions:
            It is assumed that the audio clips is of 24000 samples. 
        """
        # If we use torchaudio or sonopytorch: process the whole batch
        if (('torchaudio' in self.features) or ('raw_audio' in self.features) or
            ('sonopytorch' in self.features)):
            x = self.extract_features(x)

        # If we use other tool: process one by one
        # Problem: this does detach gradients
        else:
            results = np.zeros((x.shape[0], ) + self.output_shape)
            for idx in range(x.shape[0]):
                results[idx, :] = self.extract_features(x[idx, :].data.cpu().numpy())
            x = torch.from_numpy(results)
            x.to(self.device)
        
        if(self.transpose):
            x = x.transpose(2, 1)

        return x
