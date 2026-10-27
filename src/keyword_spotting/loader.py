# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of EdgeWuW.
#
# EdgeWuW is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# EdgeWuW is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with EdgeWuW. If not, see https://www.gnu.org/licenses/

import os
import math
import json
import random

import torch
import numpy as np
import pandas as pd
import torchaudio

from keyword_spotting.augments import Convolve
from utils.audio_utils import trim_silence


def load_train_partitions(path, window_size=24000, fs=16000, bn_prob=0.0, rir_prob=0.0, dir_prob=0.0,
    non_speech_prob=0.0, snr_range=[5, 30], random_crop=True, time_stretch=False, se_model=None):

    # Read class index
    with open(path + 'classes_index.json') as f:
        class_index = json.load(f)
    
    # Load train set
    train_df = pd.read_csv(path + 'train.tsv', sep='\t')
    train_noise_df = train_df[(train_df['Sound_Type'] != 'speech') & (train_df['Sound_Type'] != 'rir')]
    train_rirs_df = train_df[train_df['Sound_Type'] == 'rir']
    train_dirs_df = train_df[train_df['Sound_Type'] == 'dir']
    train_df = train_df[train_df['Sound_Type'] == 'speech']

    # Load validation set
    validation_df = pd.read_csv(path + 'dev.tsv', sep='\t')
    validation_noise_df = validation_df[(validation_df['Sound_Type'] != 'speech') & (validation_df['Sound_Type'] != 'rir')]
    validation_rirs_df = validation_df[validation_df['Sound_Type'] == 'rir']
    validation_dirs_df = validation_df[validation_df['Sound_Type'] == 'dir']
    validation_df = validation_df[validation_df['Sound_Type'] == 'speech']

    # Generate Datasets
    train_dataset = AudioDataset(
        train_df,
        class_index,
        window_size=window_size,
        fs=fs,
        bn_prob=bn_prob,
        non_speech_prob=non_speech_prob,
        rir_prob=rir_prob,
        rirs_df=train_rirs_df,
        dir_prob=dir_prob,
        dirs_df=train_dirs_df,
        noise_df=train_noise_df,
        snr_range=snr_range,
        random_crop=random_crop,
        time_stretch=time_stretch,
        se_model=se_model
        )
    
    # NOTE: I am adding some augmentations to the validation set
    # I believe they are relevant for the model to generalize better
    # Cause I am introducing some noise and non-speech data
    # Also, I am augmenting just a small part of the data
    validation_dataset = AudioDataset(
        validation_df,
        class_index,
        window_size=window_size,
        fs=fs,
        bn_prob=0.2,
        non_speech_prob=0.1,
        rir_prob=0.0,
        rirs_df=validation_rirs_df,
        dir_prob=0.0,
        dirs_df=validation_dirs_df,
        noise_df=validation_noise_df,
        snr_range=snr_range,
        random_crop=random_crop,
        time_stretch=False,
        se_model=se_model
        )

    return train_dataset, validation_dataset

def load_test_partition(path, window_size=24000, fs=16000, bn_prob=0.0, rir_prob=0.0, dir_prob=0.0,
    non_speech_prob=0.0, snr_range=[5, 30], random_crop=True, time_stretch=False, se_model=None, dataset_name='test.tsv'):

    # Read class index
    with open(path + 'classes_index.json') as f:
        class_index = json.load(f)

    # Load test set
    test_df = pd.read_csv(os.path.join(path, dataset_name), header=0, sep='\t')
    print("Using test partition: ", dataset_name)
    
    test_noise_df = test_df[(test_df['Sound_Type'] != 'speech') & (test_df['Sound_Type'] != 'rir')]
    test_rirs_df = test_df[test_df['Sound_Type'] == 'rir']
    test_dirs_df = test_df[test_df['Sound_Type'] == 'dir']
    test_df = test_df[test_df['Sound_Type'] == 'speech']

    # Generate Dataset
    test_dataset = AudioDataset(
        test_df,
        class_index,
        window_size=window_size,
        fs=fs,
        bn_prob=bn_prob,
        non_speech_prob=non_speech_prob,
        rir_prob=rir_prob,
        rirs_df=test_rirs_df,
        dir_prob=dir_prob,
        dirs_df=test_dirs_df,
        noise_df=test_noise_df,
        snr_range=snr_range,
        random_crop=random_crop,
        time_stretch=time_stretch,
        se_model=se_model
        )

    return test_dataset


class AudioDataset(torch.utils.data.Dataset):
    """
    Torch dataset for lazy load.
    """
    def __init__(self, dataframe, class_index, window_size=24000, fs=16000, bn_prob=0.0,
                 rir_prob=0.0, rirs_df=None, dir_prob=0.0, dirs_df=None, non_speech_prob=0.0,
                 noise_df=None, snr_range=[5, 30], random_crop=True, time_stretch=False, se_model=None):
        
        self.window_size = window_size # samples
        self.fs = fs # Hz

        # Data information
        self.list_IDs = list(dataframe['Sample_ID'])
        self.labels = list(dataframe['Label'])
        self.dataframe = dataframe
        unique = list(self.dataframe['Label'].unique())
        self.n_classes = len(unique)
        self.class_index = {y:x for x,y in class_index.items()}

        # Add background sound: can be an empty list
        self.bn_prob = bn_prob
        self.snr_range = snr_range
        self.noise_df = noise_df
        self.noise_paths = noise_df['Sample_Path'].tolist()

        # Non speech data as negative samples
        self.non_speech_prob = non_speech_prob

        # Add IRs: can be an empty list
        self.rir_prob = rir_prob
        self.rirs_df = rirs_df
        self.rirs_paths = rirs_df['Sample_Path'].tolist()

        self.dir_prob = dir_prob
        self.dirs_df = dirs_df
        self.dirs_paths = dirs_df['Sample_Path'].tolist()

        self.convolve = Convolve()

        # Random crop:
        self.random_crop = random_crop

        # Time Stretching Augment
        self.time_stretch = time_stretch

        # Speech Enhancement
        self.se_model = se_model

    def __len__(self):
        """
        Denote dataset sample.
        """
        return len(self.list_IDs)

    def __repr__(self):
        """
        Data infromation string.
        """
        # Get all databases
        databases_list = self.dataframe['Database'].unique().tolist()
        databases_list += self.noise_df['Database'].unique().tolist()
        databases_list += self.rirs_df['Database'].unique().tolist()
        databases_list += self.dirs_df['Database'].unique().tolist()
        databases_list = np.unique(databases_list)
        repr_str = (
            "Dataset information:\n"
            "\tClasses: " + str(list(self.class_index.keys())) + "\n"
            "\tWindow size: " + str(self.window_size) + "\n"
            "\tSampling Rate: " + str(self.fs) + "\n"
            "\tNumber of samples: " + str(len(self.labels)) + "\n"
            "\tRandom crop on negative samples: " + str(self.random_crop) + "\n"
            "\tDatabases: " + str(databases_list) + "\n"
        )
        return repr_str
        
    def __getitem__(self, index):
        """
        Get a single sample.

        Args:
            index: index to recover a single sample
        Returns:
            x,y: features extracted and label
        """

        # Select sample
        ID = self.list_IDs[index]
        y = int(self.class_index[self.labels[index]])

        # If empty audio and label unknown: reload a different chunk
        empty_audio = True

        while empty_audio:
            # Load Audio
            clean = self.__get_audio(ID, y, pad=False)

            # Renormalize audio: cropped
            clean = self.__normalize_audio(clean)

            if(clean.sum().item() != 0.0):
                empty_audio = False
            elif(y != int(self.class_index['unknown'])):
                raise NameError('Empty audio file with non valid label: ' + str(ID))

        # Check audio: remove in the future
        assert not torch.isnan(clean.sum()), 'THE AUDIO FAILS... ID: ' + str(ID)

        # Augment with rir
        if self.rirs_paths:
            clean = self.__convolve_with_ir(clean, self.rir_prob, self.rirs_paths)

        # Augment with dir
        if self.dirs_paths:
            clean = self.__convolve_with_ir(clean, self.dir_prob, self.dirs_paths)
            
        # Check dimension
        if clean.shape[0] < self.window_size:
            clean = self.__pad_audio(clean, clean.shape[0])
        else:
            # Just in case as RIR augment changes the length
            # Take the first window_size samples
            clean = clean[:self.window_size]
        
        # Combine with background sound if defined
        if self.noise_paths:
            noisy = self.__add_background_noise(clean, y)
            # Renormalize audio: bigger than 1.0
            noisy = self.__normalize_audio(noisy, eps=0)
        else:
            noisy = clean   # No background noise

        # Check audio: remove in the future
        assert not torch.isnan(noisy.sum()), 'THE NOISY AUDIO FAILS... ID: ' + str(ID)

        return ID, noisy, clean, y

    def get_class_weights(self):
        """
        Gets class weights.

        Args:
            None.
        Returns:
            Class weight vector.
        """
        classes = list(self.dataframe['Label'].unique())
        count_series = self.dataframe['Label'].value_counts().to_dict()

        # Reset all values of the dictionary to 0.
        classes_counter = dict.fromkeys(classes, 0)
        for class_label in classes:
            classes_counter[class_label] = count_series[class_label]

        weights = {}
        for key, value in classes_counter.items():
            weights[key] = 1.0/value

        return weights

    def get_samples_weight(self):
        """
        Gets sample weights to deal with data unbalancing.

        Args:
            None
        Returns:
            Sample weight vector (size of dataset len)
        """
        weight = self.get_class_weights()
        samples_weight = np.array([weight[label] for label in self.labels])
        return samples_weight

    def __read_wav(self, filepath, frame_offset=0, num_frames=-1):
        """
        Read audio wave file applying normalization.

        Args:
            filepath: audio file path
        Returns:
            audio_signal: numpy array containing audio signal
        """
        try:
            audio_signal, sr = torchaudio.load(
                filepath,
                frame_offset=frame_offset,
                num_frames=num_frames
                )
        
            if(sr != self.fs):
                print('Sampling frequency of {0} is {1} and should be {2}'.format(filepath, sr, self.fs))
                print('Performing resampling...')
                audio_signal = torchaudio.transforms.Resample(sr, self.fs)(audio_signal)
            audio_signal = self.__normalize_audio(audio_signal[0])
            return audio_signal

        except:
            print(f'Error loading path: {filepath} - {frame_offset} - {num_frames}')
    
    def __normalize_audio(self, audio, eps=0.001):
        return (audio.float() / (torch.max(torch.abs(audio)).float() + eps))

    def __pad_audio(self, audio, length):
        """Pad audio file to give it the length of the window size"""
        padding = self.window_size - length
        bounds_sizes = np.random.multinomial(padding, np.ones(2)/2, size=1)[0]
        return torch.nn.functional.pad(audio, (bounds_sizes[0], bounds_sizes[1]), mode='constant', value=0)

    def __get_audio(self, ID, y, pad=True):
        """
        Get audio clip using segmentation informatio when it exist or
        using random crop otherwise.

        Args:
            ID: sample id
            y: target
        """
        time_stretch = False
        audio_path = self.dataframe.set_index('Sample_ID').at[ID, 'Sample_Path']
        audio_time = self.dataframe.set_index('Sample_ID').at[ID, 'Audio_Length']
        audio_length = int(audio_time * self.fs)

        # Segmentation information: VAD/SAD
        start_time = self.dataframe.set_index('Sample_ID').at[ID, 'Start']
        end_time = self.dataframe.set_index('Sample_ID').at[ID, 'End']

        # Get speech activity length: 
        # utterance_length is equal to audio_length for all unknown labels
        utterance_start = int(start_time * self.fs)
        utterance_end = int(end_time * self.fs)
        utterance_length = utterance_end - utterance_start

        if(utterance_length >= self.window_size):
            num_frames = self.window_size
            
            # If there is not segmentation information
            if (utterance_length == audio_length):
                    
                # Apply random crop on negative data
                left_bound = random.randint(0, max(0, audio_length - self.window_size))

            # When there is valid segmentation information
            else:
                
                # If negative label: we do not care about the speech we take so random crop
                # If time stretch is not applied then random crop also done in target samples
                if (y == int(self.class_index['unknown'])) or (not self.time_stretch):
                    left_bound = random.randint(utterance_start, utterance_end-self.window_size)
                
                # If time stretch is configured, then we apply it to positive labels
                # We want all speech content to be included, so we compress the time axis
                else:
                    time_stretch = True
                    left_bound = utterance_start
                    num_frames = utterance_end

            audio_signal = self.__read_wav(audio_path, frame_offset=left_bound, num_frames=num_frames)

        else:
            audio_signal = self.__read_wav(audio_path, frame_offset=utterance_start, num_frames=utterance_length) 
            
            if pad:
                audio_signal = self.__pad_audio(audio_signal, utterance_length)

        if time_stretch:
            audio_signal = self.__time_stretch(audio_signal)

        return audio_signal

    def __add_background_noise(self, data, y):
        """
        Add background noise to a single sample.

        Args:
            data: original audio signal
        """
        # Use probability of applying background noise
        if random.random() > self.bn_prob:
            return data # Not apply bn
        
        empty_audio = True

        while empty_audio:
            background_noise_path = random.sample(self.noise_paths, 1)[0]
            background_length = self.noise_df.set_index('Sample_Path').at[background_noise_path, 'Audio_Length']
            background_length = int(background_length * self.fs)

            # Get left bound for audio: 0 if background_length is smaller than window_size
            left_bound = random.randint(0, max(0, background_length - self.window_size))

            # Load audio
            background_noise = self.__read_wav(
                background_noise_path,
                frame_offset=left_bound,
                num_frames=self.window_size
                )

            background_length = int(background_noise.shape[0])

            # Check audio: remove in the future
            if(background_noise.sum().item() != 0.0):
                empty_audio = False
        
        # Pad with 0z if backgorund noise is shorter than window
        if(background_length < self.window_size):
           background_noise = self.__pad_audio(background_noise, background_length)

        # Use probability of applying non speech data for negative samples
        if (random.random() < self.non_speech_prob) and (y == int(self.class_index['unknown'])):
            return background_noise
        
        # Calculate SNR
        SNR_DB = np.random.uniform(low=self.snr_range[0], high=self.snr_range[1])
        data_rms = torch.sqrt(torch.mean(data**2))
        background_noise_rms = torch.sqrt(torch.mean(background_noise**2))
        desired_noise_rms = torch.sqrt((data_rms**2 / math.pow(10, SNR_DB/10)))
        k = desired_noise_rms / background_noise_rms
        data = data + k*background_noise

        return data
    
    def __time_stretch(self, data, eps=0.0001):
        n_fft = int(self.fs // 64)
        hop_length = int(n_fft // 32)
        audio_spec = torch.stft(data.unsqueeze(0), n_fft, hop_length, return_complex=True)
        fast_spec = torchaudio.transforms.TimeStretch(
            hop_length=hop_length,
            n_freq=audio_spec.shape[1]
            )(audio_spec, data.shape[0]/self.window_size + eps)
        return torch.istft(fast_spec, n_fft, hop_length).squeeze(0)
    
    def __convolve_with_ir(self, data, probability, list_of_ir_paths):
        """
        Convolve an audio signal with an impulse response.
        Here we use time convolution instead of frequency multiplication.
        Because this way we avoid temporal aliasing with circular convolution.
        Nevertheless, the output signal will be longer than the input signal.
        It is necessary to truncate the output signal to the same length as the input signal.

        Args:
            data: original audio signal
        """
        if random.random() < probability:
            empty_audio = True

            while empty_audio:
                ir_path = random.sample(list_of_ir_paths, 1)[0]
                ir = self.__read_wav(ir_path)
                ir, _ = trim_silence(ir)

                # Check audio: remove in the future
                if(ir.sum().item() != 0.0):
                    empty_audio = False
            
            # NOTE: This changes the windows size
            reverberated = self.convolve(data.unsqueeze(0), ir.unsqueeze(0)) 
            return reverberated.squeeze(0)
        else:
            return data
