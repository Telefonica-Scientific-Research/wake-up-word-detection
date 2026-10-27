# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of wake-up-word-detection.
#
# wake-up-word-detection is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# wake-up-word-detection is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with wake-up-word-detection. If not, see https://www.gnu.org/licenses/

import argparse


def parse_arguments():

    parser = argparse.ArgumentParser(description='KWS models')

    #############################
    # General
    #############################

    # Source data
    parser.add_argument('--partition_path', default='partitions/', help='path to the partition folder containing the train, dev and test dataframes')

    # Data Loaders
    parser.add_argument('--num_workers', type=int, default=3, metavar='N', help='DataLoader workers')

    # Hyperparameters
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')
    parser.add_argument('--use_sampler', dest='use_batch_sampler', action='store_true', help='use Weighted Random Sampler to deal with data unbalance')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='Batch size to use')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--lr_patience', type=int, default=20, help='number of epochs of no loss improvement before updating lr')
    parser.add_argument('--patience', type=int, default=60, help='number of epochs of no loss improvement before stop training')
    
    # Start from checkpoint
    parser.add_argument('--from_checkpoint', default='', help='path to the folder containing a checkpoint to start')
    
    # GPU
    parser.add_argument('--no_cuda', dest='cuda', action='store_false', help='do not use cuda')
    parser.add_argument('--cuda_device', type=int, default=0, help='device to use')
    
    # Optimizer
    parser.add_argument('--optimizer', default='adam', help='optimization method: sgd | adam | rmsprop')
    parser.add_argument('--weight_decay', type=float, default=0.0, metavar='WD', help='weight decay for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
    
    # Feature extraction
    parser.add_argument('--feature_extractor', default='melspectrogram_torchaudio', help='feature extractor to use')
    parser.add_argument('--transpose', dest='transpose', action='store_true', help='to transpose the features')

    # Store results
    parser.add_argument('--checkpoint', default='models/checkpoints', metavar='CHECKPOINT', help='checkpoints directory')
    parser.add_argument('--experiment_description', default='train with 2stages partition', help='experiment name of the folder where best models are saved at')

    # Audio
    parser.add_argument('--sampling_rate', type=float, default=16000, metavar='SR', help='sampling rate of the audio signal')
    parser.add_argument('--time_window', type=float, default=1.5, metavar='TW', help='time window covered by every data sample')
    parser.add_argument('--no_random_crop', dest='random_crop', action='store_false', help='do not use random crop on negative samples')

    # KWS
    parser.add_argument('--arc', default='sgru', help='network architecture name')
    parser.add_argument('--no_one_hot', dest='one_hot', action='store_false', help='target format in one hot encoding')

    # Arguments to test a concrete version or model
    parser.add_argument('--version', type=int, default=None, help='version to be tested')
    parser.add_argument('--test_seed', type=int, default=2022, help='version to be tested')

    #############################
    # Data Augmentation
    #############################

    # The following parameters are used to augment in the time-frequency domain
    # Specificaly: FrequencyMasking, TimeMasking, TimeStretching, RandAugment, FilterAugment
    parser.add_argument('--augment_nfft', type=int, default=250, help='number of fft components')
    parser.add_argument('--augment_hop_len', type=int, default=7, help='number of audio frames between STFT columns (hop length)')

    # White Nosie
    parser.add_argument('--wn_min_snr', type=int, default=20, help='white noise min snr in dB')
    parser.add_argument('--wn_max_snr', type=int, default=40, help='white noise max snr in dB')
    parser.add_argument('--wn_prob', type=float, default=0.0, help='probability of applying white noise')

    # Fade
    parser.add_argument('--fade_prob', type=float, default=0.0, help='probability of applying fade to audio')
    parser.add_argument('--fade_in', type=int, default=1000, help='length of fade-in (time frames)')
    parser.add_argument('--fade_out', type=int, default=1000, help='length of fade-out (time frames)')
    parser.add_argument('--fade_shape', default='linear', help='shape of fade: “quarter_sine”, “half_sine”, “linear”, “logarithmic”, “exponential”')

    # Frequency Masking
    parser.add_argument('--freqmask_max_len', type=int, default=30, help='maximum possible length for frequency masking in Mel bands')
    parser.add_argument('--freqmask_n_masks', type=int, default=2, help='number of frequency masks to apply')
    parser.add_argument('--freqmask_prob', type=float, default=0.0, help='probability of applying frequency masking')

    # Time Masking
    #NOTE: the following parameter could be renamed to time_mask_length
    parser.add_argument('--timemask_max_len', type=int, default=40, help='maximum possible length for time masking in time frames')
    parser.add_argument('--timemask_n_masks', type=int, default=2, help='number of time masks to apply')
    parser.add_argument('--timemask_prob', type=float, default=0.0, help='probability of applying time masking')

    # Impulsive responses
    parser.add_argument('--rir_prob', type=float, default=0.0, help='probability of applying RIR to speech data')
    parser.add_argument('--dir_prob', type=float, default=0.0, help='probability of applying DIR to speech data')

    # Background noise
    parser.add_argument('--bn_prob', type=float, default=0.0, help='probability of applying background noise speech data')
    parser.add_argument('--min_snr', type=int, default=5, help='Minimum SNR applied to original audio')
    parser.add_argument('--max_snr', type=int, default=30, help='Maximum SNR applied to original audio')

    # Non speech data
    parser.add_argument('--non_speech_prob', type=float, default=0.0, help='probability of using non speech data as negatives')

    # Time stretching: time domain: applied in the dataloader
    # Adapts audios longer than window size to the window size: in the past it did not work well
    parser.add_argument('--time_stretch', dest='time_stretch', action='store_true', help='do not use time stretching augment in time domain')

    # Time stretching: time-frequency domain
    parser.add_argument('--timestretch_prob', type=float, default=0.0, help='probability of applying time stretching to audio')
    parser.add_argument('--timestretch_eps', type=float, default=0.0001, help='small epsilon value to avoid division by zero or log of zero')
    
    # RandAugment: time-frequency domain
    parser.add_argument('--randaugment_prob', type=float, default=0.0, help='probability of applying randaugment to the spectrogram')
    parser.add_argument('--randaugment_n_ops', type=int, default=2, help='number of augmentation operations to apply')
    parser.add_argument('--randaugment_magnitude', type=int, default=9, help='amplitude or strength of the transformation applied to the spectrogram')
    parser.add_argument('--randaugment_n_mag_bins', type=int, default=31, help='The number of discrete bins used to represent the magnitude spectrum or range of values in a spectrogram')

    # FilterAugment: time-frequency domain
    parser.add_argument('--filteraugment_prob', type=float, default=0.0, help='probability of applying filteraugment to the spectrogram')
    parser.add_argument('--filteraugment_min_db', type=int, default=-3, help='minimum value of the filter gain')
    parser.add_argument('--filteraugment_max_db', type=int, default=3, help='maximum value of the filter gain')
    parser.add_argument('--filteraugment_min_bands', type=int, default=3, help='number minimum of bands to divide the frequency')
    parser.add_argument('--filteraugment_max_bands', type=int, default=9, help='number minimum of bands to divide the frequency')
    parser.add_argument('--filteraugment_min_bw', type=int, default=6, help='minimum value of the bandwidth of each band')
    parser.add_argument('--filteraugment_filter_type', default='linear', help='type of filter to apply: linear | step')
    
    # FreqMixStyle: normalization
    parser.add_argument('--freqmixstyle_prob', type=float, default=0.0, help='probability of applying freqmixstyle to the spectrogram')
    parser.add_argument('--freqmixstyle_alpha', type=float, default=0.4, help='alpha value for the beta distribution')

    #############################
    # Evaluation stack
    #############################
    
    # Evaluation by group (e.g. a column of the test TSV)
    parser.add_argument('--specific_group', default=None, help='specific column name group to use for evaluation')
    parser.add_argument('--test_filename', default=None, help='specific filename to be used for the evaluation')


    return parser.parse_args()