#!/bin/bash

# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of wake-up-word-detection.
#
# wake-up-word-detection is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# wake-up-word-detection is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with wake-up-word-detection. If not, see https://www.gnu.org/licenses/

# Select cuda device
export CUDA_VISIBLE_DEVICES=0

# Audio configuration
sampling_rate=16000
time_window=1.5

# Train hyperparameters
batch_size=128
epochs=700   
seed=2022
lr=0.001
lr_patience=40 # step scheduler: like ReduceLROnPlateau
patience=$(($lr_patience*4))

# Data Augments
min_snr=-10 # Minimum SNR to combine with background noise
max_snr=50 # Maximum SNR to combine with background noise
wn_min_snr=5 # Minimum SNR to combine with white noise
wn_max_snr=50 # Maximum SNR to combine with white noise
num_workers=3

# Augments probability
wn_prob=0.2 # White noise probability
rir_prob=0.2 # RIR probability
bn_prob=1.0 # Background noise probability
non_speech_prob=0.1 # Non speech probability

##############################################
# Data Augments: to deal with device diversity
##############################################

augment_nfft=512 # FFT size
augment_hop=160 # Hop size

# Batch-wise data augmentation: applies a transformation to each sample in a batch
dir_prob=0.0 # DIR probability
filteraugment_prob=0.0 # Filter augment probability
freqmixstyle_prob=0.0 # Frequency mix style probability
freqmask_prob=0.0 # Frequency masking probability
timemask_prob=0.0 # Time masking probability
timestretch_prob=0.0 # Time stretching probability
randaugment_prob=0.0 # RandAugment probability
fade_prob=0.0 # Fade probability

# Partition: folder with train.tsv, dev.tsv, test.tsv, classes_index.json and partition.json (see README)
partition=partitions/

# Experiment description
experiment_description="b_"$batch_size"-e_"$epochs"-s_"$seed"-lr_"$lr"-lrp_"$lr_patience"-p_"$patience

# Architectures and features pairs
#   
#   +-----------------+-------------------------------+
#   |      Arch       |       Feature Extractor       |
#   +-----------------+-------------------------------+
#   | sgru            | mfccs_sonopytorch_13_w100_h50 |
#   |                 | mfccs_sonopytorch_13_w100_h20 |
#   |                 | mfccs_sonopytorch_13_w30_h10  |
#   |                 | mfccs_sonopytorch_13_w20_h10  |
#   |                 | mfccs_sonopytorch_40_w30_h10  |
#   |-----------------|-------------------------------|
#   | sgru_avg        | mfccs_sonopytorch_40_w30_h10  |
#   |-----------------|-------------------------------|
#   | sgru_max        | mfccs_sonopytorch_40_w30_h10  |
#   |-----------------|-------------------------------|
#   | sgru_att        | mfccs_sonopytorch_40_w30_h10  |
#   |-----------------|-------------------------------|
#   | bcresnet        | mfccs_sonopytorch_40_w30_h10  |
#   |-----------------|-------------------------------|
#   | cnn_trad_pool2s | mfccs_sonopytorch_13_w30_h10  |
#   |-----------------|-------------------------------|
#   | cnn_trad_pool2  | mfccs_sonopytorch_40_w30_h10  |
#   |-----------------|-------------------------------|
#   | resnet15        | mfccs_sonopytorch_40_w30_h10  |
#   |-----------------|-------------------------------|
#   | resnet15-narrow | mfccs_sonopytorch_40_w30_h10  |
#   |-----------------|-------------------------------|
#   | resnet8         | mfccs_sonopytorch_40_w30_h10  |
#   |-----------------|-------------------------------|
#   | lenet           | mfccs_sonopytorch_40_w30_h10  |
#   |-----------------|-------------------------------|
#   | cnn_fat2019     | mfccs_sonopytorch_40_w30_h10  |
#   |-----------------|-------------------------------|
#   | conformer       | mfccs_sonopytorch_40_w30_h10  |
#   |-----------------|-------------------------------|
#   | audiomer-2d     | mfccs_sonopytorch_40_w30_h10  |
#   |-----------------|-------------------------------|
#   | audiomer-l      | raw_audio                     |
#   |-----------------|-------------------------------|
#   | audiomer-s      | raw_audio                     |
#   +-----------------+-------------------------------+
#

archs=(sgru)
feature_extractors=(mfccs_sonopytorch_13_w100_h50)

n_archs=${#archs[@]}
n_features=${#feature_extractors[@]}

if [ $n_archs -ne $n_features ];
then
    echo "Number of architectures and features must be the same."
    exit 1
fi

for (( i=0; i<$n_archs; i++ ))
do
    arch=${archs[i]}
    feature_extractor=${feature_extractors[i]}
    echo -e "\n********************$arch********************\n"

    # Training and test
    python -u src/train_and_test.py --arc $arch --feature_extractor $feature_extractor \
    --batch_size $batch_size --sampling_rate $sampling_rate --time_window $time_window \
    --seed $seed --lr $lr --lr_patience $lr_patience --patience $patience \
    --min_snr $min_snr --max_snr $max_snr --epochs $epochs --partition_path $partition \
    --wn_min_snr $wn_min_snr --wn_max_snr $wn_max_snr --wn_prob $wn_prob --rir_prob $rir_prob \
    --bn_prob $bn_prob --use_sampler --experiment_description $experiment_description \
    --non_speech_prob $non_speech_prob --dir_prob $dir_prob --freqmask_prob $freqmask_prob \
    --timemask_prob $timemask_prob --timestretch_prob $timestretch_prob \
    --randaugment_prob $randaugment_prob --fade_prob $fade_prob \
    --augment_nfft $augment_nfft \
    --augment_hop_len $augment_hop --num_workers $num_workers --filteraugment_prob $filteraugment_prob \
    --freqmixstyle_prob $freqmixstyle_prob

done
