#!/bin/bash

# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of EdgeWuW.
#
# EdgeWuW is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# EdgeWuW is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with EdgeWuW. If not, see https://www.gnu.org/licenses/

# Speed and load benchmark of an exported model (.ptl) on an Android device connected with adb.
# It requires the PyTorch benchmark binaries (speed_benchmark_torch and, optionally,
# load_benchmark_torch) built for Android and copied to /data/local/tmp (see README).
#
# Usage: ./benchmark_android.sh <model.ptl> <input_dims> [extra speed_benchmark_torch flags]
#   input_dims: as printed by src/export_model.py, e.g. "1,24000" or "1,24000;1,1,200" for GRU models

if [ $# -lt 2 ]; then
    echo "Usage: $0 <model.ptl> <input_dims> [extra speed_benchmark_torch flags]"
    exit 1
fi

model=$1
input_dims=$2
extra_flags="${@:3}"
device_dir=/data/local/tmp
model_name=$(basename "$model")

# One type and memory format per input
n_inputs=$(echo "$input_dims" | tr ';' '\n' | wc -l)
input_type=$(printf 'float;%.0s' $(seq $n_inputs)); input_type=${input_type%;}
input_memory_format=$(printf 'contiguous_format;%.0s' $(seq $n_inputs)); input_memory_format=${input_memory_format%;}

adb push "$model" $device_dir/$model_name || exit 1

echo "Speed benchmark of model: $model_name"
adb shell "$device_dir/speed_benchmark_torch --model=$device_dir/$model_name \
    --input_dims='$input_dims' --input_type='$input_type' \
    --input_memory_format='$input_memory_format' $extra_flags"

if adb shell "[ -x $device_dir/load_benchmark_torch ]"; then
    echo "Load benchmark of model: $model_name"
    adb shell "$device_dir/load_benchmark_torch --model=$device_dir/$model_name"
fi
