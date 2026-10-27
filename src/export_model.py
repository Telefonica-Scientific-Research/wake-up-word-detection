# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of EdgeWuW.
#
# EdgeWuW is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# EdgeWuW is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with EdgeWuW. If not, see https://www.gnu.org/licenses/

"""
Export a trained model to TorchScript (.jit) and to the PyTorch Lite interpreter
format (.ptl) used on Android.

The exported model includes the feature extractor and a softmax:
    - CNN-like models: scores = model(audio)
    - GRU models:      scores, h = model(audio, h)
where audio is a float tensor [1, n_samples] at the training sampling rate,
scores is [1, n_classes] and h is the GRU hidden state [n_layers, 1, hidden_dim].
"""

import os
import glob
import json
import time
import argparse

import torch
import torchaudio
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.jit.mobile import _load_for_lite_interpreter

from keyword_spotting.feature_extractor import FeatureExtractor
from keyword_spotting.reporter import Reporter
from utils.experiment_utils import check_path


class LogMelSpectrogram(torch.nn.Module):
    """Same log Mel-spectrogram as FeatureExtractor('melspectrogram_torchaudio')."""
    def __init__(self, sample_rate, melkwargs, eps=0.0001):
        super().__init__()
        self.mels = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, **melkwargs)
        self.eps = eps

    def forward(self, x):
        return torch.log(self.mels(x) + self.eps)


class FrontEnd(torch.nn.Module):
    """Scriptable version of the FeatureExtractor used during training."""
    def __init__(self, fe):
        super().__init__()
        self.transpose = fe.transpose

        if fe.features == 'raw_audio':
            self.extract_features = torch.nn.Identity()
        elif fe.features == 'melspectrogram_torchaudio':
            self.extract_features = LogMelSpectrogram(fe.fs, fe.melkwargs)
        elif isinstance(fe.extract_features, torch.nn.Module):
            # sonopytorch MFCCs and torchaudio spectrogram/MFCCs
            self.extract_features = fe.extract_features
        else:
            raise ValueError('Feature extractor cannot be exported: ' + fe.features)

    def forward(self, x):
        x = self.extract_features(x)
        if self.transpose:
            x = x.transpose(2, 1)
        return x


class RNNWrapper(torch.nn.Module):
    def __init__(self, front_end, model):
        super().__init__()
        self.front_end = front_end
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, h):
        x, h = self.model(self.front_end(x), h)
        return self.softmax(x), h


class CNNWrapper(torch.nn.Module):
    def __init__(self, front_end, model):
        super().__init__()
        self.front_end = front_end
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.model(self.front_end(x)))


def run(model, inputs):
    """Returns the class scores of a model for the given inputs."""
    out = model(*inputs)
    return out[0] if isinstance(out, tuple) else out


def to_torchscript(model, inputs, mobile_optimization):
    """
    Scripts the model (tracing as fallback). Returns the TorchScript model, its
    version for mobile (optionally optimized) and the conversion method.
    """
    attempts = [('scripting', lambda: torch.jit.script(model)), ('tracing', lambda: torch.jit.trace(model, inputs))]
    for name, convert in attempts:
        try:
            print("\t" + name + "...")
            torchscript_model = convert()
            run(torchscript_model, inputs)
            mobile_model = optimize_for_mobile(torchscript_model) if mobile_optimization else torchscript_model
            run(mobile_model, inputs)
            return torchscript_model, mobile_model, name
        except Exception as excep:
            print("\tProblem found while " + name + ": " + str(excep).strip().split('\n')[0])
    raise RuntimeError('The model could not be converted to TorchScript')


def get_forward_time(model, inputs, n_runs):
    """Average CPU forward time in milliseconds."""
    with torch.no_grad():
        run(model, inputs) # warm-up
        start = time.time()
        for _ in range(n_runs):
            run(model, inputs)
    return 1000 * (time.time() - start) / n_runs


def main(args):

    experiment_path = os.path.normpath(args.experiment_path)
    if not os.path.isdir(experiment_path):
        raise ValueError('Experiment path is not a directory: ' + args.experiment_path)

    # Load training report
    report_paths = glob.glob(os.path.join(experiment_path, '*_report.json'))
    if len(report_paths) != 1:
        raise ValueError('Expected one *_report.json file in ' + experiment_path)
    reporter = Reporter(experiment_path, os.path.basename(report_paths[0]))
    reporter.load(report_paths[0])
    train_args = reporter.data['arguments']
    arch = train_args['arc']

    # Deployment path
    deployment_path = os.path.join(experiment_path, 'deployment')
    check_path(deployment_path)

    device = torch.device('cpu')
    torch.set_grad_enabled(False)

    # Load model
    model = torch.load(os.path.join(experiment_path, arch + '_entire.pt'), map_location=device)
    model.eval()

    # Build the full model: feature extraction + model + softmax
    fe = FeatureExtractor(
        device,
        train_args['time_window'],
        train_args['sampling_rate'],
        train_args['feature_extractor'],
        train_args['transpose']
        )
    front_end = FrontEnd(fe)
    n_samples = int(train_args['time_window'] * train_args['sampling_rate'])
    audio = torch.rand(1, n_samples) * 2 - 1

    is_rnn = 'sgru' in arch
    if is_rnn:
        full_model = RNNWrapper(front_end, model)
        hidden_shape = (model.n_layers, 1, model.hidden_dim)
        inputs = (audio, torch.zeros(hidden_shape))
    else:
        full_model = CNNWrapper(front_end, model)
        inputs = (audio, )
    full_model.eval()

    print("\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
    print("Pre-conversion check")
    reference = run(full_model, inputs)
    print("Output sample: ", reference)

    # Quantization: dynamic quantization of linear and recurrent layers
    if args.do_quantization:
        print("\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
        print("Dynamic quantization...")
        torch.backends.quantized.engine = 'qnnpack' # widely supported on Android
        full_model = torch.quantization.quantize_dynamic(full_model, {torch.nn.Linear, torch.nn.GRU}, dtype=torch.qint8)

    # Script/trace and optimize
    print("\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
    print("TorchScript conversion" + (" and mobile optimization" if args.do_mobile_optimization else "") + "...")
    torchscript_model, mobile_model, conversion = to_torchscript(full_model, inputs, args.do_mobile_optimization)

    # Save TorchScript (desktop/server) and Lite interpreter (mobile) models
    # NOTE: mobile optimization targets ARM devices, it slows down x86 CPUs
    jit_path = os.path.join(deployment_path, arch + '.jit')
    ptl_path = os.path.join(deployment_path, arch + '.ptl')
    torchscript_model.save(jit_path)
    mobile_model._save_for_lite_interpreter(ptl_path)

    # Model information for the integration in an application
    classes_index_path = os.path.join(experiment_path, 'classes_index.json')
    classes_index = json.load(open(classes_index_path)) if os.path.isfile(classes_index_path) else None
    model_info = {
        'arch': arch,
        'conversion': conversion,
        'mobile_optimization': args.do_mobile_optimization,
        'quantization': args.do_quantization,
        'sampling_rate': train_args['sampling_rate'],
        'inputs': {'audio': [1, n_samples]},
        'outputs': {'scores': [1, int(reference.shape[1])]},
        'classes_index': classes_index,
    }
    if is_rnn:
        model_info['inputs']['hidden_state'] = list(hidden_shape)
        model_info['outputs']['hidden_state'] = list(hidden_shape)
    with open(os.path.join(deployment_path, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=4)

    # Post-conversion check: reload exported models
    print("\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
    print("Post-conversion check")
    exported = {
        'jit': torch.jit.load(jit_path),
        'ptl': _load_for_lite_interpreter(ptl_path),
    }
    for name, exported_model in exported.items():
        out = run(exported_model, inputs)
        print("\t{0}: max abs difference with the original model: {1:.2e}".format(name, (out - reference).abs().max().item()))

    if args.benchmark_runs > 0:
        print("\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
        print("CPU forward time (average of {0} runs)".format(args.benchmark_runs))
        if args.do_mobile_optimization:
            print("\tNOTE: ptl is optimized for mobile, use benchmark_android.sh to measure it on device")
        print("\toriginal: {0:.2f} ms".format(get_forward_time(full_model, inputs, args.benchmark_runs)))
        for name, exported_model in exported.items():
            print("\t{0}: {1:.2f} ms".format(name, get_forward_time(exported_model, inputs, args.benchmark_runs)))

    input_dims = ';'.join(','.join(str(d) for d in shape) for shape in model_info['inputs'].values())
    print("\nExported models: " + jit_path + ", " + ptl_path)
    print("Android benchmark: ./benchmark_android.sh " + ptl_path + " \"" + input_dims + "\"")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Export a trained model to TorchScript and PyTorch Lite (Android)')
    parser.add_argument('--experiment_path', required=True, help='experiment path, e.g. models/checkpoints/kws_sgru/version_1')
    parser.add_argument('--do_quantization', action='store_true', help='apply dynamic int8 quantization to linear and GRU layers')
    parser.add_argument('--no_optimization', dest='do_mobile_optimization', action='store_false', help='do not optimize the .ptl model for mobile')
    parser.add_argument('--benchmark_runs', type=int, default=0, help='number of runs to measure CPU forward time (0 to skip)')
    args = parser.parse_args()

    main(args)
