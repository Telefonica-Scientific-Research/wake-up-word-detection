# Wake-Up-Word Detection

PyTorch code to train and evaluate wake-up-word (WuW) / keyword spotting models.
It includes several RNN, CNN and Transformer-based architectures, a configurable
audio feature extractor and a set of waveform and spectrogram data augmentations.

Contact: fernando.lopez@telefonica.com

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pinned versions were tested with Python 3.10.

## Data format

The code reads a *partition* folder that you prepare from your own data:

```
partition/
├── train.tsv
├── dev.tsv
├── test.tsv
├── classes_index.json
└── partition.json
```

Each `.tsv` file is tab-separated with one row per audio file and the following columns:

| Column         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `Sample_ID`    | Unique identifier of the sample                                             |
| `Sample_Path`  | Path to a mono 16 kHz `.wav` file                                           |
| `Label`        | Class name (must match `classes_index.json`); negatives use `unknown`       |
| `Sound_Type`   | `speech` for samples to classify, `noise` for background noise, `rir`/`dir` for impulse responses used in augmentation |
| `Audio_Length` | Length of the file in seconds                                               |
| `Start`, `End` | Speech segment within the file in seconds (`0` and `Audio_Length` if unknown) |
| `Database`     | Name of the source corpus (informative only)                                |

Any extra column (e.g. speaker or recording-device metadata) can be used to compute
metrics per group at test time (see `--specific_group`).

`classes_index.json` maps output indices to class names, for example:

```json
{"0": "WuW", "1": "unknown"}
```

`partition.json` contains the number of training `speech` samples per class, used by
the weighted sampler (`--use_sampler`):

```json
{"Train": {"Samples information": {"speech": {"WuW": 1200, "unknown": 25000}}}}
```

## Training

`train_kws.sh` contains the default hyperparameters and the list of architecture /
feature-extractor pairs. Set `partition` to your partition folder and run:

```bash
./train_kws.sh
```

or call the training script directly:

```bash
python src/train_and_test.py --arc sgru --feature_extractor mfccs_sonopytorch_13_w100_h50 \
    --partition_path partitions/ --epochs 200 --use_sampler --bn_prob 1.0 --wn_prob 0.2
```

Each run creates `models/checkpoints/kws_<arch>/version_<N>/` with the model
(`<arch>.pt`, `<arch>_entire.pt`), a JSON report, TensorBoard logs and test metrics.
Run `python src/train_and_test.py --help` for all options.

### Architectures

`sgru`, `sgru_max`, `sgru_avg`, `sgru_att`, `resnet8`, `resnet15`, `resnet15-narrow`,
`cnn_trad_pool2`, `cnn_trad_pool2s`, `cnn_fat2019`, `cnn10`, `lenet`,
`bcresnet`, `lambda-resnet18`, `conformer`, `audiomer-2d`, `audiomer-l`, `audiomer-s`.

Recommended feature extractors for each architecture are listed in `train_kws.sh`.

### Data augmentation

- Dataloader: background noise (`--bn_prob`), non-speech negatives (`--non_speech_prob`),
  room/device impulse responses (`--rir_prob`, `--dir_prob`).
- Batch, waveform/spectrogram: white noise, fade, frequency and time masking, time
  stretching, RandAugment, FilterAugment.
- Label-based, mixing samples and their labels (soft targets): MixUp (`--mixup_prob`) and
  SpecMix (`--specmix_prob`). A fraction of each batch (`--mixup_perc_real_samples`,
  `--specmix_perc_real_samples`) is never mixed.
- Features: Freq-MixStyle (`--freqmixstyle_prob`), applied per (dominant) class.

## Evaluation

`src/train_and_test.py` evaluates on `test.tsv` after training. To evaluate a trained
version again, optionally on another test file and per group:

```bash
python src/test.py --arc sgru --partition_path partitions/ --version 1 \
    --test_filename test.tsv --specific_group <column_name>
```

## Export for TorchScript and Android

`src/export_model.py` bundles the feature extractor, the model and a softmax into a single
TorchScript module that takes raw audio and returns class scores:

```bash
python src/export_model.py --experiment_path models/checkpoints/kws_sgru/version_1 --benchmark_runs 100
```

It writes to `<experiment_path>/deployment/`:

- `<arch>.jit`: TorchScript model, loadable with `torch.jit.load` (Python or C++ LibTorch).
- `<arch>.ptl`: model for the PyTorch Lite interpreter, used by the PyTorch Android library
  (`org.pytorch:pytorch_android_lite`).
- `model_info.json`: input/output shapes, sampling rate and class index.

Inputs are a float tensor `[1, n_samples]` with audio at the training sampling rate
(e.g. `[1, 24000]` for 1.5 s at 16 kHz). GRU models (`sgru*`) also take and return the
hidden state `[n_layers, 1, hidden_dim]`, which can be carried between consecutive windows
in streaming or reset to zeros.

Options:

- `--do_quantization`: dynamic int8 quantization of linear and GRU layers.
- `--no_optimization`: skip `optimize_for_mobile` for the `.ptl` model. The `.jit` model is never
  optimized for mobile, as those optimizations target ARM devices and slow down x86 CPUs.
- `--benchmark_runs N`: CPU forward time of the original and exported models.

The script checks that the exported models give the same scores as the original one and prints
the maximum difference. Models that cannot be scripted are traced with a fixed-size input.

### Android benchmark

`benchmark_android.sh` measures the speed of a `.ptl` model on a device connected with `adb`.
It needs the `speed_benchmark_torch` binary (and optionally `load_benchmark_torch`) from the
PyTorch sources, built for Android with the Lite interpreter and copied to `/data/local/tmp`
on the device. The input dimensions are printed by `export_model.py`:

```bash
./benchmark_android.sh models/checkpoints/kws_sgru/version_1/deployment/sgru.ptl "1,24000;1,1,200"
```

## Citation

This code is the basis of the following works. If you use it, please cite the relevant ones.
The speech enhancement models of the IberSPEECH 2021 and TASE papers are not included in
this repository.

```bibtex
@inproceedings{lopez2026befairme,
  author    = {L{\'o}pez, Fernando and Delgado-Santos, Paula and G{\'o}mez, Pablo and Solans, David and Luque, Jordi},
  title     = {``{OK} {A}ura, Be Fair with Me'': Demographics-Agnostic Training for Bias Mitigation in Wake-up Word Detection},
  booktitle = {Proceedings of the Workshop on Speech Language Models in Low-Resource Settings: Performance, Evaluation, and Bias Analysis (SPEAKABLE) @ LREC 2026},
  pages     = {47--58},
  year      = {2026},
  month     = may,
  address   = {Palma, Mallorca, Spain},
  publisher = {ELRA Language Resources Association},
  isbn      = {978-2-493814-83-8}
}

@misc{lopez2023robust,
  author        = {L{\'o}pez, Fernando and Luque, Jordi and Segura, Carlos and G{\'o}mez, Pablo},
  title         = {Robust Wake-Up Word Detection by Two-stage Multi-resolution Ensembles},
  year          = {2023},
  eprint        = {2310.11379},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/2310.11379},
  note          = {Accepted in IberSPEECH 2026}
}

@article{cambara2022tase,
  author  = {C{\'a}mbara, Guillermo and L{\'o}pez, Fernando and Bonet, David and G{\'o}mez, Pablo and Segura, Carlos and Farr{\'u}s, Mireia and Luque, Jordi},
  title   = {{TASE}: Task-Aware Speech Enhancement for Wake-Up Word Detection in Voice Assistants},
  journal = {Applied Sciences},
  volume  = {12},
  number  = {4},
  pages   = {1974},
  year    = {2022},
  doi     = {10.3390/app12041974}
}

@inproceedings{bonet2021speech,
  author    = {Bonet, David and C{\'a}mbara, Guillermo and L{\'o}pez, Fernando and G{\'o}mez, Pablo and Segura, Carlos and Luque, Jordi and Farr{\'u}s, Mireia},
  title     = {Speech Enhancement for Wake-Up-Word Detection in Voice Assistants},
  booktitle = {Proc. IberSPEECH 2021},
  pages     = {41--45},
  year      = {2021},
  doi       = {10.21437/IberSPEECH.2021-9}
}
```

## License

© 2026 Telefónica Innovación Digital S.L.

This project is licensed under the GNU Affero General Public License v3.0 or later.
See [LICENSE](LICENSE) for the full text and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for code adapted from other projects.
