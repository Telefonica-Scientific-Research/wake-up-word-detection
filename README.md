# Wake-Up-Word Detection

PyTorch code to train and evaluate wake-up-word (WuW) / keyword spotting models.
It includes several RNN, CNN and Transformer-based architectures, a configurable
audio feature extractor and a set of waveform and spectrogram data augmentations.

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
- Features: Freq-MixStyle (`--freqmixstyle_prob`).

## Evaluation

`src/train_and_test.py` evaluates on `test.tsv` after training. To evaluate a trained
version again, optionally on another test file and per group:

```bash
python src/test.py --arc sgru --partition_path partitions/ --version 1 \
    --test_filename test.tsv --specific_group <column_name>
```

## License

© 2026 Telefónica Innovación Digital S.L.

This project is licensed under the GNU Affero General Public License v3.0 or later.
See [LICENSE](LICENSE) for the full text and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for code adapted from other projects.
