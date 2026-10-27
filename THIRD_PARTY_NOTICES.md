# Third-party code

Parts of this repository are adapted from, or follow, the following works:

| File | Component | Source | License |
|------|-----------|--------|---------|
| `src/utils/dsp.py` | Amplitude, rescaling, 1-D convolution and reverberation utilities | [SpeechBrain](https://github.com/speechbrain/speechbrain) (`speechbrain/processing/signal_processing.py`) | Apache-2.0 |
| `src/keyword_spotting/augments.py` | `FreqMixStyle` | [cpjku_dcase23](https://github.com/fschmid56/cpjku_dcase23) (`helpers/utils.py`) | See source repository |
| `src/keyword_spotting/models.py` | `cnn10` | [audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) | MIT |
| `src/keyword_spotting/models.py`, `src/utils/model_utils.py` | BC-ResNet (`bcresnet`) | Implementation of [Kim et al., 2021](https://arxiv.org/abs/2106.04140) | — |
| `src/utils/lambda_utils.py` | Lambda layers (`lambda-resnet18`) | Implementation of [Bello, 2021](https://arxiv.org/abs/2102.08602) | — |
| `src/utils/audiomer2d_utils.py`, `src/utils/model_utils.py` | Audiomer (`audiomer-*`) | Implementation of [Sahu et al., 2021](https://arxiv.org/abs/2109.10994) | — |
