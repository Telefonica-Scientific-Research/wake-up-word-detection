# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of EdgeWuW.
#
# EdgeWuW is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# EdgeWuW is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with EdgeWuW. If not, see https://www.gnu.org/licenses/

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
except ImportError as e:
    raise ImportError("Self-supervised models (e.g. w2v-bert-2) need the optional requirements: "
                      "pip install -r requirements-ssl.txt") from e

from utils.model_utils import SerializableModule


# Pre-trained self-supervised encoder, downloaded from the Hugging Face Hub (MIT license)
W2V_BERT_2_MODEL = "facebook/w2v-bert-2.0"


class SelfAttention(nn.Module):
    """
    Multi-head scaled dot-product self-attention without output projection.
    """

    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        bs, seq_len, dim = x.shape

        # [bs, seq_len, dim] -> [bs, n_heads, seq_len, head_dim]
        q, k, v = [
            w(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            for w in (self.wq, self.wk, self.wv)
        ]
        out = F.scaled_dot_product_attention(q, k, v)

        # [bs, n_heads, seq_len, head_dim] -> [bs, seq_len, dim]
        return out.transpose(1, 2).reshape(bs, seq_len, dim)


class AttentionPooling(nn.Module):
    """
    Attentive pooling over time with a single learnable query vector.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.u = nn.Parameter(torch.ones(self.hidden_size), requires_grad=True)

    def forward(self, x):
        # Attention scores: [bs, seq_len]
        scores = torch.matmul(x, self.u) / (self.hidden_size ** 0.5)
        scores = F.softmax(scores, dim=1).unsqueeze(2)

        # Weighted sum over time: [bs, hidden_size]
        return torch.sum(x * scores, dim=1)


@SerializableModule.register_model('w2v-bert-2')
class Wav2VecBert2(SerializableModule):
    """
    Classifier on top of a frozen w2v-BERT 2.0 encoder.

    The hidden states of the encoder layers are combined with learnable weights, followed by
    self-attention, attentive pooling and a linear classifier. Only these layers are trained.
    It takes raw audio as input (use the `raw_audio` feature extractor). It is intended as a
    teacher for knowledge distillation (see `--teacher_checkpoint`), not for on-device use.
    """

    def __init__(self, input_shape, num_classes=2):
        super().__init__()

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(W2V_BERT_2_MODEL)
        self.encoder = Wav2Vec2BertModel.from_pretrained(W2V_BERT_2_MODEL)
        self.inner_dim = self.encoder.config.hidden_size # 1024
        self.n_layers = self.encoder.config.num_hidden_layers # 24 encoder layers
        self.n_classes = num_classes
        self.n_heads = 8
        self.sr = 16000

        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Weights to combine the hidden states of the encoder layers
        self.emb_weights = nn.Parameter(data=torch.ones(self.n_layers), requires_grad=True)

        # Attention layer
        self.attention = SelfAttention(dim=self.inner_dim, n_heads=self.n_heads)

        # Attention pooling
        self.attention_pooling = AttentionPooling(hidden_size=self.inner_dim)

        # Classification layer
        self.classification = nn.Sequential(
            nn.Linear(self.inner_dim, self.n_classes)
        )

    def forward(self, x):
        # Filterbank features computed by the Hugging Face feature extractor (on CPU)
        device = self.emb_weights.device
        features = self.feature_extractor(
            list(x.detach().cpu().numpy()),
            sampling_rate=self.sr,
            return_tensors="pt"
            ).to(device) # [bs, n_frames, 160]
        embeddings = self.encoder(**features, output_hidden_states=True)

        # Hidden states of the encoder layers: [bs, seq_len, hidden_dim, n_layers]
        embeddings = torch.stack(embeddings['hidden_states'], dim=3)[:, :, :, -self.n_layers:]

        # Weighted sum of the hidden states: [bs, seq_len, hidden_dim]
        weighted_embeddings = torch.sum(embeddings.mul(F.softmax(self.emb_weights, dim=-1)), dim=-1)

        # Self-attention: [bs, seq_len, hidden_dim]
        att_embedding = self.attention(weighted_embeddings)

        # Attentive pooling: [bs, hidden_dim]
        final_embedding = self.attention_pooling(att_embedding)

        return self.classification(final_embedding) # [bs, n_classes]
