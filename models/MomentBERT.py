import torch
from torch import nn
from transformers import BertModel, BertConfig
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Time-series forecasting model using BERT encoder for bidirectional context learning.

    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.test_pred_len
        self.use_norm = configs.use_norm
        patch_len = configs.input_token_len
        stride = patch_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # BERT Encoder
        self.bert_config = BertConfig(
            hidden_size=configs.d_model,
            num_attention_heads=configs.n_heads,
            num_hidden_layers=configs.e_layers,
            intermediate_size=configs.d_ff,
            hidden_dropout_prob=configs.dropout,
            attention_probs_dropout_prob=configs.dropout,
            max_position_embeddings=configs.seq_len
        )
        self.bert = BertModel(self.bert_config)

        # Prediction Head
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(self.head_nf, configs.test_pred_len, head_dropout=configs.dropout)

    def forecast(self, x, x_mark, y_mark):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        # do patching and embedding
        x = x.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x)

        # BERT Encoder
        enc_out = enc_out.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        enc_out, _ = self.bert(inputs_embeds=enc_out)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * \
                    (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + \
                    (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x, x_mark, y_mark):
        dec_out = self.forecast(x, x_mark, y_mark)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
