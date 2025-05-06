'''
#Inverted PatcTST

import torch
import torch.nn as nn
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.Embed import PatchEmbedding


class InvertedTransformerEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, embed, freq, dropout):
        super(InvertedTransformerEmbedding, self).__init__()
        # Initialize embedding layers for your data
        self.seq_len = seq_len
        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        self.dropout = dropout

    def forward(self, x, x_mark):
        """
        x: Input time series data
        x_mark: Time-related features
        """
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        return x, means, stdev


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch length for patch_embedding
        stride: int, stride for patch_embedding
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # Inverted Embedding Layer
        self.inverted_embedding = InvertedTransformerEmbedding(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder Layers
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Step 1: Inverted Embedding
        x_enc, means, stdev = self.inverted_embedding(x_enc, x_mark_enc)

        # Step 2: Patching
        x_enc = x_enc.permute(0, 2, 1)  # Shape: [bs, nvars, seq_len]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Step 3: Encoder
        enc_out, _ = self.encoder(enc_out)

        # Step 4: Reshaping and Permuting
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Step 5: Decoder (Prediction)
        dec_out = self.head(enc_out)  # [bs, nvars, target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # Step 6: De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

# Define FlattenHead to match with PatchTST model:
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

# Define Transpose class to handle dimensional transpositions:
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: 
            return x.transpose(*self.dims).contiguous()
        else: 
            return x.transpose(*self.dims)'''
            
'''

#Dynamic Window PatchTST

import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.e_layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.activation = configs.activation
       
        # Determine dynamic patch size
        self.patch_len, self.stride = self.dynamic_patch_size(configs.seq_len)
        padding = self.stride

        # Patching and embedding
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, padding, self.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=self.dropout,
                                      output_attention=configs.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(self.d_model), Transpose(1, 2))
        )

        # Prediction Head
        self.head_nf = self.d_model * int((self.seq_len - self.patch_len) / self.stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, self.pred_len, head_dropout=self.dropout)

    def dynamic_patch_size(self, seq_len):
        """Dynamically selects patch size based on sequence length."""
        if seq_len <= 64:
            return 8, 4
        elif seq_len <= 128:
            return 16, 8
        elif seq_len <= 256:
            return 32, 16
        else:
            return 64, 32

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = (x_enc - means) / (torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5))

        # Patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        return dec_out '''
        
'''

Convolutional Encoding with PatchTST

import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

class ConvTokenEncoding(nn.Module):
    """
    Convolutional token encoding layer applied after patching.
    """
    def __init__(self, d_model, kernel_size=3, stride=1, padding=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
   
    def forward(self, x):
        # Input: [batch_size, patch_num, d_model]
        x = x.permute(0, 2, 1)  # [batch, d_model, patch_num]
        x = self.conv(x)        # Apply 1D convolution
        x = self.norm(x)        # Batch normalization
        x = self.activation(x)  # Non-linearity
        x = self.dropout(x)     # Dropout
        x = x.permute(0, 2, 1)  # Restore shape: [batch, patch_num, d_model]
        return x

class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8, kernel_size=3):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
       
        # Patch Embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, stride, configs.dropout)
       
        # Convolutional Token Encoding
        self.conv_token_encoding = ConvTokenEncoding(
            d_model=configs.d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dropout=configs.dropout
        )
       
        # Transformer Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
       
        # Prediction Head
        self.head = nn.Linear(configs.d_model, self.pred_len)
   
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        # Patch embedding
        enc_out, _ = self.patch_embedding(x_enc)
       
        # Convolutional Token Encoding
        enc_out = self.conv_token_encoding(enc_out)
       
        # Transformer Encoder
        enc_out, _ = self.encoder(enc_out)
       
        # Prediction
        dec_out = self.head(enc_out[:, -1, :])  # Predict using last encoded token
        return dec_out  '''
    
    
import torch
from torch import nn
from layers.Transformer_EncDec import TimerBlock, TimerLayer
from layers.SelfAttention_Family import AttentionLayer, TimeAttention
from layers.Embed import PatchEmbedding

class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.input_token_len = configs.input_token_len
        self.use_norm = configs.use_norm
        self.pred_len = configs.test_pred_len
        self.patch_len = patch_len
        self.stride = stride

        # Patch Embedding Layer
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, stride, configs.dropout
        )

        # Encoder
        self.encoder = TimerBlock(
            [
                TimerLayer(
                    AttentionLayer(
                        TimeAttention(False, attention_dropout=configs.dropout, 
                                      output_attention=False, d_model=configs.d_model, 
                                      num_heads=configs.n_heads), configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Output Head
        self.head = nn.Linear(configs.d_model, configs.input_token_len)

    def forecast(self, x, x_mark, y_mark):
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        
        # Reshape Input: [B, L, C] → [B, C, L]
        x = x.permute(0, 2, 1)

        # Patch Embedding: [B * C, N, D]
        enc_out, n_vars = self.patch_embedding(x)
        B = x.shape[0]  # Batch size
        N = enc_out.shape[1]  # Number of patches (tokens)

        # Encoder: Pass enc_out, n_vars, n_tokens
        enc_out, attns = self.encoder(enc_out, n_vars=n_vars, n_tokens=N)

        # Reshape Back: [B, C, N * P] → [B, L, C]
        enc_out = enc_out.reshape(B, n_vars, -1)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # Final Normalization
        dec_out = dec_out[:, -self.pred_len:, :]
        if self.use_norm:
            dec_out = dec_out * stdev + means
        
        return dec_out

    def forward(self, x, x_mark, y_mark):
        return self.forecast(x, x_mark, y_mark)




