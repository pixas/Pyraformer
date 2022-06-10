from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .Modules import ScaledDotProductAttention
from efficient_attention import AMLP, ABC

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            if len(mask.size()) == 3:
                mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class EfficientAttnLayer(nn.Module):
    def __init__(self, opt, normalize_before=True):
        super().__init__()
        self.normalize_before = normalize_before

        self.d_k = opt.d_k
        self.d_v = opt.d_v

        dropout = opt.dropout
        self.attn = self.build_self_attention(opt)
        
        self.layer_norm = nn.LayerNorm(opt.d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None):
        # d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # bsz, q_len, _ = q.shape
        # k_len = k.shape[1]
        residual = q
        
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        if self.normalize_before:
            q = self.layer_norm(q)

        output, attn = self.attn(q, k, v, attn_mask=mask)
        output = output.transpose(0, 1)
        output = self.dropout(output)
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn
    
    def build_self_attention(self, opt):
        attn_type = opt.enc_attn_type
        if attn_type == 'amlp':
            attn = AMLP(
                num_heads=opt.n_head,
                embed_dim=opt.d_model,
                ffn_dimension=opt.enc_amlp_dim,
                activation_fn=opt.enc_amlp_fn
            )
        elif attn_type == 'abc':
            attn = ABC(
                num_heads=opt.n_head,
                embed_dim=opt.d_model,
                num_landmarks=opt.enc_landmarks
                
            )
        else:
            attn = MultiHeadAttention(
                opt.n_head,
                opt.d_model,
                opt.d_k,
                opt.d_v,
                dropout = opt.dropout,
                normalize_before=self.normalize_before
            )
        
        return attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        #self.layer_norm = GraphNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x

