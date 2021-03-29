import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_attn_pad_mask(seq_q, seq_k):
    # seq_q: [batch_size, seq_len, n_feature]
    # seq_k: [batch_size, seq_len, n_feature]

    len_q = seq_q.size(1)
    numpy_seq_k = seq_k.detach().cpu().numpy()
    padding_rows_cols = np.where(~numpy_seq_k.any(axis=2))
    pad_attn_mask = np.zeros((numpy_seq_k.shape[:2]))
    pad_attn_mask[padding_rows_cols[0], padding_rows_cols[1]] = 1
    pad_attn_mask = torch.from_numpy(pad_attn_mask).unsqueeze(1).expand(-1, len_q, -1)  # pad_attn_mask: [batch_size, seq_len, seq_len]
    return pad_attn_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)  # pe: [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # position: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # div_term: [d_model / 2]
        pe[:, 0::2] = torch.sin(position * div_term)  # pe: [max_len, d_model]
        pe[:, 1::2] = torch.cos(position * div_term)  # pe: [max_len, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)  # pe: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]

        return x + self.pe[:x.size(0), :]  # [seq_len, batch_size, d_model]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, _config, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.config = _config

        self.temperature = temperature

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, nhead, len_q, d_k]
        # K: [batch_size, nhead, len_k, d_k]
        # V: [batch_size, nhead, len_v(=len_k), d_k]
        # attn_mask: [batch_size, nhead, seq_len, seq_len]

        Q.to(self.config['device'])
        K.to(self.config['device'])
        V.to(self.config['device'])
        attn_mask.to(self.config['device'])

        score = torch.matmul(Q, K.transpose(2, 3)) / np.sqrt(self.temperature)  # score: [batch_size, nhead, len_q, len_k]
        score.masked_fill_(attn_mask, -1e9)

        attn = F.softmax(score, dim=-1)
        output = torch.matmul(attn, V)  # output: [batch_size, nhead, len_q, d_v]

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, _config, d_model, nhead, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.config = _config

        self.nhead = nhead
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, self.d_k * self.nhead, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k * self.nhead, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v * self.nhead, bias=False)

        self.attention = ScaledDotProductAttention(_config=self.config, temperature=self.d_k ** 0.5)
        self.fc = nn.Linear(self.nhead * self.d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q_input, K_input, V_input, attn_mask):
        # Q_input: [batch_size, len_q, d_model]
        # K_input: [batch_size, len_k, d_model]
        # V_input: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]

        Q_input.to(self.config['device'])
        K_input.to(self.config['device'])
        V_input.to(self.config['device'])
        attn_mask.to(self.config['device'])

        residual, batch_size = Q_input, Q_input.size(0)

        # [batch_size, len, d_model] -> [batch_size, len, d_k/d_v * nhead] -> [batch_size, len, nhead, d_k/d_v] -> [batch_size, nhead, len, d_k/d_v]
        Q = self.W_Q(Q_input).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)  # Q: [batch_size, nhead, len_q, d_k]
        K = self.W_K(K_input).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)  # K: [batch_size, nhead, len_k, d_k]
        V = self.W_V(V_input).view(batch_size, -1, self.nhead, self.d_v).transpose(1, 2)  # V: [batch_size, nhead, len_v(=len_k), d_k]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1)  # attn_mask: [batch_size, nhead, seq_len, seq_len]

        q, attn = self.attention(Q, K, V, attn_mask)
        q = q.transpose(1, 2).reshape(batch_size, -1, self.nhead * self.d_v)  # q: [batch_size, len_q, nhead * d_v]
        q = self.fc(q)  # q: [batch_size, len_q, d_model]
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        # input: [batch_size, seq_len, d_model]

        residual = input

        output = self.fc(input)  # output: [batch_size, seq_len, d_model]
        output = self.layer_norm(output + residual)  # output: [batch_size, seq_len, d_model]

        return output


class EncoderLayer(nn.Module):
    def __init__(self, _config, d_model, nhead, d_ff, d_k, d_v):
        super(EncoderLayer, self).__init__()
        self.config = _config

        self.self_attention = MultiHeadAttention(_config=self.config, d_model=d_model, nhead=nhead, d_k=d_k, d_v=d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_ff)

    def forward(self, enc_input, enc_self_attn_mask):
        # enc_input: [batch_size, src_len, d_model]
        # enc_self_attn_mask: [batch_size, src_len, src_len]

        enc_input.to(self.config['device'])
        enc_self_attn_mask.to(self.config['device'])

        # enc_output: [batch_size, src_len, d_model], attn: [batch_size, nhead, src_len, src_len]
        enc_output, attn = self.self_attention(enc_input, enc_input, enc_input, enc_self_attn_mask)
        enc_output = self.pos_ffn(enc_output)  # enc_output: [batch_size, src_len, d_model]
        return enc_output, attn


class Encoder(nn.Module):
    def __init__(self, _config, src_n_feature, d_model, nhead, n_layer, d_ff, d_k, d_v, dropout=0.1):
        super(Encoder, self).__init__()
        self.config = _config

        self.src_emb = nn.Linear(src_n_feature, d_model)
        self.pos_emb = PositionalEncoding(d_model=d_model)
        self.layers = nn.ModuleList([EncoderLayer(_config=self.config, d_model=d_model, nhead=nhead, d_ff=d_ff, d_k=d_k, d_v=d_v) for _ in range(n_layer)])

    def forward(self, enc_input):
        # enc_input: [batch_size, src_len, src_n_feature]

        enc_input.to(self.config['device'])

        enc_output = self.src_emb(enc_input)  # enc_output: [batch_size, src_len, d_model]
        enc_output = self.pos_emb(enc_output.transpose(0, 1)).transpose(0, 1)  # enc_output: [batch_size, src_len, d_model]

        enc_self_attn_mask = get_attn_pad_mask(enc_input, enc_input)  # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attn_mask = enc_self_attn_mask.type(torch.bool).to(self.config['device'])
        enc_self_attns = []
        for layer in self.layers:
            enc_output, enc_self_attn = layer(enc_output, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_output, enc_self_attns


class Transformer(nn.Module):
    def __init__(self, _config, src_n_feature, tgt_n_feature, max_seq_len, d_model, nhead, n_layer, d_ff, d_k, d_v, dropout=0.1):
        super(Transformer, self).__init__()
        self.config = _config

        self.encoder = Encoder(
            _config=self.config,
            src_n_feature=src_n_feature,
            d_model=d_model,
            nhead=nhead,
            n_layer=n_layer,
            d_ff=d_ff,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )
        self.fc = nn.Linear(d_model * max_seq_len, tgt_n_feature)

    def forward(self, enc_input):
        # enc_input: [batch_size, src_len, src_n_feature]

        enc_input.to(self.config['device'])

        enc_output, enc_self_attn = self.encoder(enc_input)  # enc_output: [batch_size, src_len, d_model]
        enc_output = torch.reshape(enc_output, (enc_output.shape[0], -1))  # enc_output: [batch_size, src_len * d_model]
        result = self.fc(enc_output)  # result: [batch_size, tgt_n_feature]
        return result


class ScheduledOptim():
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer =  optimizer
        self.n_warmup_steps = n_warmup_steps
        self.init_lr = np.power(d_model, -0.5)
        self.n_current_steps = 0

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
