import os
import time
import math
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

DATASET_PATH = os.path.join('.', 'Dataset')
DATA_FOLDS_FILE = os.path.join(DATASET_PATH, 'data_folds.pkl')
EMB_LIST_FILE = os.path.join(DATASET_PATH, 'word_embedding_list.pkl')
T_FILE = os.path.join(DATASET_PATH, 'language_sdk.pkl')
A_FILE = os.path.join(DATASET_PATH, 'covarep_features_sdk.pkl')
V_FILE = os.path.join(DATASET_PATH, 'openface_features_sdk.pkl')
LABEL_FILE = os.path.join(DATASET_PATH, 'humor_label_sdk.pkl')


def load_pickle(pickle_file):
    pickle_data = None
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data {} : {}'.format(pickle_file, e))
    return pickle_data


EMB_LIST_SDK = load_pickle(EMB_LIST_FILE)
T_SDK = load_pickle(T_FILE)
A_SDK = load_pickle(A_FILE)
V_SDK = load_pickle(V_FILE)
LABEL_SDK = load_pickle(LABEL_FILE)

TEXT_N_FEATURE = 300
AUDIO_N_FEATURE = 81
VIDEO_N_FEATURE = 371

MAX_CONTEXT_LEN = 5
MAX_SENTENCE_LEN = 20

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_EPOCH = 15
BATCH_SIZE = 128
SHUFFLE = True

LEARNING_RATE = random.choice([0.001, 0.002, 0.005, 0.008, 0.01])

# unimodal context network config
UNI_T_IN_DIM = TEXT_N_FEATURE
UNI_A_IN_DIM = AUDIO_N_FEATURE
UNI_V_IN_DIM = VIDEO_N_FEATURE

UNI_T_N_HIDDEN = random.choice([32, 64, 88, 128, 156, 256])
UNI_A_N_HIDDEN = random.choice([8, 16, 32, 48, 64, 80])
UNI_V_N_HIDDEN = random.choice([8, 16, 32, 48, 64, 80])

USE_T_CONTEXT = True
USE_A_CONTEXT = True
USE_V_CONTEXT = True

# multimodal context network config
MUL_T_IN_DIM = MAX_CONTEXT_LEN * UNI_T_N_HIDDEN
MUL_A_IN_DIM = MAX_CONTEXT_LEN * UNI_A_N_HIDDEN
MUL_V_IN_DIM = MAX_CONTEXT_LEN * UNI_V_N_HIDDEN

MUL_T_DROPOUT = random.choice([0.0, 0.1, 0.2, 0.5])
MUL_A_DROPOUT = random.choice([0.0, 0.2, 0.5, 0.1])
MUL_V_DROPOUT = random.choice([0.0, 0.2, 0.5, 0.1])

SRC_N_FEATURE = UNI_T_N_HIDDEN + UNI_A_N_HIDDEN + UNI_V_N_HIDDEN
MAX_SEQ_LEN = MAX_CONTEXT_LEN
D_MODEL = 512
NHEAD = 8
N_LAYER = 6
D_FF = 2048
D_K = 64
D_V = 64

MUL_DROPOUT = random.choice([0.0, 0.2, 0.5, 0.1])

# memory fusion network config
MFN_T_IN_DIM = TEXT_N_FEATURE
MFN_A_IN_DIM = AUDIO_N_FEATURE
MFN_V_IN_DIM = VIDEO_N_FEATURE

MFN_T_N_HIDDEN = random.choice([32, 64, 88, 128, 156, 256])
MFN_A_N_HIDDEN = random.choice([8, 16, 32, 48, 64, 80])
MFN_V_N_HIDDEN = random.choice([8, 16, 32, 48, 64, 80])
MFN_N_HIDDEN = MFN_T_N_HIDDEN + MFN_A_N_HIDDEN + MFN_V_N_HIDDEN
MFN_WINDOW_DIM = 2
MFN_MEM_DIM = random.choice([64, 128, 256, 300, 400])

MFN_ATTN_IN_DIM = MFN_N_HIDDEN * MFN_WINDOW_DIM

MFN_NN1_DIM = random.choice([32, 64, 128, 256])
MFN_NN1_DROPOUT = random.choice([0.0, 0.2, 0.5, 0.7])

MFN_NN2_DIM = random.choice([32, 64, 128, 256])
MFN_NN2_DROPOUT = random.choice([0.0, 0.2, 0.5, 0.7])

MFN_GAMMA_IN_DIM = MFN_ATTN_IN_DIM + MFN_MEM_DIM

MFN_GAMMA1_DIM = random.choice([32, 64, 128, 256])
MFN_GAMMA1_DROPOUT = random.choice([0.0, 0.2, 0.5, 0.7])

MFN_GAMMA2_DIM = random.choice([32, 64, 128, 256])
MFN_GAMMA2_DROPOUT = random.choice([0.0, 0.2, 0.5, 0.7])

MFN_OUTPUT_IN_DIM = MFN_N_HIDDEN + MFN_MEM_DIM
MFN_OUTPUT_HIDDEN_DIM = random.choice([32, 64, 128, 256])
MFN_OUTPUT_DROPOUT = random.choice([0.0, 0.2, 0.5, 0.7])
MFN_OUTPUT_DIM = 1

USE_PUNCHLINE = True
USE_T_PUNCHLINE = True
USE_A_PUNCHLINE = True
USE_V_PUNCHLINE = True

# contextual memory fusion network config
USE_CONTEXT = True


class HumorDataset(Dataset):
    def __init__(self, id_list):
        self.id_list = id_list

        self.t_dim = TEXT_N_FEATURE
        self.a_dim = AUDIO_N_FEATURE
        self.v_dim = VIDEO_N_FEATURE
        self.all_dim = self.t_dim + self.a_dim + self.v_dim

        self.max_context_len = MAX_CONTEXT_LEN
        self.max_sentence_len = MAX_SENTENCE_LEN

    def padded_t_feature(self, seq):
        seq = seq[:self.max_sentence_len]
        padded_t = np.concatenate((np.zeros(self.max_sentence_len - len(seq)), seq), axis=0)
        padded_t = np.array([EMB_LIST_SDK[int(t_id)] for t_id in padded_t])
        return padded_t

    def padded_a_feature(self, seq):
        seq = seq[:self.max_sentence_len]
        return np.concatenate((np.zeros((self.max_sentence_len - len(seq), self.a_dim)), seq), axis=0)

    def padded_v_feature(self, seq):
        seq = seq[:self.max_sentence_len]
        return np.concatenate((np.zeros((self.max_sentence_len - len(seq), self.v_dim)), seq), axis=0)

    def padded_context_feature(self, t_context, a_context, v_context):
        t_context = t_context[-self.max_context_len:]
        a_context = a_context[-self.max_context_len:]
        v_context = v_context[-self.max_context_len:]

        padded_context = []
        for i in range(len(t_context)):
            padded_seq_t = self.padded_t_feature(t_context[i])
            padded_seq_a = self.padded_a_feature(a_context[i])
            padded_seq_v = self.padded_v_feature(v_context[i])
            padded_context.append(np.concatenate((padded_seq_t, padded_seq_a, padded_seq_v), axis=1))

        padded_context_len = self.max_context_len - len(padded_context)
        padded_context = np.array(padded_context)

        if not padded_context.any():
            return np.zeros((self.max_context_len, self.max_sentence_len, self.all_dim))
        return np.concatenate((np.zeros((padded_context_len, self.max_sentence_len, self.all_dim)), padded_context), axis=0)

    def padded_punchline_feature(self, t_punchline, a_punchline, v_punchline):
        padded_seq_t = self.padded_t_feature(t_punchline)
        padded_seq_a = self.padded_a_feature(a_punchline)
        padded_seq_v = self.padded_v_feature(v_punchline)
        return np.concatenate((padded_seq_t, padded_seq_a, padded_seq_v), axis=1)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, item):
        hid = self.id_list[item]

        t_context = np.array(T_SDK[hid]['context_embedding_indexes'])
        a_context = np.array(A_SDK[hid]['context_features'])
        v_context = np.array(V_SDK[hid]['context_features'])

        t_punchline = np.array(T_SDK[hid]['punchline_embedding_indexes'])
        a_punchline = np.array(A_SDK[hid]['punchline_features'])
        v_punchline = np.array(V_SDK[hid]['punchline_features'])

        x_c = torch.FloatTensor(self.padded_context_feature(t_context, a_context, v_context))
        x_p = torch.FloatTensor(self.padded_punchline_feature(t_punchline, a_punchline, v_punchline))
        y = torch.FloatTensor([LABEL_SDK[hid]])

        # x_c: [batch_size, max_context_len, max_sentence_len, n_feature]
        # x_p: [batch_size, max_sentence_len, n_feature]
        return x_c, x_p, y


# ---------- Transformer ----------


def get_attn_pad_mask(seq_q, seq_k):
    # seq_q: [batch_size, seq_len, n_feature]
    # seq_k: [batch_size, seq_len, n_feature]

    len_q = seq_q.size(1)
    numpy_seq_k = seq_k.detach().cpu().numpy()
    padding_rows_cols = np.where(~numpy_seq_k.any(axis=2))
    pad_attn_mask = np.zeros((numpy_seq_k.shape[:2]))
    pad_attn_mask[padding_rows_cols[0], padding_rows_cols[1]] = 1
    pad_attn_mask = torch.from_numpy(pad_attn_mask).unsqueeze(1).expand(-1, len_q, -1).to(DEVICE)  # pad_attn_mask: [batch_size, seq_len, seq_len]
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
    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, nhead, len_q, d_k]
        # K: [batch_size, nhead, len_k, d_k]
        # V: [batch_size, nhead, len_v(=len_k), d_k]
        # attn_mask: [batch_size, nhead, seq_len, seq_len]

        Q.to(DEVICE)
        K.to(DEVICE)
        V.to(DEVICE)
        attn_mask.to(DEVICE)

        score = torch.matmul(Q, K.transpose(2, 3)) / np.sqrt(self.temperature)  # score: [batch_size, nhead, len_q, len_k]
        score.masked_fill_(attn_mask, -1e9)

        attn = F.softmax(score, dim=-1)
        output = torch.matmul(attn, V)  # output: [batch_size, nhead, len_q, d_v]

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, d_k, d_v):
        super(MultiHeadAttention, self).__init__()

        self.nhead = nhead
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, self.d_k * self.nhead, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k * self.nhead, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v * self.nhead, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.fc = nn.Linear(self.nhead * self.d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q_input, K_input, V_input, attn_mask):
        # Q_input: [batch_size, len_q, d_model]
        # K_input: [batch_size, len_k, d_model]
        # V_input: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]

        Q_input.to(DEVICE)
        K_input.to(DEVICE)
        V_input.to(DEVICE)
        attn_mask.to(DEVICE)

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
    def __init__(self, d_model, nhead, d_ff, d_k, d_v):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model=d_model, nhead=nhead, d_k=d_k, d_v=d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_ff)

    def forward(self, enc_input, enc_self_attn_mask):
        # enc_input: [batch_size, src_len, d_model]
        # enc_self_attn_mask: [batch_size, src_len, src_len]

        enc_input.to(DEVICE)
        enc_self_attn_mask.to(DEVICE)

        # enc_output: [batch_size, src_len, d_model], attn: [batch_size, nhead, src_len, src_len]
        enc_output, attn = self.self_attention(enc_input, enc_input, enc_input, enc_self_attn_mask)
        enc_output = self.pos_ffn(enc_output)  # enc_output: [batch_size, src_len, d_model]
        return enc_output, attn


class Encoder(nn.Module):
    def __init__(self, src_n_feature, d_model, nhead, n_layer, d_ff, d_k, d_v, dropout=0.1):
        super(Encoder, self).__init__()

        self.src_emb = nn.Linear(src_n_feature, d_model)
        self.pos_emb = PositionalEncoding(d_model=d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, nhead=nhead, d_ff=d_ff, d_k=d_k, d_v=d_v) for _ in range(n_layer)])

    def forward(self, enc_input):
        # enc_input: [batch_size, src_len, src_n_feature]

        enc_input.to(DEVICE)

        enc_output = self.src_emb(enc_input)  # enc_output: [batch_size, src_len, d_model]
        enc_output = self.pos_emb(enc_output.transpose(0, 1)).transpose(0, 1)  # enc_output: [batch_size, src_len, d_model]

        enc_self_attn_mask = get_attn_pad_mask(enc_input, enc_input)  # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attn_mask = enc_self_attn_mask.type(torch.bool).to(DEVICE)
        enc_self_attns = []
        for layer in self.layers:
            enc_output, enc_self_attn = layer(enc_output, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_output, enc_self_attns


class Transformer(nn.Module):
    def __init__(self, src_n_feature, tgt_n_feature, max_seq_len, d_model, nhead, n_layer, d_ff, d_k, d_v, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
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

        enc_input.to(DEVICE)

        enc_output, enc_self_attn = self.encoder(enc_input)  # enc_output: [batch_size, src_len, d_model]
        enc_output = torch.reshape(enc_output, (enc_output.shape[0], -1))  # enc_output: [batch_size, src_len * d_model]
        result = self.fc(enc_output)  # result: [batch_size, tgt_n_feature]
        return result


# ---------- Transformer End ----------


class UnimodalContextNet(nn.Module):
    def __init__(self):
        super(UnimodalContextNet, self).__init__()

        self.t_lstm = nn.LSTM(input_size=UNI_T_IN_DIM, hidden_size=UNI_T_N_HIDDEN, batch_first=True)
        self.a_lstm = nn.LSTM(input_size=UNI_A_IN_DIM, hidden_size=UNI_A_N_HIDDEN, batch_first=True)
        self.v_lstm = nn.LSTM(input_size=UNI_V_IN_DIM, hidden_size=UNI_V_N_HIDDEN, batch_first=True)

    def forward(self, x_c):
        # x_c: [batch_size, context_len, sentence_len, n_feature]

        x_c.to(DEVICE)

        old_batch_size, context_len, seq_len, num_feats = x_c.size()
        new_batch_size = old_batch_size * context_len
        x_c = torch.reshape(x_c, (new_batch_size, seq_len, num_feats))  # x_c: [batch_size * context_len, sentence_len, n_feature]

        t_context = x_c[:, :, :UNI_T_IN_DIM]  # t_context: [batch_size * context_len, sentence_len, t_n_feature]
        a_context = x_c[:, :, UNI_T_IN_DIM:UNI_T_IN_DIM + UNI_A_IN_DIM]  # a_context: [batch_size * context_len, sentence_len, a_n_feature]
        v_context = x_c[:, :, UNI_T_IN_DIM + UNI_A_IN_DIM:]  # v_context: [batch_size * context_len, sentence_len, v_n_feature]

        if not USE_T_CONTEXT:
            t_context = torch.zeros_like(t_context, requires_grad=True)
        if not USE_A_CONTEXT:
            a_context = torch.zeros_like(a_context, requires_grad=True)
        if not USE_V_CONTEXT:
            t_context = torch.zeros_like(v_context, requires_grad=True)

        t_h0 = torch.zeros(new_batch_size, UNI_T_N_HIDDEN).unsqueeze(0).to(DEVICE)  # t_h0: [1, batch_size * context_len, UNI_T_N_HIDDEN]
        t_c0 = torch.zeros(new_batch_size, UNI_T_N_HIDDEN).unsqueeze(0).to(DEVICE)  # t_c0: [1, batch_size * context_len, UNI_T_N_HIDDEN]
        t_o, (t_hn, t_cn) = self.t_lstm(t_context, (t_h0, t_c0))  # t_hn: [1, batch_size * context_len, UNI_T_N_HIDDEN]

        a_h0 = torch.zeros(new_batch_size, UNI_A_N_HIDDEN).unsqueeze(0).to(DEVICE)  # a_h0: [1, batch_size * context_len, UNI_A_N_HIDDEN]
        a_c0 = torch.zeros(new_batch_size, UNI_A_N_HIDDEN).unsqueeze(0).to(DEVICE)  # a_c0: [1, batch_size * context_len, UNI_A_N_HIDDEN]
        a_o, (a_hn, a_cn) = self.a_lstm(a_context, (a_h0, a_c0))  # a_hn: [1, batch_size * context_len, UNI_A_N_HIDDEN]

        v_h0 = torch.zeros(new_batch_size, UNI_V_N_HIDDEN).unsqueeze(0).to(DEVICE)  # v_h0: [1, batch_size * context_len, UNI_V_N_HIDDEN]
        v_c0 = torch.zeros(new_batch_size, UNI_V_N_HIDDEN).unsqueeze(0).to(DEVICE)  # v_c0: [1, batch_size * context_len, UNI_V_N_HIDDEN]
        v_o, (v_hn, v_cn) = self.v_lstm(v_context, (v_h0, v_c0))  # v_hn: [1, batch_size * context_len, UNI_V_N_HIDDEN]

        t_result = torch.reshape(t_hn, (old_batch_size, context_len, -1))  # t_result: [batch_size, context_len, UNI_T_N_HIDDEN]
        a_result = torch.reshape(a_hn, (old_batch_size, context_len, -1))  # a_result: [batch_size, context_len, UNI_A_N_HIDDEN]
        v_result = torch.reshape(v_hn, (old_batch_size, context_len, -1))  # v_result: [batch_size, context_len, UNI_V_N_HIDDEN]

        return t_result, a_result, v_result


class MultimodalContextNet(nn.Module):
    def __init__(self):
        super(MultimodalContextNet, self).__init__()

        self.t_fc = nn.Linear(MUL_T_IN_DIM, MFN_T_N_HIDDEN)
        self.t_dropout = nn.Dropout(MUL_T_DROPOUT)

        self.a_fc = nn.Linear(MUL_A_IN_DIM, MFN_A_N_HIDDEN)
        self.a_dropout = nn.Dropout(MUL_A_DROPOUT)

        self.v_fc = nn.Linear(MUL_V_IN_DIM, MFN_V_N_HIDDEN)
        self.v_dropout = nn.Dropout(MUL_V_DROPOUT)

        self.self_attention = Transformer(
            src_n_feature=SRC_N_FEATURE,
            tgt_n_feature=MFN_MEM_DIM,
            max_seq_len=MAX_SEQ_LEN,
            d_model=D_MODEL,
            nhead=NHEAD,
            n_layer=N_LAYER,
            d_ff=D_FF,
            d_k=D_K,
            d_v=D_V
        )
        self.dropout = nn.Dropout(MUL_DROPOUT)

    def forward(self, uni_t, uni_a, uni_v):
        # uni_t: [batch_size, context_len, UNI_T_N_HIDDEN]
        # uni_a: [batch_size, context_len, UNI_A_N_HIDDEN]
        # uni_v: [batch_size, context_len, UNI_V_N_HIDDEN]

        uni_t.to(DEVICE)
        uni_a.to(DEVICE)
        uni_v.to(DEVICE)

        reshaped_uni_t = uni_t.reshape((uni_t.shape[0], -1))  # reshaped_uni_t: [batch_size, context_len * UNI_T_N_HIDDEN]
        reshaped_uni_a = uni_a.reshape((uni_a.shape[0], -1))  # reshaped_uni_a: [batch_size, context_len * UNI_A_N_HIDDEN]
        reshaped_uni_v = uni_v.reshape((uni_v.shape[0], -1))  # reshaped_uni_v: [batch_size, context_len * UNI_V_N_HIDDEN]

        mfn_c_t = self.t_dropout(self.t_fc(reshaped_uni_t))  # mfn_ht_input: [batch_size, MFN_T_N_HIDDEN]
        mfn_c_a = self.a_dropout(self.a_fc(reshaped_uni_a))  # mfn_ha_input: [batch_size, MFN_A_N_HIDDEN]
        mfn_c_v = self.v_dropout(self.v_fc(reshaped_uni_v))  # mfn_hv_input: [batch_size, MFN_V_N_HIDDEN]

        concat = torch.cat([uni_t, uni_a, uni_v], dim=2)  # concat: [batch_size, context_len, hidden_size(t + a + v)]

        mfn_mem = self.dropout(self.self_attention(concat)).squeeze(0)  # mfn_mem: [batch_size, MFN_MEM_DIM]

        return mfn_c_t, mfn_c_a, mfn_c_v, mfn_mem


class MFN(nn.Module):
    def __init__(self):
        super(MFN, self).__init__()

        self.t_lstm = nn.LSTMCell(MFN_T_IN_DIM, MFN_T_N_HIDDEN)
        self.a_lstm = nn.LSTMCell(MFN_A_IN_DIM, MFN_A_N_HIDDEN)
        self.v_lstm = nn.LSTMCell(MFN_V_IN_DIM, MFN_V_N_HIDDEN)

        self.attn1_fc1 = nn.Linear(MFN_ATTN_IN_DIM, MFN_NN1_DIM)
        self.attn1_fc2 = nn.Linear(MFN_NN1_DIM, MFN_ATTN_IN_DIM)
        self.attn1_dropout = nn.Dropout(MFN_NN1_DROPOUT)

        self.attn2_fc1 = nn.Linear(MFN_ATTN_IN_DIM, MFN_NN2_DIM)
        self.attn2_fc2 = nn.Linear(MFN_NN2_DIM, MFN_MEM_DIM)
        self.attn2_dropout = nn.Dropout(MFN_NN2_DROPOUT)

        self.gamma1_fc1 = nn.Linear(MFN_GAMMA_IN_DIM, MFN_GAMMA1_DIM)
        self.gamma1_fc2 = nn.Linear(MFN_GAMMA1_DIM, MFN_MEM_DIM)
        self.gamma1_dropout = nn.Dropout(MFN_GAMMA1_DROPOUT)

        self.gamma2_fc1 = nn.Linear(MFN_GAMMA_IN_DIM, MFN_GAMMA2_DIM)
        self.gamma2_fc2 = nn.Linear(MFN_GAMMA2_DIM, MFN_MEM_DIM)
        self.gamma2_dropout = nn.Dropout(MFN_GAMMA2_DROPOUT)

        self.output_fc1 = nn.Linear(MFN_OUTPUT_IN_DIM, MFN_OUTPUT_HIDDEN_DIM)
        self.output_fc2 = nn.Linear(MFN_OUTPUT_HIDDEN_DIM, MFN_OUTPUT_DIM)
        self.output_dropout = nn.Dropout(MFN_OUTPUT_DROPOUT)

    def forward(self, x_p, c_t, c_a, c_v, mem):
        # x_p: [sentence_len, batch_size, n_feature]
        # c_t: [batch_size, MFN_T_N_HIDDEN]
        # c_a: [batch_size, MFN_A_N_HIDDEN]
        # c_v: [batch_size, MFN_V_N_HIDDEN]
        # mem: [batch_size, MFN_MEM_DIM]

        x_p.to(DEVICE)
        c_t.to(DEVICE)
        c_a.to(DEVICE)
        c_v.to(DEVICE)
        mem.to(DEVICE)

        t_punchline = x_p[:, :, :MFN_T_IN_DIM] # t_punchline: [sentence_len, batch_size, n_feature]
        a_punchline = x_p[:, :, MFN_T_IN_DIM:MFN_T_IN_DIM + MFN_A_IN_DIM]  # a_punchline: [sentence_len, batch_size, n_feature]
        v_punchline = x_p[:, :, MFN_T_IN_DIM + MFN_A_IN_DIM:]  # v_punchline: [sentence_len, batch_size, n_feature]

        if not USE_PUNCHLINE:
            t_punchline = torch.zeros_like(t_punchline, requires_grad=True)
            a_punchline = torch.zeros_like(a_punchline, requires_grad=True)
            v_punchline = torch.zeros_like(v_punchline, requires_grad=True)
        if not USE_T_PUNCHLINE:
            t_punchline = torch.zeros_like(t_punchline, requires_grad=True)
        if not USE_A_PUNCHLINE:
            a_punchline = torch.zeros_like(a_punchline, requires_grad=True)
        if not USE_V_PUNCHLINE:
            v_punchline = torch.zeros_like(v_punchline, requires_grad=True)

        t, n, d = x_p.shape  # x_p: [sentence_len, batch_size, n_feature]

        self.t_h = torch.zeros(n, MFN_T_N_HIDDEN).to(DEVICE)  # self.t_h: [n, MFN_T_N_HIDDEN]
        self.a_h = torch.zeros(n, MFN_A_N_HIDDEN).to(DEVICE)  # self.a_h: [n, MFN_A_N_HIDDEN]
        self.v_h = torch.zeros(n, MFN_V_N_HIDDEN).to(DEVICE)  # self.v_h: [n, MFN_V_N_HIDDEN]

        self.t_c = c_t.to(DEVICE)  # self.t_c: [batch_size, MFN_T_N_HIDDEN]
        self.a_c = c_a.to(DEVICE)  # self.a_c: [batch_size, MFN_A_N_HIDDEN]
        self.v_c = c_v.to(DEVICE)  # self.v_c: [batch_size, MFN_V_N_HIDDEN]

        self.mem = mem.to(DEVICE)  # self.mem: [batch_size, MFN_MEM_DIM]

        t_h_list, t_c_list = [], []
        a_h_list, a_c_list = [], []
        v_h_list, v_c_list = [], []

        mem_list = []

        for i in range(t):

            # previous time step
            pre_t_c = self.t_c  # pre_t_c: [batch_size, MFN_T_N_HIDDEN]
            pre_a_c = self.a_c  # pre_a_c: [batch_size, MFN_A_N_HIDDEN]
            pre_v_c = self.v_c  # pre_v_c: [batch_size, MFN_V_N_HIDDEN]

            # current time step
            cur_t_h, cur_t_c = self.t_lstm(t_punchline[i], (self.t_h, self.t_c))  # cur_t_h, cur_t_c: [batch_size, MFN_T_N_HIDDEN]
            cur_a_h, cur_a_c = self.a_lstm(a_punchline[i], (self.a_h, self.a_c))  # cur_a_h, cur_a_c: [batch_size, MFN_A_N_HIDDEN]
            cur_v_h, cur_v_c = self.v_lstm(v_punchline[i], (self.v_h, self.v_c))  # cur_v_h, cur_v_c: [batch_size, MFN_V_N_HIDDEN]

            # concatenate
            pre_c = torch.cat([pre_t_c, pre_a_c, pre_v_c], dim=1)  # pre_c: [batch_size, MFN_T_N_HIDDEN + MFN_A_N_HIDDEN + MFN_V_N_HIDDEN]
            cur_c = torch.cat([cur_t_c, cur_a_c, cur_v_c], dim=1)  # cur_c: [batch_size, MFN_T_N_HIDDEN + MFN_A_N_HIDDEN + MFN_V_N_HIDDEN]
            c_star = torch.cat([pre_c, cur_c], dim=1)  # c_star: [batch_size, MFN_ATTN_IN_DIM]
            attention = F.softmax(self.attn1_fc2(self.attn1_dropout(F.relu(self.attn1_fc1(c_star)))), dim=1)  # attention: [batch_size, MFN_ATTN_IN_DIM]
            attended = attention * c_star  # attended: [batch_size, MFN_ATTN_IN_DIM]
            c_hat = torch.tanh(self.attn2_fc2(self.attn2_dropout(F.relu(self.attn2_fc1(attended)))))  # c_hat: [batch_size, MFN_MEM_DIM]
            both = torch.cat([attended, self.mem], dim=1)  # both: [batch_size, MFN_GAMMA_IN_DIM]
            gamma1 = torch.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))  # gamma1: [batch_size, MFN_GAMMA_IN_DIM]
            gamma2 = torch.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))  # gamma2: [batch_size, MFN_GAMMA_IN_DIM]
            self.mem = gamma1 * self.mem + gamma2 * c_hat  # self.mem: [batch_size, MFN_MEM_DIM]
            mem_list.append(self.mem)

            # update
            self.t_h, self.t_c = cur_t_h, cur_t_c
            self.a_h, self.a_c = cur_a_h, cur_a_c
            self.v_h, self.v_c = cur_v_h, cur_v_c

            t_h_list.append(self.t_h)
            t_c_list.append(self.t_c)
            a_h_list.append(self.a_h)
            a_c_list.append(self.a_c)
            v_h_list.append(self.v_h)
            v_c_list.append(self.v_c)

        last_t_h = t_h_list[-1]
        last_a_h = a_h_list[-1]
        last_v_h = v_h_list[-1]
        last_mem = mem_list[-1]
        last_h = torch.cat([last_t_h, last_a_h, last_v_h, last_mem], dim=1)  # last_h: [batch_size, MFN_OUTPUT_IN_DIM]
        output = self.output_fc2(self.output_dropout(F.relu(self.output_fc1(last_h))))  # output: [batch_size, MFN_OUTPUT_DIM]
        return output


class C_MFN(nn.Module):
    def __init__(self):
        super(C_MFN, self).__init__()

        self.unimodal_context = UnimodalContextNet().to(DEVICE)
        self.multimodal_context = MultimodalContextNet().to(DEVICE)
        self.mfn = MFN().to(DEVICE)

    def forward(self, x_c, x_p, y):
        # x_c: [batch_size, context_len, sentence_len, n_feature]
        # x_p: [batch_size, sentence_len, n_feature]

        x_c.to(DEVICE)
        x_p.to(DEVICE)
        y.to(DEVICE)

        if not USE_CONTEXT:
            x_c = torch.zeros_like(x_c, requires_grad=True).to(DEVICE)

        # x_p: [sentence_len, batch_size, n_feature]
        x_p = x_p.permute(1, 0, 2)

        # uni_t: [batch_size, context_len, UNI_T_N_HIDDEN]
        # uni_a: [batch_size, context_len, UNI_T_A_HIDDEN]
        # uni_v: [batch_size, context_len, UNI_T_V_HIDDEN]
        uni_t, uni_a, uni_v = self.unimodal_context(x_c)

        # mfn_c_t: [batch_size, MFN_T_N_HIDDEN]
        # mfn_c_a: [batch_size, MFN_A_N_HIDDEN]
        # mfn_c_v: [batch_size, MFN_V_N_HIDDEN]
        # mfn_mem: [1, batch_size, MFN_MEM_DIM]
        mfn_c_t, mfn_c_a, mfn_c_v, mfn_mem = self.multimodal_context(uni_t, uni_a, uni_v)

        pred = self.mfn(x_p, mfn_c_t, mfn_c_a, mfn_c_v, mfn_mem)  # pred: [batch_size, MFN_OUTPUT_DIM]
        return pred


def show_config(optimizer, criterion):
    print('-' * 32 + 'base' + '-' * 32)
    print('device:{}'.format(DEVICE))
    print('epoch:{}\nbatch_size:{}\nshuffle:{}\nlearning_rate:{}'.format(N_EPOCH, BATCH_SIZE, SHUFFLE, LEARNING_RATE))
    print('language_feature_size:{}\naudio_feature_size:{}\nvideo_feature_size:{}'.format(TEXT_N_FEATURE, AUDIO_N_FEATURE, VIDEO_N_FEATURE))
    print('max_context_len:{}\nmax_sentence_len:{}'.format(MAX_CONTEXT_LEN, MAX_SENTENCE_LEN))
    context_list = []
    context_list_str = ''
    if USE_T_CONTEXT:
        context_list.append('language')
    if USE_A_CONTEXT:
        context_list.append('audio')
    if USE_V_CONTEXT:
        context_list.append('video')
    if context_list:
        context_list_str = '(' + ', '.join(context_list) + ')'
    print('use context:{}{}'.format(USE_CONTEXT, context_list_str if USE_CONTEXT else ''))
    punchline_list = []
    punchline_list_str = ''
    if USE_T_PUNCHLINE:
        punchline_list.append('language')
    if USE_A_PUNCHLINE:
        punchline_list.append('audio')
    if USE_V_PUNCHLINE:
        punchline_list.append('video')
    if punchline_list:
        punchline_list_str = '(' + ', '.join(punchline_list) + ')'
    print('use punchline:{}{}'.format(USE_PUNCHLINE, punchline_list_str if USE_PUNCHLINE else ''))
    print('optimizer:{}'.format(optimizer))
    print('criterion:{}'.format(criterion))

    print('-' * 32 + 'unimodal context network' + '-' * 32)
    print('language_in_dim:{}\naudio_in_dim:{}\nvideo_in_dim:{}'.format(UNI_T_IN_DIM, UNI_A_IN_DIM, UNI_V_IN_DIM))
    print('language_hidden_size:{}\naudio_hidden_size:{}\nvideo_hidden_size:{}'.format(UNI_T_N_HIDDEN, UNI_A_N_HIDDEN, UNI_V_N_HIDDEN))

    print('-' * 32 + 'multimodal context network' + '-' * 32)
    print('language_in_dim:{}\naudio_in_dim:{}\nvideo_in_dim:{}'.format(MUL_T_IN_DIM, MUL_A_IN_DIM, MUL_V_IN_DIM))
    print('language_dropout:{}\naudio_dropout:{}\nvideo_dropout:{}'.format(MUL_T_DROPOUT, MUL_A_DROPOUT, MUL_V_DROPOUT))
    print('dropout:{}'.format(MUL_DROPOUT))
    print('-' * 16 + 'transformer encoder' + '-' * 16)
    print('src_feature_size:{}\nmax_seq_len:{}'.format(SRC_N_FEATURE, MAX_SEQ_LEN))
    print('d_model:{}\nnhead:{}\nn_layer:{}\nd_ff:{}\nd_k:{}\nd_v:{}'.format(D_MODEL, NHEAD, N_LAYER, D_FF, D_K, D_V))

    print('-' * 32 + 'memory fusion network' + '-' * 32)
    print('language_in_dim:{}\naudio_in_dim:{}\nvideo_in_dim:{}'.format(MFN_T_IN_DIM, MFN_A_IN_DIM, MFN_V_IN_DIM))
    print('language_hidden_size:{}\naudio_hidden_size:{}\nvideo_hidden_size:{}'.format(MFN_T_N_HIDDEN, MFN_A_N_HIDDEN, MFN_V_N_HIDDEN))
    print('total_hidden_size:{}\nwindow_dim:{}\nmem_dim:{}'.format(MFN_N_HIDDEN, MFN_WINDOW_DIM, MFN_MEM_DIM))
    print('attn_in_dim:{}'.format(MFN_ATTN_IN_DIM))
    print('nn1_dim:{}\nnn1_dropout:{}\nnn2_dim:{}\nnn2_dropout:{}'.format(MFN_NN1_DIM, MFN_NN1_DROPOUT, MFN_NN2_DIM, MFN_NN2_DROPOUT))
    print('gamma_in_dim:{}'.format(MFN_GAMMA_IN_DIM))
    print('gamma1_dim:{}\ngamma1_dropout:{}\ngamma2_dim:{}\ngamma2_dropout:{}'.format(MFN_GAMMA1_DIM, MFN_GAMMA1_DROPOUT, MFN_GAMMA2_DIM, MFN_GAMMA2_DROPOUT))
    print('output_in_dim:{}\noutput_hidden_dim:{}\noutput_dropout:{}\noutput_dim:{}'.format(MFN_OUTPUT_IN_DIM, MFN_OUTPUT_HIDDEN_DIM, MFN_OUTPUT_DROPOUT, MFN_OUTPUT_DIM))


def set_dataloader():
    data_folds = load_pickle(DATA_FOLDS_FILE)

    train = data_folds['train']
    train_set = HumorDataset(train)
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    dev = data_folds['dev']
    dev_set = HumorDataset(dev)
    dev_dataloader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    test = data_folds['test']
    test_set = HumorDataset(test)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    return train_dataloader, dev_dataloader, test_dataloader


def train_epoch(model, train_dataloader, optimizer, criterion):
    epoch_loss = 0.0
    n_batch = 0

    model.train()
    for batch in train_dataloader:
        x_c, x_p, y = map(lambda x: x.to(DEVICE), batch)

        optimizer.zero_grad()
        preds = model(x_c, x_p, y).squeeze(0)
        loss = criterion(preds, y.float())
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        n_batch += 1

    if n_batch == 0:
        n_batch += 1
    return epoch_loss / n_batch


def train(model, train_dataloader, optimizer, criterion):
    for epoch in range(N_EPOCH):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion)
        print('epoch:{}  train loss:{}'.format(epoch, round(train_loss, 4)))


def test_epoch(model, test_dataloader):
    n_batch = 0
    preds = None
    y_test = None

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            x_p, x_c, y = map(lambda x: x.to(DEVICE), batch)

            pred = model(x_p, x_c, y).squeeze(0)

            y_test = y.squeeze(1).cpu().numpy()
            preds = pred.squeeze(1).cpu().data.numpy()

            n_batch += 1

    return preds, y_test


def test_score(model, test_dataloader):
    preds, y_test = test_epoch(model, test_dataloader)
    preds = (preds >= 0)

    f1 = f1_score(np.round(preds), np.round(y_test), average='weighted')
    accuracy = accuracy_score(y_test, preds)

    print('accuracy:{}\nf1:{}'.format(round(accuracy, 4), round(f1, 4)))


def main():
    train_dataloader, dev_dataloader, test_dataloader = set_dataloader()

    model = C_MFN().to(DEVICE)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)

    show_config(optimizer='optim.Adam(betas=(0.9, 0.98), eps=1e-9)', criterion='nn.BCEWithLogitsLoss')

    print('-' * 32 + 'train' + '-' * 32)
    train_start_time = time.time()
    train(model, train_dataloader, optimizer, criterion)
    train_finish_time = time.time()
    print('train cost:{}s'.format(train_finish_time - train_start_time))

    print('-' * 32 + 'test' + '-' * 32)
    test_start_time = time.time()
    test_score(model, test_dataloader)
    test_finish_time = time.time()
    print('test cost:{}s'.format(test_finish_time - test_start_time))


if __name__ == '__main__':
    main()
