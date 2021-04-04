import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer


class UnimodalContextNet(nn.Module):
    def __init__(self, _config):
        super(UnimodalContextNet, self).__init__()
        self.config = _config

        self.t_lstm = nn.LSTM(input_size=self.config['uni_t_in_dim'], hidden_size=self.config['uni_t_n_hidden'], batch_first=True)
        self.a_lstm = nn.LSTM(input_size=self.config['uni_a_in_dim'], hidden_size=self.config['uni_a_n_hidden'], batch_first=True)
        self.v_lstm = nn.LSTM(input_size=self.config['uni_v_in_dim'], hidden_size=self.config['uni_v_n_hidden'], batch_first=True)

    def forward(self, x_c):
        # x_c: [batch_size, context_len, sentence_len, n_feature]

        x_c.to(self.config['device'])

        old_batch_size, context_len, seq_len, num_feats = x_c.size()
        new_batch_size = old_batch_size * context_len
        x_c = torch.reshape(x_c, (new_batch_size, seq_len, num_feats))  # x_c: [batch_size * context_len, sentence_len, n_feature]

        t_context = x_c[:, :, :self.config['uni_t_in_dim']]  # t_context: [batch_size * context_len, sentence_len, t_n_feature]
        a_context = x_c[:, :, self.config['uni_t_in_dim']:self.config['uni_t_in_dim'] + self.config['uni_a_in_dim']]  # a_context: [batch_size * context_len, sentence_len, a_n_feature]
        v_context = x_c[:, :, self.config['uni_t_in_dim'] + self.config['uni_a_in_dim']:]  # v_context: [batch_size * context_len, sentence_len, v_n_feature]

        if not self.config['use_t_context']:
            t_context = torch.zeros_like(t_context, requires_grad=True)
        if not self.config['use_a_context']:
            a_context = torch.zeros_like(a_context, requires_grad=True)
        if not self.config['use_v_context']:
            v_context = torch.zeros_like(v_context, requires_grad=True)

        t_h0 = torch.zeros(new_batch_size, self.config['uni_t_n_hidden']).unsqueeze(0).to(self.config['device'])  # t_h0: [1, batch_size * context_len, uni_t_n_hidden]
        t_c0 = torch.zeros(new_batch_size, self.config['uni_t_n_hidden']).unsqueeze(0).to(self.config['device'])  # t_c0: [1, batch_size * context_len, uni_t_n_hidden]
        t_o, (t_hn, t_cn) = self.t_lstm(t_context, (t_h0, t_c0))  # t_hn: [1, batch_size * context_len, uni_t_n_hidden]

        a_h0 = torch.zeros(new_batch_size, self.config['uni_a_n_hidden']).unsqueeze(0).to(self.config['device'])  # a_h0: [1, batch_size * context_len, uni_a_n_hidden]
        a_c0 = torch.zeros(new_batch_size, self.config['uni_a_n_hidden']).unsqueeze(0).to(self.config['device'])  # a_c0: [1, batch_size * context_len, uni_a_n_hidden]
        a_o, (a_hn, a_cn) = self.a_lstm(a_context, (a_h0, a_c0))  # a_hn: [1, batch_size * context_len, uni_a_n_hidden]

        v_h0 = torch.zeros(new_batch_size, self.config['uni_v_n_hidden']).unsqueeze(0).to(self.config['device'])  # v_h0: [1, batch_size * context_len, uni_v_n_hidden]
        v_c0 = torch.zeros(new_batch_size, self.config['uni_v_n_hidden']).unsqueeze(0).to(self.config['device'])  # v_c0: [1, batch_size * context_len, uni_v_n_hidden]
        v_o, (v_hn, v_cn) = self.v_lstm(v_context, (v_h0, v_c0))  # v_hn: [1, batch_size * context_len, uni_v_n_hidden]

        t_result = torch.reshape(t_hn, (old_batch_size, context_len, -1))  # t_result: [batch_size, context_len, uni_t_n_hidden]
        a_result = torch.reshape(a_hn, (old_batch_size, context_len, -1))  # a_result: [batch_size, context_len, uni_a_n_hidden]
        v_result = torch.reshape(v_hn, (old_batch_size, context_len, -1))  # v_result: [batch_size, context_len, uni_v_n_hidden]

        return t_result, a_result, v_result


class MultimodalContextNet(nn.Module):
    def __init__(self, _config):
        super(MultimodalContextNet, self).__init__()
        self.config = _config

        self.t_fc = nn.Linear(self.config['mul_t_in_dim'], self.config['mfn_t_n_hidden'])
        self.t_dropout = nn.Dropout(self.config['mul_t_dropout'])

        self.a_fc = nn.Linear(self.config['mul_a_in_dim'], self.config['mfn_a_n_hidden'])
        self.a_dropout = nn.Dropout(self.config['mul_a_dropout'])

        self.v_fc = nn.Linear(self.config['mul_v_in_dim'], self.config['mfn_v_n_hidden'])
        self.v_dropout = nn.Dropout(self.config['mul_v_dropout'])

        self.self_attention = Transformer(
            _config=self.config,
            src_n_feature=self.config['src_n_feature'],
            tgt_n_feature=self.config['mfn_mem_dim'],
            max_seq_len=self.config['max_seq_len'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            n_layer=self.config['n_layer'],
            d_ff=self.config['d_ff'],
            d_k=self.config['d_k'],
            d_v=self.config['d_v']
        )
        self.dropout = nn.Dropout(self.config['mul_dropout'])

    def forward(self, uni_t, uni_a, uni_v):
        # uni_t: [batch_size, context_len, uni_t_n_hidden]
        # uni_a: [batch_size, context_len, uni_a_n_hidden]
        # uni_v: [batch_size, context_len, uni_v_n_hidden]

        uni_t.to(self.config['device'])
        uni_a.to(self.config['device'])
        uni_v.to(self.config['device'])

        reshaped_uni_t = uni_t.reshape((uni_t.shape[0], -1))  # reshaped_uni_t: [batch_size, context_len * uni_t_n_hidden]
        reshaped_uni_a = uni_a.reshape((uni_a.shape[0], -1))  # reshaped_uni_a: [batch_size, context_len * uni_a_n_hidden]
        reshaped_uni_v = uni_v.reshape((uni_v.shape[0], -1))  # reshaped_uni_v: [batch_size, context_len * uni_v_n_hidden]

        mfn_c_t = self.t_dropout(self.t_fc(reshaped_uni_t))  # mfn_ht_input: [batch_size, mfn_t_n_hidden]
        mfn_c_a = self.a_dropout(self.a_fc(reshaped_uni_a))  # mfn_ha_input: [batch_size, mfn_a_n_hidden]
        mfn_c_v = self.v_dropout(self.v_fc(reshaped_uni_v))  # mfn_hv_input: [batch_size, mfn_v_n_hidden]

        concat = torch.cat([uni_t, uni_a, uni_v], dim=2)  # concat: [batch_size, context_len, hidden_size(t + a + v)]

        mfn_mem = self.dropout(self.self_attention(concat)).squeeze(0)  # mfn_mem: [batch_size, mfn_mem_dim]

        return mfn_c_t, mfn_c_a, mfn_c_v, mfn_mem


class MFN(nn.Module):
    def __init__(self, _config):
        super(MFN, self).__init__()
        self.config = _config

        self.t_lstm = nn.LSTMCell(self.config['mfn_t_in_dim'], self.config['mfn_t_n_hidden'])
        self.a_lstm = nn.LSTMCell(self.config['mfn_a_in_dim'], self.config['mfn_a_n_hidden'])
        self.v_lstm = nn.LSTMCell(self.config['mfn_v_in_dim'], self.config['mfn_v_n_hidden'])

        self.attn1_fc1 = nn.Linear(self.config['mfn_attn_in_dim'], self.config['mfn_nn1_dim'])
        self.attn1_fc2 = nn.Linear(self.config['mfn_nn1_dim'], self.config['mfn_attn_in_dim'])
        self.attn1_dropout = nn.Dropout(self.config['mfn_nn1_dropout'])

        self.attn2_fc1 = nn.Linear(self.config['mfn_attn_in_dim'], self.config['mfn_nn2_dim'])
        self.attn2_fc2 = nn.Linear(self.config['mfn_nn2_dim'], self.config['mfn_mem_dim'])
        self.attn2_dropout = nn.Dropout(self.config['mfn_nn2_dropout'])

        self.gamma1_fc1 = nn.Linear(self.config['mfn_gamma_in_dim'], self.config['mfn_gamma1_dim'])
        self.gamma1_fc2 = nn.Linear(self.config['mfn_gamma1_dim'], self.config['mfn_mem_dim'])
        self.gamma1_dropout = nn.Dropout(self.config['mfn_gamma1_dropout'])

        self.gamma2_fc1 = nn.Linear(self.config['mfn_gamma_in_dim'], self.config['mfn_gamma2_dim'])
        self.gamma2_fc2 = nn.Linear(self.config['mfn_gamma2_dim'], self.config['mfn_mem_dim'])
        self.gamma2_dropout = nn.Dropout(self.config['mfn_gamma2_dropout'])

        self.output_fc1 = nn.Linear(self.config['mfn_output_in_dim'], self.config['mfn_output_hidden_dim'])
        self.output_fc2 = nn.Linear(self.config['mfn_output_hidden_dim'], self.config['mfn_output_dim'])
        self.output_dropout = nn.Dropout(self.config['mfn_output_dropout'])

    def forward(self, x_p, c_t, c_a, c_v, mem):
        # x_p: [sentence_len, batch_size, n_feature]
        # c_t: [batch_size, mfn_t_n_hidden]
        # c_a: [batch_size, mfn_a_n_hidden]
        # c_v: [batch_size, mfn_v_n_hidden]
        # mem: [batch_size, mfn_mem_dim]

        x_p.to(self.config['device'])
        c_t.to(self.config['device'])
        c_a.to(self.config['device'])
        c_v.to(self.config['device'])
        mem.to(self.config['device'])

        t_punchline = x_p[:, :, :self.config['mfn_t_in_dim']]  # t_punchline: [sentence_len, batch_size, n_feature]
        a_punchline = x_p[:, :, self.config['mfn_t_in_dim']:self.config['mfn_t_in_dim'] + self.config['mfn_a_in_dim']]  # a_punchline: [sentence_len, batch_size, n_feature]
        v_punchline = x_p[:, :, self.config['mfn_t_in_dim'] + self.config['mfn_a_in_dim']:]  # v_punchline: [sentence_len, batch_size, n_feature]

        if not self.config['use_punchline']:
            t_punchline = torch.zeros_like(t_punchline, requires_grad=True)
            a_punchline = torch.zeros_like(a_punchline, requires_grad=True)
            v_punchline = torch.zeros_like(v_punchline, requires_grad=True)
        if not self.config['use_t_punchline']:
            t_punchline = torch.zeros_like(t_punchline, requires_grad=True)
        if not self.config['use_a_punchline']:
            a_punchline = torch.zeros_like(a_punchline, requires_grad=True)
        if not self.config['use_v_punchline']:
            v_punchline = torch.zeros_like(v_punchline, requires_grad=True)

        t, n, d = x_p.shape  # x_p: [sentence_len, batch_size, n_feature]

        self.t_h = torch.zeros(n, self.config['mfn_t_n_hidden']).to(self.config['device'])  # self.t_h: [n, mfn_t_n_hidden]
        self.a_h = torch.zeros(n, self.config['mfn_a_n_hidden']).to(self.config['device'])  # self.a_h: [n, mfn_a_n_hidden]
        self.v_h = torch.zeros(n, self.config['mfn_v_n_hidden']).to(self.config['device'])  # self.v_h: [n, mfn_v_n_hidden]

        self.t_c = c_t.to(self.config['device'])  # self.t_c: [batch_size, mfn_t_n_hidden]
        self.a_c = c_a.to(self.config['device'])  # self.a_c: [batch_size, mfn_a_n_hidden]
        self.v_c = c_v.to(self.config['device'])  # self.v_c: [batch_size, mfn_v_n_hidden]

        self.mem = mem.to(self.config['device'])  # self.mem: [batch_size, mfn_mem_dim]

        t_h_list, t_c_list = [], []
        a_h_list, a_c_list = [], []
        v_h_list, v_c_list = [], []

        mem_list = []

        for i in range(t):
            # previous time step
            pre_t_c = self.t_c  # pre_t_c: [batch_size, mfn_t_n_hidden]
            pre_a_c = self.a_c  # pre_a_c: [batch_size, mfn_a_n_hidden]
            pre_v_c = self.v_c  # pre_v_c: [batch_size, mfn_v_n_hidden]

            # current time step
            cur_t_h, cur_t_c = self.t_lstm(t_punchline[i], (self.t_h, self.t_c))  # cur_t_h, cur_t_c: [batch_size, mfn_t_n_hidden]
            cur_a_h, cur_a_c = self.a_lstm(a_punchline[i], (self.a_h, self.a_c))  # cur_a_h, cur_a_c: [batch_size, mfn_a_n_hidden]
            cur_v_h, cur_v_c = self.v_lstm(v_punchline[i], (self.v_h, self.v_c))  # cur_v_h, cur_v_c: [batch_size, mfn_v_n_hidden]

            # concatenate
            pre_c = torch.cat([pre_t_c, pre_a_c, pre_v_c], dim=1)  # pre_c: [batch_size, mfn_t_n_hidden + mfn_a_n_hidden + mfn_v_n_hidden]
            cur_c = torch.cat([cur_t_c, cur_a_c, cur_v_c], dim=1)  # cur_c: [batch_size, mfn_t_n_hidden + mfn_a_n_hidden + mfn_v_n_hidden]
            c_star = torch.cat([pre_c, cur_c], dim=1)  # c_star: [batch_size, mfn_attn_in_dim]
            attention = F.softmax(self.attn1_fc2(self.attn1_dropout(F.relu(self.attn1_fc1(c_star)))),
                                  dim=1)  # attention: [batch_size, mfn_attn_in_dim]
            attended = attention * c_star  # attended: [batch_size, mfn_attn_in_dim]
            c_hat = torch.tanh(self.attn2_fc2(self.attn2_dropout(F.relu(self.attn2_fc1(attended)))))  # c_hat: [batch_size, mfn_mem_dim]
            both = torch.cat([attended, self.mem], dim=1)  # both: [batch_size, mfn_gamma_in_dim]
            gamma1 = torch.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))  # gamma1: [batch_size, mfn_gamma_in_dim]
            gamma2 = torch.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))  # gamma2: [batch_size, mfn_gamma_in_dim]
            self.mem = gamma1 * self.mem + gamma2 * c_hat  # self.mem: [batch_size, mfn_mem_dim]
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
        last_h = torch.cat([last_t_h, last_a_h, last_v_h, last_mem], dim=1)  # last_h: [batch_size, mfn_output_in_dim]
        output = self.output_fc2(self.output_dropout(F.relu(self.output_fc1(last_h))))  # output: [batch_size, mfn_output_dim]
        return output


class C_MFN(nn.Module):
    def __init__(self, _config):
        super(C_MFN, self).__init__()
        self.config = _config

        self.unimodal_context = UnimodalContextNet(self.config).to(self.config['device'])
        self.multimodal_context = MultimodalContextNet(self.config).to(self.config['device'])
        self.mfn = MFN(self.config).to(self.config['device'])

    def forward(self, x_c, x_p, y):
        # x_c: [batch_size, context_len, sentence_len, n_feature]
        # x_p: [batch_size, sentence_len, n_feature]

        x_c.to(self.config['device'])
        x_p.to(self.config['device'])
        y.to(self.config['device'])

        if not self.config['use_context']:
            x_c = torch.zeros_like(x_c, requires_grad=True).to(self.config['device'])

        # x_p: [sentence_len, batch_size, n_feature]
        x_p = x_p.permute(1, 0, 2)

        # uni_t: [batch_size, context_len, uni_t_n_hidden]
        # uni_a: [batch_size, context_len, uni_t_a_hidden]
        # uni_v: [batch_size, context_len, uni_t_v_hidden]
        uni_t, uni_a, uni_v = self.unimodal_context(x_c)

        # mfn_c_t: [batch_size, mfn_t_n_hidden]
        # mfn_c_a: [batch_size, mfn_a_n_hidden]
        # mfn_c_v: [batch_size, mfn_v_n_hidden]
        # mfn_mem: [1, batch_size, mfn_mem_dim]
        mfn_c_t, mfn_c_a, mfn_c_v, mfn_mem = self.multimodal_context(uni_t, uni_a, uni_v)

        pred = self.mfn(x_p, mfn_c_t, mfn_c_a, mfn_c_v, mfn_mem)  # pred: [batch_size, mfn_output_dim]
        return pred
