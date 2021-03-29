import os
import pickle
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sacred import Experiment
from transformer import ScheduledOptim
from models import C_MFN

ex = Experiment('Multimodal Humor Detection')


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


@ex.config
def config():
    # experiment
    experiment_idx = 0
    experiment = 0

    experiment_name = ''
    experiment_path = os.path.join('.', 'Experiment', str(experiment_idx) + '-' + experiment_name)

    best_model_file = os.path.join(experiment_path, 'best_model.pth')
    best_config_file = os.path.join(experiment_path, 'best_config.pkl')
    test_accuracy_file = os.path.join(experiment_path, 'test_accuracy.txt')

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # feature size
    t_n_feature = 300
    a_n_feature = 81
    v_n_feature = 371

    # dataset file
    dataset_path = os.path.join('.', 'Dataset')
    data_folds_file = os.path.join(dataset_path, 'data_folds.pkl')
    emb_list_file = os.path.join(dataset_path, 'word_embedding_list.pkl')
    t_file = os.path.join(dataset_path, 'language_sdk.pkl')
    a_file = os.path.join(dataset_path, 'covarep_features_sdk.pkl')
    v_file = os.path.join(dataset_path, 'openface_features_sdk.pkl')
    label_file = os.path.join(dataset_path, 'humor_label_sdk.pkl')

    # hyper parameter
    n_epoch = 30
    train_batch_size = 512
    dev_batch_size = 2645
    test_batch_size = 3305
    shuffle = True

    learning_rate = random.choice([0.001, 0.002, 0.005, 0.008, 0.01])

    # data
    max_context_len = 5
    max_sentence_len = 20

    use_context = True
    use_t_context = True
    use_a_context = True
    use_v_context = True

    use_punchline = True
    use_t_punchline = True
    use_a_punchline = True
    use_v_punchline = True

    # model
    uni_t_in_dim = t_n_feature
    uni_a_in_dim = a_n_feature
    uni_v_in_dim = v_n_feature
    uni_t_n_hidden = random.choice([32, 64, 88, 128, 156, 256])
    uni_a_n_hidden = random.choice([8, 16, 32, 48, 64, 80])
    uni_v_n_hidden = random.choice([8, 16, 32, 48, 64, 80])

    mul_t_in_dim = max_context_len * uni_t_n_hidden
    mul_a_in_dim = max_context_len * uni_a_n_hidden
    mul_v_in_dim = max_context_len * uni_v_n_hidden
    mul_t_dropout = random.choice([0.0, 0.1, 0.2, 0.5])
    mul_a_dropout = random.choice([0.0, 0.2, 0.5, 0.1])
    mul_v_dropout = random.choice([0.0, 0.2, 0.5, 0.1])
    src_n_feature = uni_t_n_hidden + uni_a_n_hidden + uni_v_n_hidden
    max_seq_len = max_context_len
    d_model = 512
    n_warmup_steps = 4000
    nhead = 8
    n_layer = 6
    d_ff = 2048
    d_k = 64
    d_v = 64
    mul_dropout = random.choice([0.0, 0.2, 0.5, 0.1])

    mfn_t_in_dim = t_n_feature
    mfn_a_in_dim = a_n_feature
    mfn_v_in_dim = v_n_feature
    mfn_t_n_hidden = random.choice([32, 64, 88, 128, 156, 256])
    mfn_a_n_hidden = random.choice([8, 16, 32, 48, 64, 80])
    mfn_v_n_hidden = random.choice([8, 16, 32, 48, 64, 80])
    mfn_n_hidden = mfn_t_n_hidden + mfn_a_n_hidden + mfn_v_n_hidden
    mfn_window_dim = 2
    mfn_mem_dim = random.choice([64, 128, 256, 300, 400])
    mfn_attn_in_dim = mfn_n_hidden * mfn_window_dim
    mfn_nn1_dim = random.choice([32, 64, 128, 256])
    mfn_nn1_dropout = random.choice([0.0, 0.2, 0.5, 0.7])
    mfn_nn2_dim = random.choice([32, 64, 128, 256])
    mfn_nn2_dropout = random.choice([0.0, 0.2, 0.5, 0.7])
    mfn_gamma_in_dim = mfn_attn_in_dim + mfn_mem_dim
    mfn_gamma1_dim = random.choice([32, 64, 128, 256])
    mfn_gamma1_dropout = random.choice([0.0, 0.2, 0.5, 0.7])
    mfn_gamma2_dim = random.choice([32, 64, 128, 256])
    mfn_gamma2_dropout = random.choice([0.0, 0.2, 0.5, 0.7])
    mfn_output_in_dim = mfn_n_hidden + mfn_mem_dim
    mfn_output_hidden_dim = random.choice([32, 64, 128, 256])
    mfn_output_dropout = random.choice([0.0, 0.2, 0.5, 0.7])
    mfn_output_dim = 1


class HumorDataset(Dataset):
    def __init__(self, _config, id_list):
        self.config = _config

        self.id_list = id_list

        self.emb_list_sdk = load_pickle(self.config['emb_list_file'])
        self.t_sdk = load_pickle(self.config['t_file'])
        self.a_sdk = load_pickle(self.config['a_file'])
        self.v_sdk = load_pickle(self.config['v_file'])
        self.label_sdk = load_pickle(self.config['label_file'])

        self.t_dim = self.config['t_n_feature']
        self.a_dim = self.config['a_n_feature']
        self.v_dim = self.config['v_n_feature']
        self.all_dim = self.t_dim + self.a_dim + self.v_dim

        self.max_context_len = self.config['max_context_len']
        self.max_sentence_len = self.config['max_sentence_len']

    def padded_t_feature(self, seq):
        seq = seq[:self.max_sentence_len]
        padded_t = np.concatenate((np.zeros(self.max_sentence_len - len(seq)), seq), axis=0)
        padded_t = np.array([self.emb_list_sdk[int(t_id)] for t_id in padded_t])
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

        t_context = np.array(self.t_sdk[hid]['context_embedding_indexes'])
        a_context = np.array(self.a_sdk[hid]['context_features'])
        v_context = np.array(self.v_sdk[hid]['context_features'])

        t_punchline = np.array(self.t_sdk[hid]['punchline_embedding_indexes'])
        a_punchline = np.array(self.a_sdk[hid]['punchline_features'])
        v_punchline = np.array(self.v_sdk[hid]['punchline_features'])

        x_c = torch.FloatTensor(self.padded_context_feature(t_context, a_context, v_context))
        x_p = torch.FloatTensor(self.padded_punchline_feature(t_punchline, a_punchline, v_punchline))
        y = torch.FloatTensor([self.label_sdk[hid]])

        # x_c: [batch_size, max_context_len, max_sentence_len, n_feature]
        # x_p: [batch_size, max_sentence_len, n_feature]
        return x_c, x_p, y


@ex.capture
def set_experiment(_config):
    if not os.path.isdir(_config['experiment_path']):
        os.makedirs(_config['experiment_path'])


@ex.capture
def set_random_seed(_seed):
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)


@ex.capture
def set_dataloader(_config):
    data_folds = load_pickle(_config['data_folds_file'])

    train = data_folds['train']
    train_set = HumorDataset(_config, train)
    train_dataloader = DataLoader(train_set, batch_size=_config['train_batch_size'], shuffle=_config['shuffle'])

    dev = data_folds['dev']
    dev_set = HumorDataset(_config, dev)
    dev_dataloader = DataLoader(dev_set, batch_size=_config['dev_batch_size'], shuffle=_config['shuffle'])

    test = data_folds['test']
    test_set = HumorDataset(_config, test)
    test_dataloader = DataLoader(test_set, batch_size=_config['test_batch_size'], shuffle=_config['shuffle'])

    return train_dataloader, dev_dataloader, test_dataloader


@ex.capture
def train_epoch(model, train_dataloader, optimizer, criterion, _config):
    epoch_loss = 0.0
    n_batch = 0

    model.train()
    for batch in train_dataloader:
        x_c, x_p, y = map(lambda x: x.to(_config['device']), batch)

        optimizer.zero_grad()
        preds = model(x_c, x_p, y).squeeze(0)
        loss = criterion(preds, y.float())
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step_and_update_lr()

        n_batch += 1

    if n_batch == 0:
        n_batch += 1
    return epoch_loss / n_batch


@ex.capture
def eval_epoch(model, valid_dataloader, criterion, _config):
    epoch_loss = 0.0
    n_batch = 0

    model.eval()
    with torch.no_grad():
        for batch in valid_dataloader:
            x_c, x_p, y = map(lambda x: x.to(_config['device']), batch)

            preds = model(x_c, x_p, y).squeeze(0)
            loss = criterion(preds, y.float())
            epoch_loss += loss.item()

            n_batch += 1

    if n_batch == 0:
        n_batch += 1
    return epoch_loss / n_batch


@ex.capture
def train(model, train_dataloader, valid_dataloader, optimizer, criterion, _config):
    valid_losses = []
    best_epoch = 0
    for epoch in tqdm(range(_config['n_epoch']), desc='{} experiment {}'.format(_config['experiment_idx'], _config['experiment'])):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion)

        valid_loss = eval_epoch(model, valid_dataloader, criterion)
        valid_losses.append(valid_loss)

        if valid_loss <= min(valid_losses):
            best_epoch = epoch
            torch.save(model.state_dict(), _config['best_model_file'])
            with open(_config['best_config_file'], 'wb') as cfg_f:
                pickle.dump(_config, cfg_f)
        elif epoch - best_epoch >= 6:
            print('early stopping break')
            break

        print("\nepoch:{},train_loss:{}, valid_loss:{}".format(epoch, train_loss, valid_loss))


@ex.capture
def test_epoch(model, test_dataloader, _config):
    n_batch = 0
    preds = None
    y_test = None

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            x_p, x_c, y = map(lambda x: x.to(_config['device']), batch)

            pred = model(x_p, x_c, y).squeeze(0)

            y_test = y.squeeze(1).cpu().numpy()
            preds = pred.squeeze(1).cpu().data.numpy()

            n_batch += 1

    return preds, y_test


@ex.capture
def test_score_from_file(test_dataloader, _config):
    model = C_MFN(_config).to(_config['device'])
    model.load_state_dict(torch.load(_config['best_model_file']))

    preds, y_test = test_epoch(model, test_dataloader)
    preds = (preds >= 0)

    acc = accuracy_score(y_test, preds)

    return acc


@ex.automain
def driver(_config):
    set_experiment()
    set_random_seed()

    train_dataloader, dev_dataloader, test_dataloader = set_dataloader()

    model = C_MFN(_config).to(_config['device'])
    optimizer = ScheduledOptim(
        optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=_config['learning_rate'], betas=(0.9, 0.98), eps=1e-9),
        _config['d_model'],
        _config['n_warmup_steps']
    )
    criterion = nn.BCEWithLogitsLoss().to(_config['device'])

    train(model, train_dataloader, dev_dataloader, optimizer, criterion)

    test_accuracy = test_score_from_file(test_dataloader)
    test_accuracy_file = open(_config['test_accuracy_file'], 'a')
    test_accuracy_file.write('The {}th test accuracy of experiment No.{}: {}'.format(_config['experiment'], _config['experiment_idx'], test_accuracy) + '\n')
    test_accuracy_file.close()
