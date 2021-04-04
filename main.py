from driver import ex
import os
import torch

experiment_configs = [
    # idx 0:(c, p), (t, a, v)
    {
        'use_context': True, 'use_t_context': True, 'use_a_context': True, 'use_v_context': True,
        'use_punchline': True, 'use_t_punchline': True, 'use_a_punchline': True, 'use_v_punchline': True
    },
    # idx 1:(p), (t, a, v)
    {
        'use_context': False, 'use_t_context': False, 'use_a_context': False, 'use_v_context': False,
        'use_punchline': True, 'use_t_punchline': True, 'use_a_punchline': True, 'use_v_punchline': True
    },
    # idx 2:(c), (t, a, v)
    {
        'use_context': True, 'use_t_context': True, 'use_a_context': True, 'use_v_context': True,
        'use_punchline': False, 'use_t_punchline': False, 'use_a_punchline': False, 'use_v_punchline': False
    },

    # idx 3:(c, p), (t, v)
    {
        'use_context': True, 'use_t_context': True, 'use_a_context': False, 'use_v_context': True,
        'use_punchline': True, 'use_t_punchline': True, 'use_a_punchline': False, 'use_v_punchline': True
    },
    # idx 4:(p), (t, v)
    {
        'use_context': False, 'use_t_context': False, 'use_a_context': False, 'use_v_context': False,
        'use_punchline': True, 'use_t_punchline': True, 'use_a_punchline': False, 'use_v_punchline': True
    },
    # idx 5:(c), (t, v)
    {
        'use_context': True, 'use_t_context': True, 'use_a_context': False, 'use_v_context': True,
        'use_punchline': False, 'use_t_punchline': False, 'use_a_punchline': False, 'use_v_punchline': False
    },

    # idx 6:(c, p), (t, a)
    {
        'use_context': True, 'use_t_context': True, 'use_a_context': True, 'use_v_context': False,
        'use_punchline': True, 'use_t_punchline': True, 'use_a_punchline': True, 'use_v_punchline': False
    },
    # idx 7:(p), (t, a)
    {
        'use_context': False, 'use_t_context': False, 'use_a_context': False, 'use_v_context': False,
        'use_punchline': True, 'use_t_punchline': True, 'use_a_punchline': True, 'use_v_punchline': False
    },
    # idx 8:(c), (t, a)
    {
        'use_context': True, 'use_t_context': True, 'use_a_context': True, 'use_v_context': False,
        'use_punchline': False, 'use_t_punchline': False, 'use_a_punchline': False, 'use_v_punchline': False
    },

    # idx 9:(c, p), (a, v)
    {
        'use_context': True, 'use_t_context': False, 'use_a_context': True, 'use_v_context': True,
        'use_punchline': True, 'use_t_punchline': False, 'use_a_punchline': True, 'use_v_punchline': True
    },
    # idx 10:(p), (a, v)
    {
        'use_context': False, 'use_t_context': False, 'use_a_context': False, 'use_v_context': False,
        'use_punchline': True, 'use_t_punchline': False, 'use_a_punchline': True, 'use_v_punchline': True
    },
    # idx 11:(c), (a, v)
    {
        'use_context': True, 'use_t_context': False, 'use_a_context': True, 'use_v_context': True,
        'use_punchline': False, 'use_t_punchline': False, 'use_a_punchline': False, 'use_v_punchline': False
    },

    # idx 12:(c, p), (t)
    {
        'use_context': True, 'use_t_context': True, 'use_a_context': False, 'use_v_context': False,
        'use_punchline': True, 'use_t_punchline': True, 'use_a_punchline': False, 'use_v_punchline': False
    },
    # idx 13:(p), (t)
    {
        'use_context': False, 'use_t_context': False, 'use_a_context': False, 'use_v_context': False,
        'use_punchline': True, 'use_t_punchline': True, 'use_a_punchline': False, 'use_v_punchline': False
    },
    # idx 14:(c), (t)
    {
        'use_context': True, 'use_t_context': True, 'use_a_context': False, 'use_v_context': False,
        'use_punchline': False, 'use_t_punchline': False, 'use_a_punchline': False, 'use_v_punchline': False
    }
]
n_experiment_config = len(experiment_configs)
experiment_names = [
    '(c, p),(t, a, v)',  # idx 0
    '(p),(t, a, v)',  # idx 1
    '(c),(t, a, v)',  # idx 2

    '(c, p),(t, v)',  # idx 3
    '(p),(t, v)',  # idx 4
    '(c),(t, v)',  # idx 5

    '(c, p),(t, a)',  # idx 6
    '(p),(t, a)',  # idx 7
    '(c),(t, a)',  # idx 8

    '(c, p),(a, v)',  # idx 9
    '(p),(a, v)',  # idx 10
    '(c),(a, v)',  # idx 11

    '(c, p),(t)',  # idx 12
    '(p), (t)',  # idx 13
    '(c),(t)'  # idx 14
]
n_experiment = 12


def run_experiment(idx):
    for experiment in range(n_experiment):
        upd_cfg_dict = {
            **experiment_configs[idx],
            'train': True,
            'experiment_name': experiment_names[idx],
            'experiment_idx': idx,
            'experiment': experiment
        }
        ex.run(config_updates=upd_cfg_dict)


def get_best_model_ckpt(idx):
    cur_experiment_path = os.path.join('.', 'Experiment', str(idx) + '-' + experiment_names[idx])

    file_list = os.listdir(cur_experiment_path)
    best_ckpt_file = None
    min_valid_loss = float('inf')
    for file in file_list:
        if file.split('.')[-1] != 'ckpt':
            continue
        file_path = os.path.join(cur_experiment_path, file)
        ckpt = torch.load(file_path)
        if ckpt['valid_loss'] <= min_valid_loss:
            best_ckpt_file = file_path
            min_valid_loss = ckpt['valid_loss']
    return best_ckpt_file


def run_test(idx, test_file):
    upd_cfg_dict = {
        **experiment_configs[idx],
        'train': False,
        'test_file': test_file,
        'experiment_name': experiment_names[idx],
        'experiment_idx': idx
    }
    ex.run(config_updates=upd_cfg_dict)


def delete_ckpt(idx, best_ckpt_file):
    cur_experiment_path = os.path.join('.', 'Experiment', str(idx) + '-' + experiment_names[idx])
    file_list = os.listdir(cur_experiment_path)
    for file in file_list:
        if file.split('.')[-1] != 'ckpt':
            continue
        file_path = os.path.join(cur_experiment_path, file)
        if file_path != best_ckpt_file:
            os.remove(file_path)


if __name__ == '__main__':
    for experiment_idx in range(n_experiment_config):
        run_experiment(experiment_idx)
        best_model_ckpt_file = get_best_model_ckpt(experiment_idx)
        run_test(experiment_idx, best_model_ckpt_file)
        delete_ckpt(experiment_idx, best_model_ckpt_file)
