from driver import ex

experiment_configs = [
    # idx 0:(c, p), (t, a, v)
    {
        'use_context': True,
        'use_t_context': True,
        'use_a_context': True,
        'use_v_context': True,
        'use_punchline': True,
        'use_t_punchline': True,
        'use_a_punchline': True,
        'use_v_punchline': True
    },
    # idx 1:(c, p), (t, a)
    {
        'use_context': True,
        'use_t_context': True,
        'use_a_context': True,
        'use_v_context': False,
        'use_punchline': True,
        'use_t_punchline': True,
        'use_a_punchline': True,
        'use_v_punchline': False
    },
    # idx 2:(c, p), (t, v)
    {
        'use_context': True,
        'use_t_context': True,
        'use_a_context': False,
        'use_v_context': True,
        'use_punchline': True,
        'use_t_punchline': True,
        'use_a_punchline': False,
        'use_v_punchline': True
    },
    # idx 3:(c, p), (a, v)
    {
        'use_context': True,
        'use_t_context': False,
        'use_a_context': True,
        'use_v_context': True,
        'use_punchline': True,
        'use_t_punchline': False,
        'use_a_punchline': True,
        'use_v_punchline': True
    },
    # idx 4:(c, p), (t)
    {
        'use_context': True,
        'use_t_context': True,
        'use_a_context': False,
        'use_v_context': False,
        'use_punchline': True,
        'use_t_punchline': True,
        'use_a_punchline': False,
        'use_v_punchline': False
    },
    # idx 5:(p), (t, a, v)
    {
        'use_context': False,
        'use_t_context': False,
        'use_a_context': False,
        'use_v_context': False,
        'use_punchline': True,
        'use_t_punchline': True,
        'use_a_punchline': True,
        'use_v_punchline': True
    },
    # idx 6:(p), (t, a)
    {
        'use_context': False,
        'use_t_context': False,
        'use_a_context': False,
        'use_v_context': False,
        'use_punchline': True,
        'use_t_punchline': True,
        'use_a_punchline': True,
        'use_v_punchline': False
    },
    # idx 7:(p), (t, v)
    {
        'use_context': False,
        'use_t_context': False,
        'use_a_context': False,
        'use_v_context': False,
        'use_punchline': True,
        'use_t_punchline': True,
        'use_a_punchline': False,
        'use_v_punchline': True
    },
    # idx 8:(p), (a, v)
    {
        'use_context': False,
        'use_t_context': False,
        'use_a_context': False,
        'use_v_context': False,
        'use_punchline': True,
        'use_t_punchline': False,
        'use_a_punchline': True,
        'use_v_punchline': True
    },
    # idx 9:(p), (t)
    {
        'use_context': False,
        'use_t_context': False,
        'use_a_context': False,
        'use_v_context': False,
        'use_punchline': True,
        'use_t_punchline': True,
        'use_a_punchline': False,
        'use_v_punchline': False
    },
    # idx 10:(c), (t, a, v)
    {
        'use_context': True,
        'use_t_context': True,
        'use_a_context': True,
        'use_v_context': True,
        'use_punchline': False,
        'use_t_punchline': False,
        'use_a_punchline': False,
        'use_v_punchline': False
    },
    # idx 11:(c), (t, a)
    {
        'use_context': True,
        'use_t_context': True,
        'use_a_context': True,
        'use_v_context': False,
        'use_punchline': False,
        'use_t_punchline': False,
        'use_a_punchline': False,
        'use_v_punchline': False
    },
    # idx 12:(c), (t, v)
    {
        'use_context': True,
        'use_t_context': True,
        'use_a_context': False,
        'use_v_context': True,
        'use_punchline': False,
        'use_t_punchline': False,
        'use_a_punchline': False,
        'use_v_punchline': False
    },
    # idx 13:(c), (a, v)
    {
        'use_context': True,
        'use_t_context': False,
        'use_a_context': True,
        'use_v_context': True,
        'use_punchline': False,
        'use_t_punchline': False,
        'use_a_punchline': False,
        'use_v_punchline': False
    },
    # idx 14:(c), (t)
    {
        'use_context': True,
        'use_t_context': True,
        'use_a_context': False,
        'use_v_context': False,
        'use_punchline': False,
        'use_t_punchline': False,
        'use_a_punchline': False,
        'use_v_punchline': False
    }
]
n_experiment_config = len(experiment_configs)
experiment_names = [
    '(c, p),(t, a, v)',
    '(c, p),(t, a)',
    '(c, p),(t, v)',
    '(c, p),(a, v)',
    '(c, p),(t)',
    '(p),(t, a, v)',
    '(p),(t, a)',
    '(p),(t, v)',
    '(p),(a, v)',
    '(c),(t, a, v)',
    '(c),(t, a)',
    '(c),(t, v)',
    '(c),(a, v)',
    '(c),(t)'
]
n_experiment = 20


def run_experiment(idx):
    for experiment in range(n_experiment):
        upd_cfg_dict = {
            **experiment_configs[idx],
            'experiment_name': experiment_names[idx],
            'experiment_idx': idx,
            'experiment': experiment
        }
        ex.run(config_updates=upd_cfg_dict)


if __name__ == '__main__':
    for experiment_idx in range(n_experiment_config):
        run_experiment(experiment_idx)
