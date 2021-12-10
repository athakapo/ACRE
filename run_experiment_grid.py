from utils.run_utils import ExperimentGrid
from algos.acre.acre import acre

if __name__ == '__main__':
    import argparse

    # Disable Tensorboard
    logger_tb_args = dict()
    logger_tb_args['enable'] = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()

    eg = ExperimentGrid(name='acre-pyt-bench')
    eg.add('suite', 'gym')
    eg.add('env_name', 'MountainCarContinuous-v0', '', True)
    eg.add('ac_kwargs', dict(hidden_sizes=[256] * 2))
    eg.add('seed', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    eg.add('steps_per_epoch', 4000)
    eg.add('gamma', 0.99)
    eg.add('epochs', 160)
    eg.add('beta', [0.007, 0.0007, 0.00007])
    eg.add('n_components', 7)
    eg.add('estimate_gmm_every', 1)
    eg.add('logger_tb_args', logger_tb_args)
    eg.add('q_powered_gmm', False)
    eg.add('plot_gmm', False)
    eg.run(acre, num_cpu=args.cpu)
