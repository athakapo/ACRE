from utils.run_utils import ExperimentGrid
#from algos.acre.acre import acre
#from algos.acre_rnd.acre_rnd import acre_rnd
from algos.ppo_gmm.ppo_gmm import ppo_gmm

if __name__ == '__main__':
    import argparse

    # Disable Tensorboard
    logger_tb_args = dict()
    logger_tb_args['enable'] = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()

    eg = ExperimentGrid(name='acre_rnd-pyt-bench2')
    eg.add('suite', 'gym')
    eg.add('env_name', 'MountainCarContinuous-v0', '', True)
    eg.add('ac_kwargs', dict(hidden_sizes=[256] * 2))
    eg.add('seed', [4, 5, 6, 7, 8, 9])
    eg.add('steps_per_epoch', 4000)
    eg.add('gamma', 0.99)
    eg.add('epochs', 125)
    eg.add('w_i', [2.0, 5.0, 10.0, 20.0])
    eg.add('n_components', 7)
    eg.add('logger_tb_args', logger_tb_args)
    eg.run(ppo_gmm, num_cpu=args.cpu)
