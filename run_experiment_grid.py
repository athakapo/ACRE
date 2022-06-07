from utils.run_utils import ExperimentGrid
#from algos.acre.acre import acre
from algos.acre_rnd.acre_rnd import acre_rnd

if __name__ == '__main__':
    import argparse

    # Disable Tensorboard
    logger_tb_args = dict()
    logger_tb_args['enable'] = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()

    eg = ExperimentGrid(name='acre_rnd-pyt-bench_MountainCarContinuous-v0')
    eg.add('suite', 'gym')
    eg.add('env_name', 'MountainCarContinuous-v0', '', True)
    eg.add('ac_kwargs', dict(hidden_sizes=[256] * 2))
    eg.add('seed', [0, 1, 2, 3, 4])
    eg.add('steps_per_epoch', 4000)
    eg.add('gamma', 0.99)
    eg.add('epochs', 125)
    eg.add('beta', 100)
    eg.add('RNDoutput_size', 4)
    eg.add('rnd_num_nodes', 256)
    eg.add('estimate_rnd_every', 1)
    eg.add('logger_tb_args', logger_tb_args)
    eg.run(acre_rnd, num_cpu=args.cpu)
