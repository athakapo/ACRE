import torch
import gym
from acre import core
from acre import acre

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--reward_type', type=str, default=None)  # None
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--gmm_samples_mult', type=int, default=10)
    parser.add_argument('--n_components', type=float, default=7)
    parser.add_argument('--estimate_gmm_every', type=int, default=1)
    parser.add_argument('--plot_gmm', type=bool, default=False)
    parser.add_argument('--q_powered_gmm', type=bool, default=False)
    parser.add_argument('--exp_name', type=str, default='acre')
    parser.add_argument('--tensorboard', type=bool, default=True)
    parser.add_argument('--aggregate_stats', type=int, default=100)
    args = parser.parse_args()

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    logger_tb_args = dict()
    logger_tb_args['enable'] = args.tensorboard
    if args.tensorboard or args.save_all_states:
        if args.reward_type is not None:
            instance_details = f"{args.env}-RT{args.reward_type}-{args.exp_name}-[{args.l}_{args.hid}]"
        else:
            instance_details = f"{args.env}-{args.exp_name}-[{args.l}_{args.hid}]"
        logger_tb_args['instance_details'] = instance_details
        logger_tb_args['aggregate_stats'] = args.aggregate_stats

    torch.set_num_threads(torch.get_num_threads())

    acre(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), reward_type=args.reward_type,
         gamma=args.gamma, seed=args.seed, epochs=args.epochs, beta=args.beta, plot_gmm=args.plot_gmm,
         n_components=args.n_components, estimate_gmm_every=args.estimate_gmm_every, q_powered_gmm=args.q_powered_gmm,
         gmm_samples_mult=args.gmm_samples_mult, logger_kwargs=logger_kwargs, logger_tb_args=logger_tb_args)


