import torch
import gym
from ppo_gmm import core
from ppo_gmm import ppo_gmm
import numpy as np

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Swimmer-v2')
    parser.add_argument('--reward_type', type=str, default=None)  # None
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--w_i', type=float, default=1.0)
    parser.add_argument('--n_components', type=int, default=7)
    parser.add_argument('--exp_name', type=str, default='ppo_gmm')
    parser.add_argument('--cpu', type=int, default=4)  # 4
    parser.add_argument('--steps', type=int, default=4000)  # 4000
    parser.add_argument('--polyak', type=float, default=0.95)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--tensorboard', type=bool, default=True)
    parser.add_argument('--aggregate_stats', type=int, default=100)

    args = parser.parse_args()

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    logger_tb_args = dict()
    logger_tb_args['enable'] = args.tensorboard
    if args.tensorboard:
        if args.reward_type is not None:
            instance_details = f"{args.env}-RT{args.reward_type}-{args.exp_name}-[{args.l}_{args.hid}]-wi_{args.w_i}"
        else:
            instance_details = f"{args.env}-{args.exp_name}-[{args.l}_{args.hid}]-wi_{args.w_i}"
        logger_tb_args['instance_details'] = instance_details
        logger_tb_args['aggregate_stats'] = args.aggregate_stats

    torch.set_num_threads(torch.get_num_threads())

    ppo_gmm(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), reward_type=args.reward_type,
        gamma=args.gamma, clip_ratio=0.4, pi_lr=args.learning_rate, vf_lr=args.learning_rate,
        train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000, w_i=args.w_i,
        target_kl=0.01, seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, n_components=args.n_components,
        logger_kwargs=logger_kwargs, logger_tb_args=logger_tb_args)
