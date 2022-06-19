import torch
import gym
from acre_rnd import core
from acre_rnd import acre_rnd

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--reward_type', type=str, default=None)  # None
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--beta', type=float, default=100.0)
    parser.add_argument('--RNDoutput_size', type=int, default=4)
    parser.add_argument('--rnd_num_nodes', type=int, default=256)
    parser.add_argument('--estimate_rnd_every', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default='acre_rnd')
    parser.add_argument('--tensorboard', type=bool, default=True)
    parser.add_argument('--aggregate_stats', type=int, default=100)
    parser.add_argument('--save_all_states', type=bool, default=True)
    args = parser.parse_args()

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    logger_tb_args = dict()
    logger_tb_args['enable'] = args.tensorboard
    if args.tensorboard or args.save_all_states:
        if args.reward_type is not None:
            instance_details = f"{args.env}-RT{args.reward_type}-{args.exp_name}-[{args.l}_{args.hid}]-beta_{args.beta}"
        else:
            instance_details = f"{args.env}-{args.exp_name}-[{args.l}_{args.hid}]-beta_{args.beta}"
        logger_tb_args['instance_details'] = instance_details
        logger_tb_args['aggregate_stats'] = args.aggregate_stats

    torch.set_num_threads(torch.get_num_threads())

    acre_rnd(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
             ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), reward_type=args.reward_type,
             gamma=args.gamma, seed=args.seed, epochs=args.epochs, beta=args.beta,
             estimate_rnd_every=args.estimate_rnd_every, RNDoutput_size=args.RNDoutput_size,
             rnd_num_nodes=args.rnd_num_nodes,  save_all_states=args.save_all_states,
             logger_kwargs=logger_kwargs, logger_tb_args=logger_tb_args)