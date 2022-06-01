import math
import random

import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import itertools

from algos.ppo_rnd import core
from algos.ppo_rnd.core import RNDModel
from utils.logx import EpochLogger, TensorBoardLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, w_i=0.1):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.adv_intr_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.intr_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.ret_intr_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.valintr_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.w_i = w_i
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, intr, val, valintr, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.intr_buf[self.ptr] = intr
        self.val_buf[self.ptr] = val
        self.valintr_buf[self.ptr] = valintr
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0, last_val_intr=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        # Define the slice
        path_slice = slice(self.path_start_idx, self.ptr)

        # 1. Work for regular reward
        # 1.1 recover rewards - vals
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # 1.2 the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # 1.3 the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        # 2. Work for intrinsic reward
        # 2.1 recover rewards - vals
        intrs = np.append(self.intr_buf[path_slice], last_val_intr)
        valintr = np.append(self.valintr_buf[path_slice], last_val_intr)

        # 2.2 the next two lines implement GAE-Lambda advantage calculation
        deltas_intrs = intrs[:-1] + self.gamma * valintr[1:] - valintr[:-1]
        self.adv_intr_buf[path_slice] = core.discount_cumsum(deltas_intrs, self.gamma * self.lam)

        # 2.3 the next line computes rewards-to-go, to be targets for the intrinsic value estimation
        self.ret_intr_buf[path_slice] = core.discount_cumsum(intrs, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, obs_rms):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # Before combining the advantages let's normalize them first

        # 1. Advantage for the problem at hand
        self.adv_buf = self.normalize_ndarray(self.adv_buf)

        # 2. Advantage for the exploration problem
        self.adv_intr_buf = self.normalize_ndarray(self.adv_intr_buf)

        # final advantage values
        final_adv = self.adv_buf + self.w_i * self.adv_intr_buf

        # normalize observation
        obs_normilized = obs_rms.normalize_me(self.obs_buf)

        data = dict(obs=self.obs_buf, obs_normal=obs_normilized, act=self.act_buf, ret=self.ret_buf,
                    ret_intr=self.ret_intr_buf, adv=final_adv, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    @staticmethod
    def normalize_ndarray(array):
        array_mean, array_std = mpi_statistics_scalar(array)
        return (array- array_mean) / array_std

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), clip_val=5):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.clip_val = clip_val

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize_me(self, x):
        return ((x - self.mean) / np.sqrt(self.var)).clip(-self.clip_val, self.clip_val)


def ppo_rnd(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), reward_type=None,
        reward_eng=False, seed=0, init_steps_obs_std=1000, steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000, w_i=1.0, RNDoutput_size = 4, #TODO check me out
        clip_obs=5, target_kl=0.01, logger_kwargs=dict(), logger_tb_args=dict(), save_freq=10):
    """
    Random Network Distillation (by clipping),

    with early stopping based on approximate KL*

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    # setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Set up Tensorboard logger
    if logger_tb_args['enable']:
        logger_tb = TensorBoardLogger(logger_tb_args)

    # Random seed
    # seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()

    # Handle goal RL setup
    if type(env.reset()) == dict:
        env.env.reward_type = reward_type
        keys = ['achieved_goal', 'desired_goal', 'observation']
        try:  # for modern Gym (>=0.15.4)
            from gym.wrappers import FilterObservation, FlattenObservation
            env = FlattenObservation(FilterObservation(env, keys))
        except ImportError:  # for older gym (<=0.15.3)
            from gym.wrappers import FlattenDictWrapper  # pytype:disable=import-error
            env = FlattenDictWrapper(env, keys)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Create RND model
    explorer = RNDModel(obs_dim[0], RNDoutput_size)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v_core, ac.v_ex, ac.v_in, explorer.predictor])
    logger.log('\nNumber of parameters: \t pi: %d, \t v_core: %d, \t v_ex: %d, \t v_in: %d, \t predictor: %d\n' % var_counts)

    # List of parameters for all V networks (save this for convenience)
    v_params = itertools.chain(ac.v_core.parameters(), ac.v_ex.parameters(), ac.v_in.parameters())

    # Set up experience buffer
    local_steps_per_epoch = steps_per_epoch  # int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, w_i)

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=env.observation_space.shape, clip_val=clip_obs)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret_ex, ret_intr = data['obs'], data['ret'], data['ret_intr']

        v_core = ac.v_core(obs)
        loss_v_ex = ((ac.v_ex(v_core) - ret_ex) ** 2).mean()
        loss_v_in = ((ac.v_in(v_core) - ret_intr) ** 2).mean() # computing curiosity-driven (Random Network Distillation) value loss
        loss_v = loss_v_ex + loss_v_in

        return loss_v

    # Set up function for computing curiosity-driven (Random Network Distillation) value loss
    def compute_loss_v_i(data):
        obs_normal, ret_intr = data['obs_normal'], data['ret_intr']
        return ((explorer.v_i(obs_normal) - ret_intr) ** 2).mean()

    # Set up function for computing curiosity-driven (Random Network Distillation) predictor loss
    def compute_loss_predictor(data):
        obs_normal = data['obs_normal']
        with torch.no_grad():
            targets = explorer.target(obs_normal)

        return ((explorer.predictor(obs_normal) - targets) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(v_params, lr=vf_lr)

    # Set up optimizers for explorer's networks
    predictor_optimizer = Adam(explorer.predictor.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get(obs_rms)

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        loss_predictor_old = compute_loss_predictor(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            #mpi_avg_grads(ac.v)  # average grads across MPI processes #TODO check me
            vf_optimizer.step()

        # Explorer Predictor function learning
        for i in range(train_v_iters):
            predictor_optimizer.zero_grad()
            loss_predictor = compute_loss_predictor(data)
            loss_predictor.backward()
            mpi_avg_grads(explorer.predictor)  # average grads across MPI processes
            predictor_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     RNDloss=loss_predictor_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old),
                     DeltaLossRND=(loss_predictor.item() -loss_predictor_old))


    """ calculate initial observation std """
    store_observation = []
    init_cnt = 0
    while True:
        obs = env.reset()
        while True:  # observation normalize
            obs_rms.update(np.stack(obs))
            action = env.action_space.sample()  # uniform random action.
            obs, _, done, _ = env.step(action)
            init_cnt += 1
            if init_cnt == init_steps_obs_std:
                obs_rms.update(np.stack(obs))
                break
            if done:
                obs_rms.update(np.stack(obs))
                break
        if init_cnt == init_steps_obs_std:
            break
    #TODO check whether we should add init_steps_obs_std to the total number of timestamps

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len, intr_ret = env.reset(), 0, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v_ex, v_in, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            # Calculate intrinsic reward
            r_i = explorer.step(torch.as_tensor(obs_rms.normalize_me(o), dtype=torch.float32))

            # update obs normalize param
            obs_rms.update(o)

            next_o, r, d, _ = env.step(a)

            ep_ret += r
            ep_len += 1
            intr_ret += r_i

            # save and log
            buf.store(o, a, r, r_i, v_ex, v_in, logp)
            logger.store(VExtrinsic=v_ex, VIntrinsic=v_in)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v_ex, v_in, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v_ex, v_in = 0, 0
                buf.finish_path(v_ex, v_in)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    if logger_tb_args['enable']:
                        logger_tb.update_tensorboard_rnd(ep_ret, ep_len, intr_ret)
                # update obs normalize param
                obs_rms.update(next_o)
                o, ep_ret, ep_len, intr_ret = env.reset(), 0, 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VExtrinsic', with_min_and_max=True)
        logger.log_tabular('VIntrinsic', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('RNDloss', average_only=True)
        logger.log_tabular('DeltaLossRND', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0')  # LunarLanderContinuous-v2, Swimmer-v2, MountainCarContinuous-v0
    parser.add_argument('--reward_type', type=str, default=None)  # None
    parser.add_argument('--hid', type=int, default=256)  # 64
    parser.add_argument('--l', type=int, default=2)  # 2
    parser.add_argument('--gamma', type=float, default=0.99)  # 0.99
    parser.add_argument('--w_i', type=float, default=1.0)
    parser.add_argument('--seed', '-s', type=int, default=60)  # 2
    parser.add_argument('--cpu', type=int, default=4)  # 4
    parser.add_argument('--steps', type=int, default=4000)  # 4000
    parser.add_argument('--init_steps_obs_std', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=80)  # 2500
    parser.add_argument('--exp_name', type=str, default='ppo_rnd')
    parser.add_argument('--polyak', type=float, default=0.95)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--tensorboard', type=bool, default=True)
    parser.add_argument('--aggregate_stats', type=int, default=100)
    args = parser.parse_args()

    # mpi_fork(args.cpu)  # run parallel code with mpi

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

    ppo_rnd(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), reward_type=args.reward_type,
        gamma=args.gamma, clip_ratio=0.2, pi_lr=args.learning_rate, vf_lr=args.learning_rate,
        train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000, w_i=args.w_i,
        target_kl=0.01, seed=args.seed, init_steps_obs_std=args.init_steps_obs_std, steps_per_epoch=args.steps,
        epochs=args.epochs, logger_kwargs=logger_kwargs, logger_tb_args=logger_tb_args)
