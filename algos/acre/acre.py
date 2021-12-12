from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from algos.acre import core
from utils.gmm import GMM
from utils.logx import EpochLogger, TensorBoardLogger


class InfoBuffer:

    def __init__(self, episodes=150, max_ep_len=1000, gamma=0.99, q_powered_gmm=False):
        self.obs_buf = np.frompyfunc(list, 0, 1)(np.empty((episodes), dtype=object))
        self.action_buf = np.frompyfunc(list, 0, 1)(np.empty((episodes), dtype=object))
        self.gamma = gamma
        self.max_ep_len = max_ep_len
        self.q_powered_gmm=q_powered_gmm
        self.episode_ptr, self.size, self.num_episodes = 0, 0, episodes

    def store(self, obs, action):
        """
        Append one timestep of agent-environment interaction to the appropriate episode of the buffer.
        """
        self.obs_buf[self.episode_ptr].append(obs)
        self.action_buf[self.episode_ptr].append(action)

    def construct_info_data(self, gmm, logger):
        obs, act, rew, ret = [], [], [], []
        for i in range(self.num_episodes):
            episode_obs = self.obs_buf[i]
            episode_actions = self.action_buf[i]

            # Concatenate observation with rewards
            if self.q_powered_gmm:
                info_states = np.concatenate((np.array(episode_obs), np.array(episode_actions)), axis=1)
            else:
                info_states = np.array(episode_obs)
            # Utilize GMM estimation to calculate information gain of being at each state
            episode_rewards = gmm.log_prob(info_states)

            # the next line computes rewards-to-go, to be targets for the value function
            episode_returns = core.discount_cumsum(episode_rewards, self.gamma)

            obs.extend(episode_obs)
            act.extend(episode_actions)
            rew.extend(episode_rewards)
            ret.extend(episode_returns)

        data = dict(obs=np.array(obs), act=np.array(act), ret=np.array(ret))

        # Record things
        for i in range(len(rew)):
            logger.store(LogProbReward=rew[i], LogProbReturn=ret[i])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def proceed_episode_counters(self):
        # Expected to be called at the end of each episode | Indices increment
        self.episode_ptr = (self.episode_ptr + 1) % self.num_episodes
        self.size = min(self.size + 1, self.num_episodes)

        # Important Step
        # Empty the next list to avoid stacking over past episodes' data
        self.obs_buf[self.episode_ptr] = []
        self.action_buf[self.episode_ptr] = []


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def get_last_aug_states(self, num_samples):
        if self.ptr - num_samples < 0:
            s_p = self.max_size - (num_samples - self.ptr)
            element1 = np.concatenate((self.obs_buf[s_p:], self.act_buf[s_p:]), axis=1)
            element2 = np.concatenate((self.obs_buf[:self.ptr], self.act_buf[:self.ptr]), axis=1)
            return np.concatenate((element1, element2))
        else:
            return np.concatenate((self.obs_buf[self.ptr - num_samples:self.ptr],
                                   self.act_buf[self.ptr - num_samples:self.ptr]), axis=1)

    def get_last_states(self, num_samples):
        if self.ptr - num_samples < 0:
            s_p = self.max_size - (num_samples - self.ptr)
            element1 = self.obs_buf[s_p:]
            element2 = self.obs_buf[:self.ptr]
            return np.concatenate((element1, element2))
        else:
            return self.obs_buf[self.ptr - num_samples:self.ptr]

    def get_random_aug_states(self, num_samples):
        idxs = np.random.randint(0, self.size, size=num_samples)
        return np.concatenate((self.obs_buf[idxs], self.act_buf[idxs]), axis=1)

    def get_random_states(self, num_samples):
        idxs = np.random.randint(0, self.size, size=num_samples)
        return self.obs_buf[idxs]




def acre(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), reward_type=None, seed=0,
            steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, n_components=7,
            polyak=0.995, lr=1e-3, beta=0.1, batch_size=100, start_steps=10000, mult_gmm_samples=3,
            update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, plot_gmm=False,
            train_v_iters=80, estimate_gmm_every=5, q_powered_gmm= False, logger_kwargs=dict(),
            logger_tb_args=dict(), save_freq=10):
    """
    Actor-Critic with Reward-Preserving Exploration (ACRE)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        n_components (int): The number of mixture components used for the
            gmm fitting.

        mult_gmm_samples (int): Number of last episodes the transitions of
            which are going to be used for the next GMM update.

        train_v_iters (int): Number of gradient descent steps to take on
            value function per novelty estimation.

        beta (float): Temperature parameter that weights the importance of
            novelty over the reward returns.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Set up Tensorboard logger
    if logger_tb_args['enable']:
        logger_tb = TensorBoardLogger(logger_tb_args)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()

    # Handle goal RL setup
    if type(env.reset()) == dict:
        env.env.reward_type = reward_type
        keys = ['achieved_goal', 'desired_goal', 'observation']
        try:  # for modern Gym (>=0.15.4)
            from gym.wrappers import FilterObservation, FlattenObservation
            env = FlattenObservation(FilterObservation(env, keys))
            test_env = FlattenObservation(FilterObservation(test_env, keys))
        except ImportError:  # for older gym (<=0.15.3)
            from gym.wrappers import FlattenDictWrapper  # pytype:disable=import-error
            env = FlattenDictWrapper(env, keys)
            test_env = FlattenDictWrapper(test_env, keys)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, q_powered_gmm=q_powered_gmm, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())


    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    info_buffer = InfoBuffer(episodes=estimate_gmm_every, max_ep_len=max_ep_len, gamma=gamma,
                             q_powered_gmm=q_powered_gmm)

    # Kernel Density Estimator
    if q_powered_gmm:
        gmm_params = ac.q_gmm.parameters()
        gmm = GMM([env.observation_space, env.action_space], n_components, plot_gmm)
        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2, ac.q_gmm])
        logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t q_gmm: %d\n' % var_counts)
    else:
        gmm_params = ac.v_gmm.parameters()
        gmm = GMM([env.observation_space], n_components, plot_gmm)
        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2, ac.v_gmm])
        logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t v_gmm: %d\n' % var_counts)

    # initialize episode logger
    # ep_logger = EpisodesLogs(mult_gmm_samples)

    # Set up function for computing information V-loss
    def compute_loss_gmm_ret(data):
        if q_powered_gmm:
            obs, a, ret = data['obs'], data['act'], data['ret']
            loss = ((ac.q_gmm(obs, a) - ret) ** 2).mean()
        else:
            obs, ret = data['obs'], data['ret']
            loss = ((ac.v_gmm(obs) - ret) ** 2).mean()
        return loss

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            if q_powered_gmm:
                gmm_next = ac.q_gmm(o2, a2) #TODO check ac_targ.q_gmm(o2, a2)
            else:
                gmm_next = ac.v_gmm(o2)

            backup = r + gamma * (1 - d) * (q_pi_targ - beta * gmm_next)



        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)

        if q_powered_gmm:
            gmm_est = ac.q_gmm(o, pi)
        else:
            gmm_est = ac.v_gmm(o)

        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (beta * gmm_est - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy(),
                       GMMVals=beta * gmm_est.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)
    gmm_optimizer = Adam(gmm_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update_gmm_net(data):
        gmm_l_old = compute_loss_gmm_ret(data).item()

        # Info Value function learning
        for i in range(train_v_iters):
            gmm_optimizer.zero_grad()
            loss_gmm = compute_loss_gmm_ret(data)
            loss_gmm.backward()
            # mpi_avg_grads(ac.q_gmm)  # average grads across MPI processes
            gmm_optimizer.step()

        # Record things
        logger.store(LossGMM=loss_gmm.item(), DeltaLossGMM=(loss_gmm.item() - gmm_l_old))

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False
        for p in gmm_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True
        for p in gmm_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32),
                      deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    bellman_update_allowed = False

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    num_last_obs_gmm_update = mult_gmm_samples * steps_per_epoch
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    episode_timer_start = time.time()
    time_gmm_estimation = 0
    t_ep = 0


    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        #env.render()
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Store observation to info buffer to be used for Q_GMM estimation
        info_buffer.store(o, a)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        timeout = ep_len == max_ep_len
        terminal = d or timeout

        # End of trajectory handling
        if terminal:
            # ep_logger.store(ep_ret=ep_ret, ep_info=ep_info)
            logger.store(EpRet=ep_ret, EpLen=ep_len)

            if logger_tb_args['enable']:
                logger_tb.update_tensorboard(ep_ret, ep_len)

            if t >= update_after and (t_ep+1) % estimate_gmm_every == 0:
                # Handle GNN updates
                gmm_estimation = time.time()
                # Update data over which the GMM estimation is going to take place
                num_data_gmm = min(t, mult_gmm_samples * steps_per_epoch)
                if q_powered_gmm:
                    #states_buf = replay_buffer.get_last_aug_states(num_data_gmm)
                    states_buf = replay_buffer.get_random_aug_states(num_data_gmm)
                else:
                    #states_buf = replay_buffer.get_last_states(num_data_gmm)
                    states_buf = replay_buffer.get_random_states(num_data_gmm)
                gmm.update(states_buf)

                # Auxiliary info about the gmm estimation
                gmm.disp_info(plot3d=plot_gmm)

                # Fetch cumulative GMM data till the end of episode
                data = info_buffer.construct_info_data(gmm, logger)
                # Perform gradient descent for the state visitation estimation
                update_gmm_net(data)
                time_gmm_estimation += (time.time() - gmm_estimation)
                # End of GMM updates

            # Mark the end of episode inside info buffer
            info_buffer.proceed_episode_counters()

            t_ep += 1

            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if gmm.trained and t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:

            # TODO Dynamically adjust beta weight
            # ar_ret, ar_info = ep_logger.get_last_values(mult_gmm_samples)
            # beta =  np.abs(np.max(ar_ret) / np.max(ar_info))

            episode_timer_end = time.time()
            print(f'Episode needed: {episode_timer_end - episode_timer_start} seconds')
            print(f'GMM estimation in episode: {time_gmm_estimation} seconds')
            print(f'Balance Exploration over Exploitation: {beta}')

            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            if gmm.trained:
                # Auxiliary info about the gmm estimation
                #gmm.disp_info(plot3d=plot_gmm)

                # Log info about epoch
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('EpRet', with_min_and_max=True)
                logger.log_tabular('TestEpRet', with_min_and_max=True)
                logger.log_tabular('LogProbReward', with_min_and_max=True)
                logger.log_tabular('LogProbReturn', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                logger.log_tabular('TestEpLen', average_only=True)
                logger.log_tabular('TotalEnvInteracts', t)
                logger.log_tabular('Q1Vals', with_min_and_max=True)
                logger.log_tabular('Q2Vals', with_min_and_max=True)
                logger.log_tabular('GMMVals', with_min_and_max=True)
                logger.log_tabular('LossGMM', average_only=True)
                logger.log_tabular('DeltaLossGMM', average_only=True)
                logger.log_tabular('LogPi', with_min_and_max=True)
                logger.log_tabular('LossPi', average_only=True)
                logger.log_tabular('LossQ', average_only=True)
                logger.log_tabular('Time', time.time() - start_time)
                logger.dump_tabular()

            episode_timer_start = time.time()
            time_gmm_estimation = 0

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--reward_type', type=str, default=None)  # None
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--beta', type=float, default=0.01)
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
    if args.tensorboard:
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
            logger_kwargs=logger_kwargs, logger_tb_args=logger_tb_args)