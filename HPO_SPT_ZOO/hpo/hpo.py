import warnings
# from typing import Any, Dict, Optional, Type, Union
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type
import numpy as np
import torch as th
from gym import spaces
import gym
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, polyak_update, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer

import math
import pdb 
from hpo.policies import ActorCriticPolicy2
# from hpo.buffer import RolloutBuffer2
from stable_baselines3.common.policies import BaseModel, BasePolicy
import random
import time
import copy

class HPO(OnPolicyAlgorithm):
    """
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy2]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        policy_base: Type[BasePolicy] = ActorCriticPolicy2,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        # classifier: int =0,
        classifier: str="AM",
        aece: str="WAE",
        entropy_hpo: bool =False,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        alpha: float = 0.1,
        rgamma: float = 0.75,
        exploration_rate: float = 0.0,
        reward_noise_std: float = 0.0,
        advantage_flipped_rate: float = 0.0,
        actor_delay: int = 20 ,
        pf_coef: float = 20.0,
        policy_update_scheme: int = 1,
        n_envs: int = 1,
        independent_value_net: bool= False,
        lunarlander_heuristic: bool= False,
    ):

        super(HPO, self).__init__(
            policy,
            env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            # classifier=classifier,#super is for  inherit class ,inherit class does not have classifier
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a multiple of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.classifier = classifier
        self.aece = aece
        self.alpha = alpha
        self.rgamma = rgamma
        self.entropy_hpo = entropy_hpo
        self.spt_clipped_prob = rgamma
        self.exploration_rate = exploration_rate
        self.reward_noise_std = reward_noise_std
        self.seed = seed
        self.advantage_flipped_rate = advantage_flipped_rate
        self.actor_delay = actor_delay
        self.pf_coef = pf_coef
        self.policy_update_scheme = policy_update_scheme
        # self.n_envs = n_envs
        self.independent_value_net = independent_value_net
        self.lunarlander_heuristic = lunarlander_heuristic
        # print("self.n_envs 177",self.n_envs)
        # pdb.set_trace()
        # self.robust_delta_y = self.ROBUSTDELTAY()
        '''policy network S->A'''
        '''value network S->A estimates Q value.''' '''Note:The value network in the original PPO code output the V value'''
        '''Our advantage function comes from Q-V, V value comes from sum a' in (ACTION_SPACE): V+=Q(s,a')*pi(s,a')''' 
        '''Note:The advantage function in the original PPO code comes from GAE(Generalized Advantage Estimate)'''
        if _init_setup_model:
            self._setup_model()
        
     
    def _setup_model(self) -> None:
        print("setup model true")
        super(HPO, self)._setup_model()
        
        # self._setup_lr_schedule()
        # print("self.lr_schedule",self.lr_schedule)
        # self.set_random_seed(self.seed)
        # self.target_policy = self.policy_class(  # pytype:disable=not-instantiable
        #     self.observation_space,
        #     self.action_space,
        #     self.lr_schedule,
        #     use_sde=self.use_sde,
        #     **self.policy_kwargs  # pytype:disable=not-instantiable
        # )
        # self.target_policy = self.target_policy.to(self.device)
        # self.target_policy.load_state_dict(self.policy.state_dict())
        # self.policy = self.policy_class(  # pytype:disable=not-instantiable
        #     self.observation_space,
        #     self.action_space,
        #     self.lr_schedule,
        #     use_sde=self.use_sde,
        #     **self.policy_kwargs  # pytype:disable=not-instantiable
        # )
        # self.policy = self.policy.to(self.device)
        '''Test: to check the influence of updating the policy network and on the value network'''
        '''Note: this option is not compatible with the save and load operation of rlbaseline3-zoo'''
        if self.independent_value_net == True:
            self.value_policy = self.policy_class(  # pytype:disable=not-instantiable
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                optimizer_class = th.optim.Adam,
                use_sde=self.use_sde,
                # **self.policy_kwargs  # pytype:disable=not-instantiable
            )
            self.value_policy = self.value_policy.to(self.device)
        # print("self.value_policy",self.value_policy)
        # print("self.target_policy",self.target_policy)
        # print("self.policy",self.policy)
        # envname = self.env.unwrapped.spec.id
        # buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer
        # print("self.n_envs 208",self.n_envs)
        # self.rollout_buffer = buffer_cls(
        #     self.n_steps,
        #     self.observation_space,
        #     self.action_space,
        #     device=self.device,
        #     gamma=self.gamma,
        #     gae_lambda=self.gae_lambda,
        #     n_envs=self.n_envs,
        # )
        # Initialize schedules for policy/value clipping
        '''clip_range of the original PPO code, not active in our HPO code'''
        '''maybe our alpha(epsilon) should be dynamic'''
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        # rollout_buffer: ReplayBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # print("envname",env)
        n_steps = 0
        print("reward_noise_std",self.reward_noise_std)
        
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        # dones = False
        # self._last_obs = env.reset()
        rollout_time = []
        computeV_time = []
        # print("self.n_envs")
        # print("self.n_envs 273",self.n_envs)
        print("n_rollout_steps",n_rollout_steps)
        
        '''Collect rollout'''
        t_rollout_start = time.time()
        while n_steps < n_rollout_steps :
            # print("n_steps:",n_steps,"n_rollout_steps: ",n_rollout_steps)
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                #actions, values, log_probs = self.policy.forward(obs_tensor) # org
                #print("collect rollout forward")
                actions, _, log_probs = self.policy.forward(obs_tensor)
            
            # print("n_steps: ",n_steps,"n_rollout_steps: ",n_rollout_steps)
            actions = actions.cpu().numpy()
            # print("actions",actions)
            # if self.exploration_rate > random.random():
            #     # print("random choose action")
            #     # actions = np.array([self.action_space.sample()])
            #     actions = np.array([ self.action_space.sample() for psx in range(len(actions)) ])
            #     print("random choose action",actions)
            # else:
                
            # actions = actions.cpu().numpy()
            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            # env.render()
            # if self.reward_error_rate >0.0:
                # flipped_array = np.random.choice( [1,-1], len(rewards), p=[1-self.reward_error_rate, self.reward_error_rate ])  
                # rewards =  rewards *flipped_array
            # if self.reward_noise_std>0:
            #     rnoise = np.random.normal(loc=0.0, scale= self.reward_noise_std, size= len(rewards) )
            #     rewards =  rewards  + rnoise
            # print("infos:",infos)
            # print("state: ",self._last_obs," next state: ",new_obs," rewards: ",rewards," dones: ",dones,"actions",clipped_actions)
            # Compute value
            t_computeV_start = time.time()
            values = th.Tensor(np.zeros_like(actions, dtype=float)).to(self.device)
            batch_actions = np.zeros_like(actions)
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                # new_obs_tensor = obs_as_tensor(new_obs, self.device)
                new_obs_tensor = obs_as_tensor(self._last_obs, self.device)
                '''Compute V_s = E_a~pi [Q_s_a] '''
                for a in range(self.action_space.n):
                    batch_actions = np.full(batch_actions.shape, a)
                    #print(batch_actions)
                    # print("collect_rollouts self.policy.evaluate_actions")
                    if self.independent_value_net == True:
                        _, next_log_probs, _ = self.policy.evaluate_actions(new_obs_tensor, th.from_numpy(batch_actions).to(self.device))
                        next_q_values, _ , _ = self.value_policy.evaluate_actions(new_obs_tensor, th.from_numpy(batch_actions).to(self.device))
                    else: 
                        next_q_values, next_log_probs, _ = self.policy.evaluate_actions(new_obs_tensor, th.from_numpy(batch_actions).to(self.device))
                    # print("print(next_q_values.shape, next_log_probs.shape) ",next_q_values.shape, next_log_probs.shape) 4,1
                    #print(rewards.shape)
                    #print(next_q_values[:,a])
                    exp_q_values = (th.exp(next_log_probs) * next_q_values[:,a]).clone().detach()
                    # exp_q_values = (th.exp(next_log_probs) * next_q_values[:,a])
                    #print(exp_q_values)
                    #print(rewards)
                    # values += th.Tensor(rewards).to(self.device) + exp_q_values
                    values += exp_q_values
                    #print(values)
            #values = th.Tensor(rewards).to(self.device) + (th.exp(next_log_probs) * next_q_values)
            #values = th.Tensor(rewards).to(self.device) + (th.exp(next_log_probs) * next_q_values)
            #print(values.shape)
            t_computeV_end = time.time()
            computeV_time.append(t_computeV_end - t_computeV_start)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            # if dones:
            # # if dones.any():
            #     new_obs = env.reset()
            #     print("env reset")
                # continue
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        values = th.Tensor(np.zeros_like(actions.reshape(-1), dtype=float)).to(self.device)
        batch_actions = np.zeros_like(actions.reshape(-1))
        with th.no_grad():
            '''# Compute value for the last timestep'''
            obs_tensor = obs_as_tensor(new_obs, self.device)
            #_, values, _ = self.policy.forward(obs_tensor)
            for a in range(self.action_space.n):
                batch_actions = np.full(batch_actions.shape, a)
                #print(batch_actions)
                # print("collect_rollouts last timestep self.policy.evaluate_actions")
                if self.independent_value_net == True:
                    _ , next_log_probs, _ = self.policy.evaluate_actions(new_obs_tensor, th.from_numpy(batch_actions).to(self.device))
                    next_q_values, _ , _ = self.value_policy.evaluate_actions(new_obs_tensor, th.from_numpy(batch_actions).to(self.device))
                else: 
                    next_q_values, next_log_probs, _ = self.policy.evaluate_actions(new_obs_tensor, th.from_numpy(batch_actions).to(self.device))
                # next_q_values, next_log_probs, _ = self.policy.evaluate_actions(obs_tensor, th.from_numpy(batch_actions).to(self.device))
                #print(next_q_values.shape, next_log_probs.shape)
                # exp_q_values = (th.exp(next_log_probs) * next_q_values[:,a]).clone().detach()
                exp_q_values = (th.exp(next_log_probs) * next_q_values[:,a]).clone().detach()
                #print(exp_q_values)
                values += exp_q_values
        t_rollout_end = time.time()
        rollout_time.append(t_rollout_end - t_rollout_start)
        logger.record("Time/collect_rollout/Sum", np.sum(rollout_time))
        logger.record("Time/collect_computeV/Sum", np.sum(computeV_time))
        logger.record("Time/collect_rollout/Mean", np.mean(rollout_time))
        logger.record("Time/collect_computeV/Mean", np.mean(computeV_time))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        # for step in reversed(range(rollout_buffer.buffer_size)):
        #     if step == rollout_buffer.buffer_size - 1:
        #         next_non_terminal = 1.0 - self._last_episode_starts 
        #         rollout_buffer.returns[step] = rollout_buffer.rewards[step] + next_non_terminal * rollout_buffer.gamma * values.clone().cpu().numpy().flatten()
        #     else:
        #         next_non_terminal = 1.0 - rollout_buffer.episode_starts[step + 1]
        #         rollout_buffer.returns[step] = rollout_buffer.rewards[step] + next_non_terminal * rollout_buffer.returns[step + 1]
        #         #rollout_buffer.returns[step] = rollout_buffer.rewards[step] + next_non_terminal * rollout_buffer.gamma * rollout_buffer.values[step + 1]
        callback.on_rollout_end()
        # print("collect rollout self.ep_info_buffer",self.ep_info_buffer)
        return True

    def lunarlander_heuristic_f(self,s):
        # Heuristic for:
        # 1. Testing. 
        # 2. Demonstration rollout.
        angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
        if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4: angle_targ = -0.4
        hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

        # PID controller: s[4] angle, s[5] angularSpeed
        angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
        #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

        # PID controller: s[1] vertical coordinate s[3] vertical speed
        hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
        #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

        if s[6] or s[7]: # legs have contact
            angle_todo = 0
            hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
        return a

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # print("train self.ep_info_buffer",self.ep_info_buffer)
        # Update optimizer learning rate
        '''using a target policy to compute the ratio of prob which is not in D(t)'''
        '''D(t) only saves old pi(s,a) which is interacted with env'''
        self.target_policy = copy.deepcopy(self.policy) 
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        #clip_fractions = []
        margins = []
        #positive_a = []
        #negative_a = []
        episode_counter = self._n_updates/self.n_epochs
        positive_p = []
        negative_p = []
        ratio_p = []
        rollout_return = []

        epoch_time = []
        loss_time = []
        computeV_time = []
        t_action_adv_time = []
        #alpha = 0.1
        # if args.algo == 'HPO':
        #     print("args.algo == 'HPO' can use in hpg.py")
        # print("custom_hyperparams: ",self.custom_hyperparams)
        # print("in hpg.py def train: self .classifier",self.classifier)
        print("in hpg.py def train: self .aece",self.aece)
        print("self.rgamma",self.rgamma)
        print("actor_delay",self.actor_delay)
        print("self.policy.optimizer",self.policy.optimizer)
        print("self.rollout_buffer.size()",self.rollout_buffer.size())
        print("self.n_epochs",self.n_epochs)
        # Hard update for target policy -6~8
        #polyak_update(self.policy.parameters(), self.target_policy.parameters(), 1.0)
        print("self.batch_size: ",self.batch_size)
        print("self.advantage_flipped_rate",self.advantage_flipped_rate)
        # train for n_epochs epochs
        t_epoch_start = time.time()
        # flipped_adv = th.randint(0, 2, (self.batch_size,)) #then mul this to advantages
        # tempa = 0.4*th.ones(self.batch_size) # 0.4 it the flipped rate
        # tempb = th.bernoulli(tempa)
        flipped_adv_list = []
        active_example_counter = 0 
        rollbutffer_get_counter = 0
        '''Swapping the rollout_buffer.get and epoch for loop to make the implements in HPO paper, 
        and it is easier to implement the flipping of the sign of advantage'''
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            '''# Do a complete pass on the rollout buffer'''
            rollbutffer_get_counter += 1
            for _ in range(self.action_space.n):
                flipped_adv = ( th.ones(self.batch_size)-2* th.bernoulli( self.advantage_flipped_rate *th.ones(self.batch_size) ) ).to(self.device)  #1 ,-1
                flipped_adv_list.append(flipped_adv)
            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                
                # Hard update for target policy - 9
                #polyak_update(self.policy.parameters(), self.target_policy.parameters(), 1.0)
                # print("len(rollout_data): ",len(rollout_data))
                
                actions = rollout_data.actions
                # print("actions: ",actions)
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                #print(actions)
                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                
                batch_actions = np.zeros(self.batch_size)
                batch_values = np.zeros(self.batch_size)

                # val_q_values = th.zeros(self.batch_size, requires_grad=True).to(self.device)
                # val_q_values = th.zeros(self.batch_size ).to(self.device)
                val_q_values = []
                #tmp_values = th.zeros(self.batch_size, requires_grad=True).to(self.device)
                #val_log_prob = th.zeros(self.batch_size).to(self.device)
                advantages = th.zeros(self.batch_size).to(self.device)
                #advantages = np.zeros(self.batch_size)

                action_advantages = []
                action_probs = []
                #action_log_probs = []
                #action_q_values = []
                positive_adv_prob = np.zeros(self.batch_size)
                negative_adv_prob = np.zeros(self.batch_size)
                # positive_adv_prob = 0
                # negative_adv_prob = 0
                # Q-value
                minMu = np.ones(self.batch_size)
                epsilon = np.zeros(self.batch_size)
                # old_p = th.exp(rollout_data.old_log_prob).detach()
                # for i in range(len(old_p)):
                #     minMu = min(old_p[i],minMu)
                # print("train self.policy.evaluate_actions real actions")
                # print("rollout_data.observations",rollout_data.observations)
                # action_q_values, val_log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                if self.independent_value_net == True:
                    action_q_values, _ , _ = self.value_policy.evaluate_actions(rollout_data.observations, actions)
                    _ , val_log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                else:
                    action_q_values, val_log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                t_computeV_start = time.time()
                minMu = th.zeros_like(val_log_prob).cpu().detach()
                gpu_zero_batchsize =  th.zeros_like(val_log_prob)
                gpu_positive_adv_prob = th.zeros_like(val_log_prob)
                gpu_negative_adv_prob = th.zeros_like(val_log_prob)
                gpu_batch_values  = th.zeros_like(val_log_prob)
                gpu_action_advantages = []
                gpu_action_probs = [] # action first then batch idx
                full_sa_log_prob = []
                target_full_sa_log_prob = []
                '''evaluate_actions return 
                q_values: with all action of the given states
                a_log_prob: only the log prob of the given action
                To get a_log_prob of other actions, we need to run evaluate_actions self.action_space.n times to get all log_prob.
                '''
                for a in range(self.action_space.n):
                    # print("action", a, batch_actions)
                    batch_actions = np.full(batch_actions.shape,a)
                    # print("train self.policy.evaluate_actions batch actions")
                    if self.independent_value_net == True:
                        q_values , _ , _ = self.value_policy.evaluate_actions(rollout_data.observations, actions)
                        _ , a_log_prob , _ = self.policy.evaluate_actions(rollout_data.observations, th.from_numpy(batch_actions).to(self.device))
                    else:
                        q_values , a_log_prob , _ = self.policy.evaluate_actions(rollout_data.observations, th.from_numpy(batch_actions).to(self.device))
                    _, target_a_log_prob, _ = self.target_policy.evaluate_actions(rollout_data.observations, th.from_numpy(batch_actions).to(self.device))
                    #_, a_log_prob, _ = self.target_policy.evaluate_actions(rollout_data.observations, th.from_numpy(batch_actions).to(self.device))
                    #q_values, _, _ = self.value_policy.evaluate_actions(rollout_data.observations, th.from_numpy(batch_actions).to(self.device))
                    gpu_q =  q_values[:,a].flatten()
                    # q = gpu_q.cpu().detach()
                    full_sa_log_prob.append( a_log_prob )
                    target_full_sa_log_prob.append( target_a_log_prob )
                    gpu_p = th.exp(a_log_prob)
                    gpu_action_probs.append(gpu_p)
                    
                    p = gpu_p.cpu().detach()
                    gpu_batch_values += gpu_q*gpu_p
                    # batch_values += (p*q).numpy()
                    minMu = th.minimum(p,minMu)
                    # action_advantages.append(advantages.cpu())
                    gpu_action_advantages.append(gpu_q)
                    # action_advantages.append(q)
                    # action_q_values.append(q)
                    # action_probs.append(p)
                    #action_log_probs.append(a_log_prob)
                    #print("val_values: ", val_values)
                t_computeV_end = time.time()
                computeV_time.append(t_computeV_end - t_computeV_start)   
                # print("batch_values", batch_values)
                # print("action_advantages",action_advantages)
                # action_advantages = action_advantages.cpu()
                t_action_adv_start = time.time()
                # for j in range(self.batch_size):
                #     minMu[j]  = th.min(action_probs,dim=0)
                # if  self.seed == 1234:
                #     pdb.set_trace()
                if self.lunarlander_heuristic:
                    '''The heuristic function in lunar lander output an action, I think it is an optimal policy'''
                    # print("rollout_data.observations",rollout_data.observations)
                    for i in range(self.batch_size):
                        # print("rollout_data.observations[i]",rollout_data.observations[i])
                        optimal_a = self.lunarlander_heuristic_f(rollout_data.observations[i].cpu().numpy())
                        # print("optimal_a",optimal_a)
                        for a in range(self.action_space.n):
                            if optimal_a == a:
                                gpu_action_advantages[a][i] = th.tensor(1)
                            else:
                                gpu_action_advantages[a][i] = th.tensor(-1)
                else:
                    for a in range(self.action_space.n):
                        # print("action_advantages shape",action_advantages[i].shape)
                        #print("Before A", action_advantages[i])
                        # action_advantages[a] -= batch_values
                        gpu_action_advantages[a] -= gpu_batch_values
                        '''To compute the adaptive margin in AE setting, AE and WAE are discarded'''
                        if self.aece == 'AE' or self.aece == 'WAE':
                            gpu_positive_adv_prob = th.where(gpu_action_advantages[a]>0,gpu_action_probs[a] +gpu_positive_adv_prob,gpu_positive_adv_prob)
                            gpu_negative_adv_prob = th.where(gpu_action_advantages[a]<0,gpu_action_probs[a] +gpu_negative_adv_prob,gpu_negative_adv_prob)
                        
                        # action_advantages.append(gpu_action_advantages[a].cpu().clone().detach().numpy())
                        # print("gpu_action_advantages[a]",gpu_action_advantages[a])
                        
                        #print("After A", action_advantages[i])
                        # for j in range(self.batch_size):
                        #     # minMu[j] = min(action_probs[a][j].clone().detach().numpy(),minMu[j])
                        #     # minMu[j] = min(action_probs[a][j],minMu[j])
                        #     if action_advantages[a][j] > 0:
                        #         positive_adv_prob[j] += action_probs[a][j] 
                        #     elif action_advantages[a][j] < 0:
                        #         negative_adv_prob[j] += action_probs[a][j] 
                            # if a == actions[j]:
                            #     #val_log_prob[j] = action_log_probs[i][j]
                            #     #val_q_values[j] = action_advantages[i][j] + batch_values[j]
                            #     val_q_values[j] = action_q_values[j][a] 
                            #     advantages[j] = action_advantages[a][j]
                        #print("val_log_prob: ", val_log_prob)
                # print("gpu_action_advantages",gpu_action_advantages)
                if self.aece == 'AE' or self.aece == 'WAE':
                    positive_adv_prob = gpu_positive_adv_prob.cpu().clone().detach().numpy()
                    negative_adv_prob = gpu_negative_adv_prob.cpu().clone().detach().numpy()
                # print("action_advantages",action_advantages)
                # action_advantages[:] = gpu_action_advantages[:].cpu().clone().detach().numpy()
                '''rearrange the q value and advantage function'''
                for j in range(self.batch_size):
                    # val_q_values[j] = action_q_values[j][ actions[j] ].clone()
                    val_q_values.append(action_q_values[j][ actions[j] ])
                    advantages[j] = gpu_action_advantages[ actions[j] ][j]
                # val_q_values = action_q_values
                t_action_adv_end = time.time()
                t_action_adv_time.append(t_action_adv_end - t_action_adv_start) 
                # HPO: max(0, epsilon - weight_a (ratio - 1))
                #      max(0, margin - y * (x1 - x2))
                policy_losses2 = []
                positive_p.append(positive_adv_prob)
                negative_p.append(negative_adv_prob)
                if self.aece == 'AE' or self.aece == 'WAE':
                    prob_ratio = negative_adv_prob / (positive_adv_prob + 1e-8)
                else:
                    '''to keep the variable prob_ratio, this is similar to pass'''
                    prob_ratio = negative_adv_prob
                #print("Prob Ratio", prob_ratio)
                ratio_p.append(prob_ratio)
                not_0_loss_counter = 0 
                policy_loss = th.tensor(0)
                '''I try to make the code close to the assumptions in papers (uniformly choose an action of s, for all s in D(t))  '''
                '''choose an action: to satisfy the Assumption(In each iteration t, the state-action pairs in D(t) have distinct states.) and 
                make the influence of the incorrect advantage will not be soothed by other actions in the same state'''
                '''episode_counter % self.actor_delay to make the policy update slower than the critic update'''
                '''self.policy_update_scheme == 1 block actor delay must be larger than action_space.n, 
                or some actions will not be updated forever'''
                if episode_counter % self.actor_delay == 0 and self.policy_update_scheme == 0:
                    # if if episode_counter % self.actor_delay < self.action_space.n :
                    # a  =  int((episode_counter ) % self.actor_delay )
                    # for a in range(self.action_space.n) :
                    '''update the action a per (self.actor_delay*self.action_space.n) episodes'''
                    a =  int((episode_counter/self.actor_delay) %  self.action_space.n )
                    # print("episode: ",episode_counter,"a: ",a)
                    if self.classifier == "AM":
                        # x1 = th.exp(val_log_prob - rollout_data.old_log_prob.detach()) # ratio
                        x1 = th.exp(full_sa_log_prob[a] - target_full_sa_log_prob[a].detach()) # ratio
                        # x1 = gpu_action_probs[a]/gpu_action_probs[a].detach() full_sa_log_prob[a]
                        x2 = th.ones_like(x1.clone().detach())
                    elif self.classifier == "AM-log":# log(pi) - log(mu)
                        x1 = full_sa_log_prob[a]
                        x2 = target_full_sa_log_prob[a].detach()
                    elif self.classifier == "AM-root":# root: (pi/mu)^(1/2) - 1
                        x1 = th.sqrt(th.exp(full_sa_log_prob[a] - target_full_sa_log_prob[a].detach())) # ratio
                        x2 = th.ones_like(x1.clone().detach())
                    # elif self.classifier == "AM-sub":
                    #     x1 = th.exp(val_log_prob )
                    #     x2 = th.exp(rollout_data.old_log_prob)
                    # elif self.classifier == "AM-square":
                    #     x1 = th.square(th.exp(val_log_prob - rollout_data.old_log_prob.detach())) # ratio
                    #     x2 = th.ones_like(x1.clone().detach())
                    #advantages = rollout_data.advantages.cpu().detach()
                    #print("advantages",advantages)
                    #abs_adv = np.abs(advantages.cpu())
                    # advantages = advantages.detach()
                    advantages = gpu_action_advantages[a].detach()
                    ''' flipped advantages'''
                    advantages = advantages*flipped_adv_list[a]
                    ''' flipped advantages'''
                    y = th.sign(advantages)
                    '''weight of advantage, WCE has the same gradients with PPO-clip'''
                    if self.aece == "WAE" or self.aece == "WCE":
                        # y = advantages
                        abs_adv = th.abs(advantages)
                    else:
                        abs_adv = th.abs(y)
                    
                    
                    policy_losses = []
                    from collections import OrderedDict
                    od = OrderedDict()
                    #policy_loss_data = []
                    t_loss_start = time.time()
                    
                    '''old code for small loss trick'''
                    # for key ,value in od:
                    #     # print("k,v",key,value)
                    #     if deltaYcounter>= deltaYnum:
                    #         break
                    #     deltaYs[key] = 1
                    #     deltaYcounter+=1
                    '''assign epsilon to marginalRankingloss'''
                    for i in range(self.batch_size):
                        if self.classifier == "AM":
                            epsilon[i] = self.alpha * min(1, prob_ratio[i])
                        elif self.classifier == "AM-log":
                            epsilon[i] = math.log(1 + self.alpha * min(1, prob_ratio[i]))
                        elif self.classifier == "AM-root":
                            epsilon[i] = math.sqrt(1 + self.alpha * min(1, prob_ratio[i])) - 1
                        elif self.classifier == "AM-sub":
                            epsilon[i] = minMu[i] * self.alpha * min(1, prob_ratio[i])
                        elif self.classifier == "AM-square":
                            epsilon[i] = ( 1 + self.alpha * min(1, prob_ratio[i]) )** 2
                        if self.aece == "WAE" or self.aece == "AE":
                            policy_loss_fn = th.nn.MarginRankingLoss(margin=epsilon[i])
                        elif self.aece == "WCE" or self.aece == "CE":
                            policy_loss_fn = th.nn.MarginRankingLoss(margin=self.alpha)
                        
                        # unclip_loss = epsilon[i] - y[i]*(x1[i]-x2[i])
                        
                        # policy_losses2.append((1-th.exp(val_log_prob[i]).item())*abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ))
                        # policy_losses2.append((1-th.exp(val_log_prob[i]).item())*abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ))
                        # policy_loss += abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]*(1-2*deltaYs[i] )) .unsqueeze(0) )
                        '''record sa pairs which are not igrored in the first epoch'''
                        if epoch == 0 and th.exp(full_sa_log_prob[a][i]).item() <= self.spt_clipped_prob:
                            # print("th.exp(full_sa_log_prob[a][i]).item()",th.exp(full_sa_log_prob[a][i]).item())
                            active_example_counter+=1
                        '''SPT'''
                        if th.exp(full_sa_log_prob[a][i]).item() <= self.spt_clipped_prob:
                            policy_losses2.append(abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ))
                            not_0_loss_counter+=1
                            
                        else:
                            # policy_losses2.append(th.tensor([0.] ).to(self.device) )
                            policy_losses2.append(0*abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ))
                        # # policy_losses2.append(abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]*(1-2*deltaYs[i] )) .unsqueeze(0) ))
                    t_loss_end = time.time()
                    loss_time.append(t_loss_end - t_loss_start)
                    # policy_loss /= self.batch_size
                    # policy_loss = th.stack(policy_losses2).sum() /self.batch_size
                    '''avoid dividing by 0'''
                    if not_0_loss_counter ==0 :
                        not_0_loss_counter = 1
                    '''mean of policy loss of active sa pairs(examples)'''
                    policy_loss = th.stack(policy_losses2).sum() / not_0_loss_counter
                
                if self.policy_update_scheme == 1 and (episode_counter % self.actor_delay) < self.action_space.n:
                    '''only update one action in a episode'''
                    a  =  int((episode_counter ) % self.actor_delay )
                    # for a in range(self.action_space.n) :
                    # a =  int((episode_counter/self.actor_delay) %  self.action_space.n )
                    # print("episode: ",episode_counter,"a: ",a)
                    # print("pitheta 0",th.exp(full_sa_log_prob[a][0]).item(),"old pi 0",th.exp(target_full_sa_log_prob[a][0]).detach().item())
                    '''different classifiers of HPO, only AM is used recently'''
                    if self.classifier == "AM":
                        # x1 = th.exp(val_log_prob - rollout_data.old_log_prob.detach()) # ratio
                        x1 = th.exp(full_sa_log_prob[a] - target_full_sa_log_prob[a].detach()) # ratio
                        # x1 = gpu_action_probs[a]/gpu_action_probs[a].detach() full_sa_log_prob[a]
                        x2 = th.ones_like(x1.clone().detach())
                    elif self.classifier == "AM-log":# log(pi) - log(mu)
                        x1 = full_sa_log_prob[a]
                        x2 = target_full_sa_log_prob[a].detach()
                    elif self.classifier == "AM-root":# root: (pi/mu)^(1/2) - 1
                        x1 = th.sqrt(th.exp(full_sa_log_prob[a] - target_full_sa_log_prob[a].detach())) # ratio
                        x2 = th.ones_like(x1.clone().detach())
                    # elif self.classifier == "AM-sub":
                    #     x1 = th.exp(val_log_prob )
                    #     x2 = th.exp(rollout_data.old_log_prob)
                    # elif self.classifier == "AM-square":
                    #     x1 = th.square(th.exp(val_log_prob - rollout_data.old_log_prob.detach())) # ratio
                    #     x2 = th.ones_like(x1.clone().detach())
                    #advantages = rollout_data.advantages.cpu().detach()
                    #print("advantages",advantages)
                    #abs_adv = np.abs(advantages.cpu())
                    # advantages = advantages.detach()
                    advantages = gpu_action_advantages[a].detach()
                    ''' flipped advantages with a predefined array generated in the start of the episode'''
                    advantages = advantages*flipped_adv_list[a]
                    
                    y = th.sign(advantages)
                    '''W means the weight(magnitude) of advantage'''
                    if self.aece == "WAE" or self.aece == "WCE":
                        # y = advantages
                        abs_adv = th.abs(advantages)
                    else:
                        abs_adv = th.abs(y)
                    
                    
                    policy_losses = []
                    from collections import OrderedDict
                    od = OrderedDict()
                    #policy_loss_data = []
                    t_loss_start = time.time()
                    
                    '''old code for small loss trick'''
                    # for key ,value in od:
                    #     # print("k,v",key,value)
                    #     if deltaYcounter>= deltaYnum:
                    #         break
                    #     deltaYs[key] = 1
                    #     deltaYcounter+=1
                    '''old code for adaptive epsilon, we only use CE or WCE(the same gradient of PPO-clip) recently'''
                    for i in range(self.batch_size):
                        if self.classifier == "AM":
                            epsilon[i] = self.alpha * min(1, prob_ratio[i])
                        elif self.classifier == "AM-log":
                            epsilon[i] = math.log(1 + self.alpha * min(1, prob_ratio[i]))
                        elif self.classifier == "AM-root":
                            epsilon[i] = math.sqrt(1 + self.alpha * min(1, prob_ratio[i])) - 1
                        elif self.classifier == "AM-sub":
                            epsilon[i] = minMu[i] * self.alpha * min(1, prob_ratio[i])
                        elif self.classifier == "AM-square":
                            epsilon[i] = ( 1 + self.alpha * min(1, prob_ratio[i]) )** 2
                        if self.aece == "WAE" or self.aece == "AE":
                            policy_loss_fn = th.nn.MarginRankingLoss(margin=epsilon[i])
                        elif self.aece == "WCE" or self.aece == "CE":
                            policy_loss_fn = th.nn.MarginRankingLoss(margin=self.alpha)
                        # policy_losses2.append((1-th.exp(val_log_prob[i]).item())*abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ))
                        # policy_losses2.append((1-th.exp(val_log_prob[i]).item())*abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ))
                        # policy_loss += abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]*(1-2*deltaYs[i] )) .unsqueeze(0) )
                        '''record how many s,a pairs are not ignored by SPT'''
                        if epoch == 0 and th.exp(full_sa_log_prob[a][i]).item() <= self.spt_clipped_prob:
                            active_example_counter+=1
                            # print("th.exp(full_sa_log_prob[a][i]).item()",th.exp(full_sa_log_prob[a][i]).item())
                        # print("(th.exp(full_sa_log_prob[a][i]).item()",th.exp(full_sa_log_prob[a][i]).item())
                        # print("y[i].item()",y[i].item())
                        '''record average policy loss in the last epoch'''
                        if epoch == self.n_epochs-1 and th.exp(full_sa_log_prob[a][i]).item() <= self.spt_clipped_prob and not (th.exp(full_sa_log_prob[a][i]).item() >= 0.99 and y[i].item()==1.0 ) :
                            # pg_log = policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ).item()
                            pg_losses.append(policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ).item())
                        
                        '''SPT implementation in HPO'''
                        if th.exp(full_sa_log_prob[a][i]).item() <= self.spt_clipped_prob:
                            policy_losses2.append(abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ))
                            not_0_loss_counter+=1
                        else:
                            # policy_losses2.append(th.tensor([0.] ).to(self.device) )
                            policy_losses2.append(0*abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ))
                        # # policy_losses2.append(abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]*(1-2*deltaYs[i] )) .unsqueeze(0) ))
                    t_loss_end = time.time()
                    loss_time.append(t_loss_end - t_loss_start)
                    # policy_loss /= self.batch_size
                    # policy_loss = th.stack(policy_losses2).sum() /self.batch_size
                    '''avoid dividing by 0'''
                    if not_0_loss_counter ==0 :
                        not_0_loss_counter = 1
                    policy_loss = th.stack(policy_losses2).sum() / not_0_loss_counter
                
                if self.policy_update_scheme == 2:
                    '''self.actor_delay must be larger than self.action_space.n'''
                    a  =  int((episode_counter ) % self.actor_delay )
                    # for a in range(self.action_space.n) :
                    # a =  int((episode_counter/self.actor_delay) %  self.action_space.n )
                    # print("episode: ",episode_counter,"a: ",a)
                    if self.classifier == "AM":
                        # x1 = th.exp(val_log_prob - rollout_data.old_log_prob.detach()) # ratio
                        x1 = th.exp(full_sa_log_prob[a] - target_full_sa_log_prob[a].detach()) # ratio
                        # x1 = gpu_action_probs[a]/gpu_action_probs[a].detach() full_sa_log_prob[a]
                        x2 = th.ones_like(x1.clone().detach())
                    elif self.classifier == "AM-log":# log(pi) - log(mu)
                        x1 = full_sa_log_prob[a]
                        x2 = target_full_sa_log_prob[a].detach()
                    elif self.classifier == "AM-root":# root: (pi/mu)^(1/2) - 1
                        x1 = th.sqrt(th.exp(full_sa_log_prob[a] - target_full_sa_log_prob[a].detach())) # ratio
                        x2 = th.ones_like(x1.clone().detach())
                    # elif self.classifier == "AM-sub":
                    #     x1 = th.exp(val_log_prob )
                    #     x2 = th.exp(rollout_data.old_log_prob)
                    # elif self.classifier == "AM-square":
                    #     x1 = th.square(th.exp(val_log_prob - rollout_data.old_log_prob.detach())) # ratio
                    #     x2 = th.ones_like(x1.clone().detach())
                    #advantages = rollout_data.advantages.cpu().detach()
                    #print("advantages",advantages)
                    #abs_adv = np.abs(advantages.cpu())
                    # advantages = advantages.detach()
                    advantages = gpu_action_advantages[a].detach()
                    ''' flipped advantages noise'''
                    advantages = advantages*flipped_adv_list[a]
                    ''' flipped advantages noise'''
                    y = th.sign(advantages)
                    if self.aece == "WAE" or self.aece == "WCE":
                        # y = advantages
                        abs_adv = th.abs(advantages)
                    else:
                        abs_adv = th.abs(y)
                    
                    
                    policy_losses = []
                    from collections import OrderedDict
                    od = OrderedDict()
                    #policy_loss_data = []
                    t_loss_start = time.time()
                    
                    
                    # for key ,value in od:
                    #     # print("k,v",key,value)
                    #     if deltaYcounter>= deltaYnum:
                    #         break
                    #     deltaYs[key] = 1
                    #     deltaYcounter+=1
                    for i in range(self.batch_size):
                        if self.classifier == "AM":
                            epsilon[i] = self.alpha * min(1, prob_ratio[i])
                        elif self.classifier == "AM-log":
                            epsilon[i] = math.log(1 + self.alpha * min(1, prob_ratio[i]))
                        elif self.classifier == "AM-root":
                            epsilon[i] = math.sqrt(1 + self.alpha * min(1, prob_ratio[i])) - 1
                        elif self.classifier == "AM-sub":
                            epsilon[i] = minMu[i] * self.alpha * min(1, prob_ratio[i])
                        elif self.classifier == "AM-square":
                            epsilon[i] = ( 1 + self.alpha * min(1, prob_ratio[i]) )** 2
                        if self.aece == "WAE" or self.aece == "AE":
                            policy_loss_fn = th.nn.MarginRankingLoss(margin=epsilon[i])
                        elif self.aece == "WCE" or self.aece == "CE":
                            policy_loss_fn = th.nn.MarginRankingLoss(margin=self.alpha)
                        # policy_losses2.append((1-th.exp(val_log_prob[i]).item())*abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ))
                        # policy_losses2.append((1-th.exp(val_log_prob[i]).item())*abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ))
                        unclip_loss = epsilon[i] - y[i]*(x1[i].unsqueeze(0)-x2[i].unsqueeze(0))
                        # policy_loss += abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]*(1-2*deltaYs[i] )) .unsqueeze(0) )
                        if epoch == 0 and th.exp(full_sa_log_prob[a][i]).item() <= self.spt_clipped_prob:
                            active_example_counter+=1
                        print("(th.exp(full_sa_log_prob[a][i]).item()",th.exp(full_sa_log_prob[a][i]).item())
                        print("y[i].item()",y[i].item())
                        if epoch == self.n_epochs-1 and th.exp(full_sa_log_prob[a][i]).item() <= self.spt_clipped_prob and not (th.exp(full_sa_log_prob[a][i]).item() >= 0.995 and y[i].item()==1.0 ) :
                            # pg_log = policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ).item()
                            pg_losses.append(policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ).item())
                        if th.exp(full_sa_log_prob[a][i]).item() <= self.spt_clipped_prob:
                            policy_losses2.append(abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ))
                            not_0_loss_counter+=1
                        else:
                            # policy_losses2.append(th.tensor([0.] ).to(self.device) )
                            policy_losses2.append(0*abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]  ) .unsqueeze(0) ))
                        # # policy_losses2.append(abs_adv[i] * policy_loss_fn( x1[i].unsqueeze(0) , x2[i].unsqueeze(0) , (y[i]*(1-2*deltaYs[i] )) .unsqueeze(0) ))
                    t_loss_end = time.time()
                    loss_time.append(t_loss_end - t_loss_start)
                    # policy_loss /= self.batch_size
                    # policy_loss = th.stack(policy_losses2).sum() /self.batch_size
                    if not_0_loss_counter ==0 :
                        not_0_loss_counter = 1
                    policy_loss = th.stack(policy_losses2).sum() / not_0_loss_counter
                #print("Policy loss", policy_loss_data)
                # debug 6
                margins.append(epsilon)

                # Logging
                # if epoch == self.n_epochs-1:
                #     pg_losses.append(policy_loss.item())
                #pg_losses.append(policy_loss)
                #clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                #clip_fractions.append(clip_fraction)
                # print("val_q_values:",val_q_values.requires_grad)
                val_q_values = th.stack(val_q_values)
                '''the clipping value function update technique, but I have never tried this'''
                if self.clip_range_vf is None:
                    # No clipping
                    #values_pred = val_values # org version
                    values_pred = val_q_values # org version
                    #values_pred = th.exp(val_log_prob) * val_values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        val_values - rollout_data.old_values, -clip_range_vf, clip_range_vf # org version
                        #th.exp(val_log_probs) * val_values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                #value_loss = F.mse_loss(rollout_data.returns.unsqueeze(1), values_pred )
                '''MSE( Computed V value, discounted cumulative rewards computed by rollout_buffer.compute_returns_and_advantage function) '''
                value_loss = F.mse_loss(rollout_data.returns, values_pred )
                value_losses.append(value_loss.item())

                rollout_return.append(rollout_data.returns.detach().cpu().numpy())

                # ?? SKIP entropy loss??
                # entropy = None
                # Entropy loss favor exploration
                '''other people did not allow me to add entropy term in HPO loss,
                 but I think this is one of the techniques that make PPO-clip succeed? 
                 If you want to use entropy, remember to add --entropy_hpo in the argument'''
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-val_log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())
                
                # org version
                # print("policy_loss.requires_grad:",policy_loss.requires_grad)
                '''If using independent_value_net, we need another optimizer to minimize the value loss'''
                if self.entropy_hpo == True and self.independent_value_net == False:
                    loss = self.pf_coef *policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss ##0810 test ,hope to find 360 HPO-62
                elif  self.entropy_hpo == True and self.independent_value_net == True:
                    loss = self.pf_coef * policy_loss+ self.ent_coef * entropy_loss
                elif self.independent_value_net == True :
                    loss = self.pf_coef * policy_loss
                    # vloss = 
                else : # if self.independent_value_net == False
                    loss = self.pf_coef * policy_loss + self.vf_coef * value_loss
                    # if episode_counter % self.actor_delay == 0:
                    #     loss = self.pf_coef * policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss ##08070 final HPO-63 with 304
                    # else :
                    #     loss = self.vf_coef * value_loss
                
                # loss = policy_loss + self.vf_coef * value_loss ##08070 final HPO-63 with 304
                #loss = policy_loss
                #loss = th.stack(policy_loss).sum() + self.vf_coef * value_loss

                ## Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                ## Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                #approx_kl_divs.append(th.mean(rollout_data.old_log_prob - val_log_prob).detach().cpu().numpy())
                approx_kl_divs.append(th.mean(rollout_data.old_log_prob.detach() - val_log_prob).detach().cpu().numpy())
                '''If using independent_value_net, we need another optimizer to minimize the value loss'''
                if self.independent_value_net:
                    self.value_policy.optimizer.zero_grad()
                    value_loss = self.vf_coef * value_loss
                    value_loss.backward()
                    self.value_policy.optimizer.step()
                ## value policy
                #value_loss = self.vf_coef * value_loss
                #self.value_policy.optimizer.zero_grad()
                #value_loss.backward()
                ### Clip grad norm
                #th.nn.utils.clip_grad_norm_(self.value_policy.parameters(), self.max_grad_norm)
                #self.value_policy.optimizer.step()

            all_kl_divs.append(np.mean(approx_kl_divs))

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break
            #if np.mean(pg_losses) < 1e-4:
            #    print(f"Early stopping at step {epoch} due to reaching ploss: {np.mean(pg_losses):.2f}")
            #    break
        # th.cuda.empty_cache()
        print("rollbutffer_get_counter",rollbutffer_get_counter)
        t_epoch_end = time.time()
        epoch_time.append(t_epoch_end - t_epoch_start)
        logger.record("Time/train_loss/Sum", np.sum(loss_time))
        logger.record("Time/train_computeV/Sum", np.sum(computeV_time))
        logger.record("Time/train_epoch/Sum", np.sum(epoch_time))
        logger.record("Time/train_action_adv/Sum", np.sum(t_action_adv_time))
        logger.record("Time/train_loss/Mean", np.mean(loss_time))
        logger.record("Time/train_computeV/Mean", np.mean(computeV_time))
        logger.record("Time/train_epoch/Mean", np.mean(epoch_time))
        logger.record("Time/train_action_adv/Mean", np.mean(t_action_adv_time))
        
        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        if len(pg_losses) == 0:
            pg_losses.append(-0.1)
        # Logs
        logger.record("train/active_example", active_example_counter)
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/approx_kl", np.mean(approx_kl_divs))
        #logger.record("train/clip_fraction", np.mean(clip_fractions))
        logger.record("train/loss", loss.item())
        logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        # logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/n_updates", self._n_updates)
        logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.record("train/clip_range_vf", clip_range_vf)

        # HPO
        logger.record("HPO/margin", np.mean(margins))
        #logger.record("HPO/positive_advantage", np.mean(positive_a))
        #logger.record("HPO/negative_advantage", np.mean(negative_a))
        logger.record("HPO/positive_advantage_prob", np.mean(positive_p))
        logger.record("HPO/negative_advantage_prob", np.mean(negative_p))
        logger.record("HPO/prob_ratio", np.mean(ratio_p))
        logger.record("HPO/rollout_return", np.mean(rollout_return))

    

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        # n_eval_episodes: int = 5,
        n_eval_episodes: int = 100,
        tb_log_name: str = "HPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "HPO":
        print("n_eval_episodes",n_eval_episodes)
        return super(HPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
