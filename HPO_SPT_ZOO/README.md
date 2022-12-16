
# HPO_SPT + RL Baselines3 Zoo: HPO_SPT Based on the Training Framework for Stable Baselines3 Reinforcement Learning Agents


[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) is a training framework for Reinforcement Learning (RL), using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).

It provides scripts for training, evaluating agents, tuning hyperparameters, plotting results and recording videos.

In addition, it includes a collection of tuned hyperparameters for common environments and RL algorithms, and agents trained with those settings.

We add our HPO algorithm to the framework.

## Installation

<!-- ### Stable-Baselines3 PyPi Package

Min version: stable-baselines3[extra] >= 1.0
and sb3_contrib >= 1.0

```
apt-get install swig cmake ffmpeg
pip install -r requirements.txt
```

Please see [Stable Baselines3 README](https://github.com/DLR-RM/stable-baselines3) for alternatives. -->

Install this with packages in rlbaseline20220719.yml.
```
conda env create -f rlbaseline20220719.yml
```
## Usage
### Train an Agent
For example
```
python train.py --algo algo_name --env env_id --seed 123 --track --wandb-project-name PROJECT_NAME  --hyperparams PARAMS1:1.0 policy_kwargs:"dict(optimizer_class=th.optim.SGD)"
```

HPO example(remember to set the classifier, aece in arguments, policy_update_scheme, actor_delay in hyperparamters)
```
python train.py --algo hpo --classifier AM --aece CE --seed 16842464 --env LunarLander-v2 --track --wandb-project-name sptLunarlander20epsilon3 --hyperparams rgamma:1.0 advantage_flipped_rate:0.2 n_timesteps:4000000 policy_update_scheme:1 device:0 actor_delay:4 policy_kwargs:"dict(optimizer_class=th.optim.SGD)" independent_value_net:True n_epochs:20 learning_rate:0.003 alpha:0.3
```
#### Experiment tracking
Tracking experiment data such as learning curves and hyperparameters via Weights and Biases with arguments "--track --wandb-project-name PROJECT_NAME".

The following command
```
python train.py --algo ppo --env CartPole-v1 --track --wandb-project-name sb3
```
yields a tracked experiment at the wandb website.

#### Arguments
 - classifier: classifiers of HPO (please choose AM)
 - aece: adaptive margin (epsilon of PPO-clip) (please choose CE)
 - seed: random seed
 - entropy_hpo: If you want to use entropy in HPO, remember to add --entropy_hpo in the argument

#### Overwrite hyperparameters
You can easily overwrite hyperparameters in the command line, using --hyperparams:
see HPO example


#### Hyperparameters of HPO + SPT
 - advantage_flipped_rate: flip rate of estimated advantage
 - rgamma: threshold of SPT
 - policy_update_scheme: scheme for actor delay and one action per state (Assumption: In each episode t, the state-action pairs in D(t) have distinct states)
 , the scheme will make the results have high variance. Only scheme 1 is used recently. 
 Maybe we should change the scheme to consider the loss of all actions of the state-action pairs in D(t).
 - actor_delay: actor delay must be larger than action space of the env in shceme 1. if actor_delay>action space of the env, 
 episode ((actor_delay-action space)~(actor_delay-1))+n*actor_delay will not update the policy.
 - independent_value_net: use another NN to estimate the Q value.
 - alpha:epsilon of PPO-Clip, margin of HPO
 - policy_kwargs: we can specify the optimizer or initialization of policy.

### Enjoy a Trained Agent
For example
```
python enjoy.py --algo algo_name --env env_id
```
```
python enjoy.py --algo hpo --env CartPole-v1 -f logs/ --exp-id 147 --stochastic
```

### Add a new argument to RL Baselines3 Zoo
Please see [this](https://hackmd.io/@_BK2lUeVSI6hlsvNLM-iLQ/SJgLZGd9u).

### Don't use the results of the eval, because they are generated with deterministic == True, which is hard to be affected by noisy critics. Use the results of the rollout/ep_rew_mean, which is sampled according to the distribution.
