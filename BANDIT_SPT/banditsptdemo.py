import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gym.envs.registration import registry, register, make, spec
import pdb

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='G',
                    help='learning-rate of optimizer (default: 1e-4)')
parser.add_argument('--classifier', type=str, required=True,
                    help='classifier of policy loss (PPO or PI_MU_1 or root)')
parser.add_argument('--value-method', type=str, required=True,
                    help='returns based or Qvalue (returns or QValue)')
parser.add_argument('--max-episode', type=int, default=10000, metavar='N',
                    help='max-episode (default: 10000)')
parser.add_argument('--actor-delay', type=int, default=1, metavar='N',
                    help='actor-delay (default: 1)')
parser.add_argument('--wandb', action='store_true',
                    help='use wandb for profiling')
parser.add_argument('--margin', type=float, default=0.1, metavar='G',
                    help='margin (default: 0.1)')
parser.add_argument('--weight', action='store_true',
                    help='marginal loss y = advantage (default:y = advantage.sign())')
parser.add_argument('--fixedEps', type=str,
                    help='fixedEps (fixedEps or fixedEpsWeight)')
parser.add_argument('--envi', type=str,
                    help='4x4 or 8x8')
parser.add_argument('--flipped', type=float, default=0.0,
                    help='flipped proportion of sign of true advantage')
parser.add_argument('--robustEpoch', type=int, default=5,
                    help='how many robust epoches in a normal epoch')
parser.add_argument('--rgamma', type=float, default=0.0,
                    help='how much deltaY would set 1')
parser.add_argument('--gammachoosing', type=str,
                    help='slt or rc')
parser.add_argument('--duration', type=int, default=0,
                    help='how many epoches with same flipped state')
parser.add_argument('--zeta', type=float, default=0.0000001,
                    help='update until loss diff < zeta ')
parser.add_argument('--nepoch', type=int, default=10,
                    help=' how much loss backward will happen in 1 episode ')
parser.add_argument('--sptth', type=float, default=0.7,
                    help='will not update s a if pi(s,a)>args.sptth ')
parser.add_argument('--probUB', type=float, default=0.99,
                    help='restrict the upper bound of probability to avoid NN get stuck ')
parser.add_argument('--modelsavepath', type=str,
                    help='model save path')
parser.add_argument('--modelloadpath', type=str,
                    help='model load path')
parser.add_argument('--optimizer', type=str,
                    help='optimizer sgd or adam')
args = parser.parse_args()
# MARGIN = 0.1 args.fixedEps

 
register(id='gridworld_randR_env-v0',entry_point='gridworld_randR_env:Gridworld_banditcase_RandReward_Env',reward_threshold=500.0,)
GRIDWIDTH = 1

envname = 'gridworld_randR_env-v0'
ACTION_DIM = 5

import gym
env = gym.make(envname)
import random
# torch.manual_seed(123)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
env.seed(args.seed)
# torch.manual_seed(args.seed)
torch.manual_seed(args.seed)
print("env.observation_space.shape:",env.observation_space.shape)
last_episode_log = []
loss_log = []
ploss_log = []
vloss_log = []

# flipped_proportion = args.flipped
import os



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from datetime import datetime
hpg_checkpoint_dir = "./models/hpg_fch256/"+envname
if not os.path.exists(hpg_checkpoint_dir):
    os.makedirs(hpg_checkpoint_dir)
'''Set the initial policy, first time advantage and rewards'''
choosed_sa = [0.0025,0.99,0.0025,0.0025,0.0025]
# choosed_sa = [0.01,0.01,0.01,0.01,0.96]
# choosed_sa = [0.1,0.1,0.1,0.6,0.1]
# choosed_sa = [0.2,0.2,0.2,0.2,0.2]
# choosed_sa = [0.05,0.05,0.05,0.05,0.8]
# specified_adv = [ -1,-1,-1,1,-1 ]
specified_adv = [ -1,-1,-1,-1, -1 ]
# PE_rewards = [5,4,3,2,1]
PE_rewards = [1,2,3,4,5]
# specified_adv = [ 1,1,1,1, -1 ]
# specified_adv = [ 1,-1,-1,1,-10 ]
# specified_adv = [ -1,-1,-1,-1, 1 ]
# specified_adv = [ -1,-1,1,-1,-1 ]
csastring = "["+''.join(str( x )+"," for x in choosed_sa )
csasignstring = "["+''.join( str( x )+"," for x in specified_adv )
log_dir = "./logs/bandit_sptDemo_{}/{}_{}".format( args.classifier, args.envi, datetime.now().strftime("%m-%d-%H-%M-%S"))+"_weight"+str(args.weight)+"_fixedEps"+str(args.fixedEps)+ "_alpha"+str(args.margin)  + "_seed" + str(args.seed)+"_flipped_proportion"+str(args.flipped)+"_rgamma"+str(args.rgamma)  +"_duration"+str(args.duration)+ "_initProb"+csastring+"]assigned_sign"+csasignstring+"]"+"_max_episode"+str(args.max_episode)+"_npoch"+str(args.nepoch)+"sptthreshold"+str(args.sptth)+"_probUB"+str(args.probUB)+"_optimizer"+str(args.optimizer)
# log_dir = "./logs/hpg_fch256/{}_{}_{}_{}".format(envname, datetime.now().strftime("%m-%d-%H-%M-%S"), args.classifier ,args.value_method )
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

def hard_update(source,target):
    for source_param, target_param in zip(source.parameters(),target.parameters()):
        target_param.data.copy_(source_param.data)

def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.1)


eps = np.finfo(np.float32).eps.item()


'''Policy evaluation to get the true sign of advantage'''
def get_sign_advantages_policy_eval_duration(flipped_proportion,model,duration,i_episode):
    
    flipped_arr = np.full(env.observation_space.n*ACTION_DIM,1)
    flip_num = int (flipped_proportion * env.observation_space.n*ACTION_DIM)
    if i_episode % duration == 0:
        print("renew ret")
        get_sign_advantages_policy_eval_duration.ret= []
    else :
        print("old ret",get_sign_advantages_policy_eval_duration.ret )
        get_sign_advantages_policy_eval_duration.ret = get_sign_advantages_policy_eval_duration.ret
        return get_sign_advantages_policy_eval_duration.ret 
    flipped_arr[0:flip_num] = -1
    # print("flipped_arr",flipped_arr)
    theta =0.00001
    # average_reward = -0.1
    # rewards = [-2,-1,0,1,2]
    rewards = PE_rewards
    policy_eval_counter = 0 
    while True:
        policy_eval_counter += 1 
        delta = 0
        for si in range( env.observation_space.n):
            s_onehot = np.zeros(env.observation_space.n)
            s_onehot[si] = 1
            # s_onehot = 0
            state = torch.from_numpy(s_onehot).float()
            probs = model(state).clone().detach().numpy()
            for aj in range(ACTION_DIM):# update every q(s,a)
                
                tmp = model.past_value_table[si, aj]
                tmp2 = 0
                for sj in range(env.observation_space.n):# sprime
                    s_onehot2 = np.zeros(env.observation_space.n)
                    s_onehot2[si] = 1

                    statej = torch.from_numpy(s_onehot2).float()
                    probs2 = model(statej).clone().detach().numpy()
                    if env.P[ aj, si , sj ]>0:
                        Vsj=0
                        
                        # for ak in range(ACTION_DIM):
                        #     Vsj+=probs2[ ak ]*model.past_value_table[sj, ak]
                        # for ak in range(ACTION_DIM):
                            # tmp2 += model.past_prob_table[si, ak]*env.P[ ak, si , sj ]*( average_reward+ args.gamma*Vsj )
                        tmp2 = env.P[ aj, si , sj ]*( rewards[aj]+ args.gamma*Vsj )
                        # print("env.P[",aj,si,sj,"]",env.P[ aj, si , sj ])
                        # print("Vsj[",sj,"]" ,Vsj)
                        # print("model.past_value_table[",si,aj,"]",tmp2)
                        
                        delta = max( delta,abs( tmp2-tmp) )
                        model.past_value_table[si, aj] = tmp2

        if delta<theta:
            break
    # advantages = []
    # import math
    print("model.past_value_table",model.past_value_table)
    for si in range(env.observation_space.n):
        Vs = 0
        s_onehot = np.zeros(env.observation_space.n)
        s_onehot[si] = 1
        # s_onehot = 0
        state = torch.from_numpy(s_onehot).float()
        probs = model(state).clone().detach().numpy()
        for aj in range(ACTION_DIM):
            Vs+=probs[  aj]*model.past_value_table[si, aj]
        print("vs",Vs)
        for aj in range(ACTION_DIM):
            idx = si*ACTION_DIM + aj
            # advantages.append( model.past_value_table[si, aj] - Vs )
            # flip_p = flipped_proportion * math.exp( -1* abs(model.past_value_table[si, aj] - Vs ))
            # flipped_arr[idx] = np.random.choice( [1,-1], p=[1-flip_p,flip_p])
            if model.past_value_table[si, aj]>Vs:
                get_sign_advantages_policy_eval_duration.ret.append(1)
            else:
                get_sign_advantages_policy_eval_duration.ret.append(-1)


    np.random.shuffle(flipped_arr[:])
    print("flipped_arr",flipped_arr)
    print("ret",get_sign_advantages_policy_eval_duration.ret)
    get_sign_advantages_policy_eval_duration.ret = flipped_arr*get_sign_advantages_policy_eval_duration.ret
    print("after flipped ret",get_sign_advantages_policy_eval_duration.ret)
    return get_sign_advantages_policy_eval_duration.ret


def get_sign_advantages_policy_eval(flipped_proportion,model):
    ret= []
    flipped_arr = np.full(env.observation_space.n*ACTION_DIM,1)
    flip_num = int (flipped_proportion * env.observation_space.n*ACTION_DIM)
    flipped_arr[0:flip_num] = -1
    theta =0.001
    # average_reward = -0.1
    # rewards = [-2,-1,0,1,2]
    rewards = PE_rewards
    policy_eval_counter = 0 
    while True:
        policy_eval_counter += 1 
        delta = 0
        for si in range( env.observation_space.n):
            s_onehot = np.zeros(env.observation_space.n)
            s_onehot[si] = 1
            # s_onehot = 0
            state = torch.from_numpy(s_onehot).float()
            probs = model(state).clone().detach().numpy()
            for aj in range(ACTION_DIM):# update every q(s,a)
                
                tmp = model.past_value_table[si, aj]
                tmp2 = 0
                for sj in range(env.observation_space.n):# sprime
                    s_onehot2 = np.zeros(env.observation_space.n)
                    s_onehot2[si] = 1

                    statej = torch.from_numpy(s_onehot2).float()
                    probs2 = model(statej).clone().detach().numpy()
                    if env.P[ aj, si , sj ]>0:
                        Vsj=0
                        
                        for ak in range(ACTION_DIM):
                            Vsj+=probs2[ ak ]*model.past_value_table[sj, ak]
                        # for ak in range(ACTION_DIM):
                            # tmp2 += model.past_prob_table[si, ak]*env.P[ ak, si , sj ]*( average_reward+ args.gamma*Vsj )
                        tmp2 = env.P[ aj, si , sj ]*( rewards[aj]+ args.gamma*Vsj )
                        # print("env.P[",aj,si,sj,"]",env.P[ aj, si , sj ])
                        # print("Vsj[",sj,"]" ,Vsj)
                        # print("model.past_value_table[",si,aj,"]",tmp2)
                        
                        delta = max( delta,abs( tmp2-tmp) )
                        model.past_value_table[si, aj] = tmp2

        if delta<theta:
            break
    # advantages = []
    # import math
    for si in range(env.observation_space.n):
        Vs = 0
        s_onehot = np.zeros(env.observation_space.n)
        s_onehot[si] = 1
        # s_onehot = 0
        state = torch.from_numpy(s_onehot).float()
        probs = model(state).clone().detach().numpy()
        for aj in range(ACTION_DIM):
            Vs+=probs[  aj]*model.past_value_table[si, aj]
        for aj in range(ACTION_DIM):
            idx = si*ACTION_DIM + aj
            # advantages.append( model.past_value_table[si, aj] - Vs )
            # flip_p = flipped_proportion * math.exp( -1* abs(model.past_value_table[si, aj] - Vs ))
            # flipped_arr[idx] = np.random.choice( [1,-1], p=[1-flip_p,flip_p])
            if model.past_value_table[si, aj]>Vs:
                ret.append(1)
            else:
                ret.append(-1)


    np.random.shuffle(flipped_arr[:])
    # print("flipped_arr",flipped_arr)
    print("ret",ret)
    ret = flipped_arr*ret
    print("after flipped ret",ret)
    return ret

# hinge

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear( env.observation_space.n , 32)
        # self.affine1.apply(init_weights)
        self.affine2 = nn.Linear(32, 32)
        # self.affine2.apply(init_weights)
        self.affine3 = nn.Linear(64, 32)
        # self.affine3 = nn.Linear(32, 32)
        # self.affine3.apply(init_weights)
        # actor's layer
        self.action_head = nn.Linear(32, ACTION_DIM)

        # critic's layer 1->4  v->q
        # self.value_head = nn.Linear(32, 1)
        # if args.value_method == 'returns':
        #     self.value_head = nn.Linear(32, 1)
        # elif args.value_method == 'Qvalue':
        #     self.value_head = nn.Linear(32, ACTION_DIM)
        # action & reward buffer
        self.past_value_table = np.full((env.observation_space.n , env.action_space.n ),0.0)
        # self.past_value_table[1] = np.array([-0.398, -0.59402, -0.59402, -0.2])
        # self.past_value_table[2] = np.array([-0.59402, -0.7880798, -0.7880798, -0.398])
        # self.past_value_table[3] = np.array([-0.7880798, -0.7880798, -0.980199, -0.59402])
        # self.past_value_table[4] = np.array([-0.2, -0.59402, -0.59402, -0.398])
        # self.past_value_table[5] = np.array([-0.398, -0.7880798, -0.7880798, -0.398])
        # self.past_value_table[6] = np.array([-0.59402, -0.980199, -0.980199,-0.59402])
        # self.past_value_table[7] = np.array([-0.7880798, -0.980199, -1.17039701, -0.7880798])
        # self.past_value_table[8] = np.array([-0.398, -0.7880798, -0.7880798, -0.59402])
        # self.past_value_table[9] = np.array([-0.59402, -0.980199, -0.980199, -0.59402])
        # self.past_value_table[10] = np.array([-0.7880798, -1.17039701, -1.17039701, -0.7880798])
        # self.past_value_table[11] = np.array([-0.980199, -1.17039701, -1.335869304, -0.980199])
        # self.past_value_table[12] = np.array([-0.59402, -0.9880199, -0.7880798, -0.7880798])
        # self.past_value_table[13] = np.array([-0.7880798, -1.17039701, -0.980199, -0.7880798])
        # self.past_value_table[14] = np.array([-0.980199, -1.35869304, -1.35869304, -1.17039701])
        # self.past_value_table[15] = np.array([-1.17039701, -1.35869304, -1.35869304, -1.17039701])
        self.saved_actions = []
        self.rewards = []
    def set_value(self, state, action, value):
        self.past_value_table[state, action] = value
    def update_value(self, state, action, value):
        #print("Loss", value, state, action)
        #print("Loss", value)
        #print("value table", self.past_value_table[state,:])
        #print(self.past_value_table[state, action])
        self.past_value_table[state, action] -= 0.01*value
    def forward(self, x):
        """
        forward of both actor and critic
        """
        
        x = self.affine1(x)
        x = self.affine2(x)
        # x = self.affine3(x)
        # x = F.relu(self.affine3(x))

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        # state_values = self.value_head( nn.Sigmoid(x) )
        # state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        # return action_prob, state_values
        return action_prob
    def get_head(self, x):
        x = self.affine1(x)
        x = self.affine2(x)
        # action_prob = F.softmax(self.action_head(x), dim=-1)
        # self.action_head(x)
        return self.action_head(x)

model = Policy() 


if args.wandb :
    wandb.watch(model)
old_model = Policy() 

'''Choosing optimizer: 'adam' or 'sgd' '''
if args.optimizer=='adam':
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
eps = np.finfo(np.float32).eps.item()

'''Get the old prob to compute the ratio of prob'''
def past_log_p(state, action,old_model):
    #state = torch.from_numpy(state).float().unsqueeze(0) 
    #state = torch.from_numpy(state).float().unsqueeze(0) 
    probs  = old_model(state) #state value -> 4 q (s,a)
    m = Categorical(probs)
    return m.log_prob(action)
def past_p(state, action,old_model):
    #state = torch.from_numpy(state).float().unsqueeze(0) 
    #state = torch.from_numpy(state).float().unsqueeze(0) 
    probs  = old_model(state) #state value -> 4 q (s,a)
    # print("past_p probs",probs)
    # m = Categorical(probs)
    return probs[action]

'''not used function'''
def select_action_tabular(state):
    state = torch.from_numpy(state).float() 
    # print("select_action_tabular state", state)
    state_numpy =  np.squeeze( np.where( state.detach().numpy() == 1 ) )
    probs = model(state) #state value -> 4 q (s,a)
    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)
    # and sample an action using the distribution
    action = m.sample()
    # print("model.past_value_table[state]",model.past_value_table[state_numpy,:])
    # [-0.2     -0.59402 -0.59402 -0.398  ]
    state_value = model.past_value_table[state_numpy,:]
    # state_value = model.past_value_table[state,:][0]
    return probs, state_value,state,action,m.log_prob(action)
'''not used function'''
def select_action(state):
    state = torch.from_numpy(state).float() 
    # print("state",state)
    probs, state_value = model(state) #state value -> 4 q (s,a)
    # print("probs",probs)
    # print("state_value",state_value)
    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()
    return probs, state_value,state,action,m.log_prob(action) #4 items probs ,4 items q s value , 1 state ,1 action

## hinge loss finish
# def finish_episode(past_prob_table):
def finish_episode(episode,past_prob_table):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    '''I try to make the code close to the assumptions in papers (uniformly choose an action of s, for all s in D(t)) 
    by choosing the action round-robin  '''
    '''choose an action: to satisfy the Assumption(In each iteration t, the state-action pairs in D(t) have distinct states.) and 
    make the influence of the incorrect advantage will not be soothed by other actions in the same state'''
    R = 0
    
    saved_actions = model.saved_actions
    print("args.margin",args.margin)
    choosed_s = 0
    # choosed_sa = [0.1,0.1,0.1,0.4,0.3]
    # # specified_adv = [ -1,-1,-1,1,-1 ]
    # # specified_adv = [ -1,-1,-1,-1, -1 ]
    # specified_adv = [ -1,-1,-1,1,-1 ]
    
    sa_set_zeta = 0.0001
    '''Load a trained model and the above initial policy setting will be ignored'''
    
    if episode == 0 and args.modelloadpath == None:
        old_specified_loss = torch.tensor(100000.0)
        while True:
            si = choosed_s
            s_losses = []
            # idx = si*ACTION_DIM + aj
            s_onehot = np.zeros(env.observation_space.n)
            s_onehot[si] = 1
            _state = torch.from_numpy(s_onehot).float()
            probs = model(_state)
            sloss = nn.L1Loss()
            for aj in range(ACTION_DIM):
                # print("probs[aj] ",probs[aj])
                
                _x2 = torch.from_numpy( np.array(  [choosed_sa[aj] ] ) )
                # print("x2 ",_x2)
                specified_loss = sloss( probs[aj].unsqueeze(0),  _x2  )
                s_losses.append(specified_loss)
            optimizer.zero_grad()
            loss = torch.stack(s_losses).sum()
            # print("loss",loss)
            # loss=loss.float()
            # perform backprop
            loss.backward()
            optimizer.step()
            zeta = 0.00001
            # if old_specified_loss.clone().detach().numpy()-loss.clone().detach().numpy()<zeta:
            if loss.clone().detach().numpy()<sa_set_zeta:
                break
            old_specified_loss = loss
        hard_update(model, old_model)
        eval2ret , l1_norm = evaluation2(model,episode)
        print("after set l1 norm",l1_norm)
    '''make the flipped advantages hold for args.duration episodes.'''
    if args.duration == 0:
        sign_Advantages = get_sign_advantages_policy_eval(args.flipped,model)
    else :
        sign_Advantages = get_sign_advantages_policy_eval_duration(args.flipped,model,args.duration, episode)
    print("true adv")
    true_sign_Advantages = get_sign_advantages_policy_eval(0,model)
    if episode < ACTION_DIM:
        sign_Advantages[choosed_s*ACTION_DIM:(choosed_s+1)*ACTION_DIM] = specified_adv
        print("changed sign_Advantage",sign_Advantages)
 
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values
    rewards = []
    value_gradients = []
    x1s = []
    epsilons = []
    x2s = []
    ys = []
    yweight = []
    # calculate the true value using rewards returned from the environment
    # print("args.value_method",args.value_method)
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        # print("R",R)
        returns.insert(0, R)
        rewards.insert(0, r)
    
    for si in range( env.observation_space.n):
        pos_adv_prob_sum = 0.0
        neg_adv_prob_sum = 0.0
        Vs = 0
        s_onehot = np.zeros(env.observation_space.n)
        s_onehot[si] = 1
        # s_onehot = 0
        state = torch.from_numpy(s_onehot).float()
        probs = model(state)
        # for aj in range(ACTION_DIM):

        #     # Vs+=model.past_prob_table[si, aj]*model.past_value_table[si, aj]
        for aj in range(ACTION_DIM):
            # if model.past_value_table[si, aj] > Vs:
            index = si*ACTION_DIM+aj
            if sign_Advantages[index]>0:
                # pos_adv_prob_sum+= model.past_prob_table[si, aj]
                pos_adv_prob_sum+=probs[aj].detach().numpy()
            # elif model.past_value_table[si, aj] < Vs:
            elif sign_Advantages[index]<0:
                neg_adv_prob_sum+=probs[aj].detach().numpy()
        # if args.fixedEps == 'fixedEps':
        #     epsilon = args.margin
        # else:
        #     epsilon = args.margin * min(1.0, neg_adv_prob_sum / (pos_adv_prob_sum + 1e-8))
        epsilon = args.margin
        epsilons.append(epsilon)#epsilons have |s|
        for aj in range(ACTION_DIM):
            m = Categorical(probs)
            action = torch.tensor(aj)
            log_p = m.log_prob(action)
            if args.classifier == 'AM-log':
                x1 = log_p.unsqueeze(0)
                x2 = past_log_p(state, aj,old_model).clone().detach().unsqueeze(0)
            elif args.classifier == 'AM':
                # Classifier 2. pi/mu - 1
                x1 = probs[aj].unsqueeze(0) / past_p(state, aj,old_model).clone().detach().unsqueeze(0)
                x2 = torch.from_numpy( np.array([1]) )
            elif args.classifier == 'AM-root':
                # Classifier 2. root(pi/mu) - 1
                x1 = torch.sqrt( probs[aj].unsqueeze(0) / past_p(state, aj ,old_model).clone().detach().unsqueeze(0) )
                x2 = torch.from_numpy( np.array([1]) )
            elif args.classifier == 'AM-sub':
                x1 = probs[ aj ].unsqueeze(0)
                x2 = past_p(state, aj ,old_model).clone().detach().unsqueeze(0)
            elif args.classifier == 'AM-square':
                x1 = torch.square( probs[ aj ].unsqueeze(0) / past_p(state, aj ,old_model).clone().detach().unsqueeze(0) )  
                x2 = torch.from_numpy( np.array([1]) )
            x1s.append( x1 )
            x2s.append( x2 )
            signedidx = si * ACTION_DIM + aj
            y = sign_Advantages[signedidx]
            # print("y",y)
            y = torch.tensor(y) .unsqueeze(0)
            ys.append( torch.sign(y) )
            yweight.append( torch.abs(y) )
    #first time HPO loss gd
    # fixed_noise = torch.randn(1  )
    old_loss = torch.tensor(100000.0)
    gd_counter = 0
    while True:
        policy_losses = []
        
        # for idx in range(len(returns)):
        for si in range(  env.observation_space.n):
        #     _probs,_value ,_state,action ,_next_state,_log_p = saved_actions[idx]
            # print("state", np.where( _state.detach().numpy() == 1 ),"probs:",_probs, )
            # print("_action",action,"deltaY",deltaYs[idx],)
            s_onehot = np.zeros(env.observation_space.n)
            s_onehot[si] = 1
            # s_onehot = 0
            _state = torch.from_numpy(s_onehot).float()
            probs = model(_state)
            m = Categorical(probs)
            
            ploss = nn.MarginRankingLoss(margin= epsilons[si] )
            
            for aj in range( ACTION_DIM ):
                # action = aj
                action = torch.tensor(aj)
                log_p = m.log_prob(action)
                idx = si*ACTION_DIM + aj
                if args.classifier == 'AM-log':
                    x1 = log_p.unsqueeze(0)
                    x2 = past_log_p(_state, action,old_model).clone().detach().unsqueeze(0)
                elif args.classifier == 'AM':
                    # Classifier 2. pi/mu - 1
                    x1 = probs[action].unsqueeze(0) / (past_p(_state, action,old_model).clone().detach()+torch.tensor(1e-8)).unsqueeze(0)
                    # x1 = probs[action].unsqueeze(0) / past_p(_state, action,old_model).clone().detach().unsqueeze(0)
                    x2 = torch.from_numpy( np.array([1]) )
                elif args.classifier == 'AM-root':
                    # Classifier 2. root(pi/mu) - 1
                    x1 = torch.sqrt( probs[action].unsqueeze(0) / past_p(_state, action,old_model).clone().detach().unsqueeze(0) )
                    x2 = torch.from_numpy( np.array([1]) )
                elif args.classifier == 'AM-sub':
                    x1 = probs[action].unsqueeze(0)
                    x2 = past_p(_state, action,old_model).clone().detach().unsqueeze(0)
                elif args.classifier == 'AM-square':
                    x1 = torch.square( probs[action].unsqueeze(0) / past_p(_state, action,old_model).clone().detach().unsqueeze(0) )  
                    x2 = torch.from_numpy( np.array([1]) )
                # policy_loss = ploss( x1 , x2 , ys[idx]*(1-2*deltaYs[idx].clone().detach()) )
                surr1 = x1 *yweight[idx]
                surr2 = torch.clamp(x1, 1-epsilons[si], 1+epsilons[si]) *yweight[idx]
                unclip_loss = epsilons[si]-ys[idx]*(x1-x2)
                print("unclip_loss.item()",unclip_loss.item())
                # pdb.set_trace()
                '''To avoid the gradient vanishing problem when the prob is close to 100%,
                 I add the args.probUB to stop the update when pi is larger than args.probUB'''
                '''args.sptth is the threshold of the small probability trick '''
                if probs[action].item() <= args.sptth and not (probs[action].item() >= args.probUB and ys[idx]==1 ): 
                    policy_loss = yweight[idx] * ploss( x1 , x2 , ys[idx] )
                    # policy_loss  = -torch.min(surr1, surr2)
                else:
                    # policy_loss = 0*yweight[idx] * ploss( x1 , x2 , ys[idx] )
                    policy_loss = torch.tensor(0)
                    # policy_loss  = -torch.min(surr1, surr2)*0
                if unclip_loss <=0:
                    policy_loss = torch.tensor(0)
                
                if aj == episode % ACTION_DIM:
                    writer.add_scalar("HPG/Loss/a{}_HPOloss".format(aj), policy_loss.item(),  gd_counter)
                # if first_iteration_EMDA_flag==False and true_sign_Advantages[idx] == sign_Advantages[idx]:
                #     first_correct_sign_advantage_losses.append(policy_loss.item())
                #     # first_correct_sign_policy_improving_magnitude+=true_sign_Advantages[idx] *(probs[aj].item()-past_p(_state, action,old_model).item() )
                # elif first_iteration_EMDA_flag==False and true_sign_Advantages[idx] == -1*sign_Advantages[idx]:
                #     first_incorrect_sign_advantage_losses.append(policy_loss.item())
                    # first_incorrect_sign_policy_improving_magnitude+=true_sign_Advantages[idx] *(probs[aj].item()-past_p(_state, action,old_model).item() )
                model_ahead = model.get_head(_state).clone().detach().numpy()
                writer.add_scalar("Reward/a{}_actionheadInwhileLoop".format(aj), model_ahead[aj]  , gd_counter)
                if aj == episode % ACTION_DIM:
                    policy_losses.append(policy_loss)
                if aj == episode % ACTION_DIM:
                    
                    writer.add_scalar("Reward/a{}_probabilityInwhileLoop".format(aj), probs[aj].item() , gd_counter)
                    # writer.add_scalar("Reward/a{}_actionheadInwhileLoop".format(aj), model_ahead[aj]  , gd_counter)
                
        # reset gradients
        # hard_update(model, old_model)
        loss = torch.stack(policy_losses).sum()
        print("loss.item()",loss.item())
        if gd_counter>= args.nepoch or loss.item()==0 :
            # if gd_counter>= args.nepoch or loss.item()==0 or loss.item()==0.9 or loss.item()==-1.1:
            # if abs(old_loss.item()-loss.item()) < args.zeta:
            break
        optimizer.zero_grad()

        
        # perform backprop
        loss.backward()
        optimizer.step()

        first_iteration_EMDA_flag = True 
        # print("loss",loss.item())
        # zeta = 0.00001
        # if old_loss.clone().detach().numpy()-loss.clone().detach().numpy()<zeta:
        gd_counter+=1
        
        
        old_loss = loss
        
    hard_update(model, old_model)
    ploss_log_unit = torch.stack(policy_losses).sum().detach().numpy()
    loss_log_unit = loss.detach().numpy()
    # vloss_log.append(vloss_log_unit)
    ploss_log.append(ploss_log_unit)
    loss_log.append(loss_log_unit)
    writer.add_scalar("HPG/Loss/gradient_descent_counter", gd_counter, episode)
    # writer.add_histogram("HPG/Loss/2step_incorrect_sign_advantage_loss",np.array(second_incorrect_sign_advantage_losses),episode)
    model_ahead = model.get_head(_state).clone().detach().numpy()
    writer.add_scalar("Reward/a0_actionhead", model_ahead[0]  , episode)
    writer.add_scalar("Reward/a1_actionhead", model_ahead[1]  , episode)
    writer.add_scalar("Reward/a2_actionhead", model_ahead[2]  , episode)
    writer.add_scalar("Reward/a3_actionhead", model_ahead[3]  , episode)
    writer.add_scalar("Reward/a4_actionhead", model_ahead[4]  , episode)
     
    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


'''check the l1 norm of the optimal policy vs current policy (100%-prob[a4])'''
def evaluation2(model, episode):
    ret = True
    l1_norm =0.0
    for i in range(env.observation_space.n):
        # [(-1, 0), (0, 1), (1, 0), (0, -1)]
        emptyarr = np.zeros(env.observation_space.n)
        emptyarr[i]=1
        # emptyarr = 0
        state = torch.from_numpy( emptyarr ).float() 
        # state = torch.from_numpy( np.array([i]) ).float().unsqueeze(0) 
        probs   = model(state)
        writer.add_scalar("Reward/a0_probability", probs[0].item() , episode)
        writer.add_scalar("Reward/a1_probability", probs[1].item() , episode)
        writer.add_scalar("Reward/a2_probability", probs[2].item() , episode)
        writer.add_scalar("Reward/a3_probability", probs[3].item() , episode)
        writer.add_scalar("Reward/a4_probability", probs[4].item() , episode)
        probs = probs .detach().numpy()
        
        # dir = np.argmax(probs)
        print(i,"prob :",probs)
        if i == 0:
            l1_norm = 1- probs[4]
            ret = ret
        
    return ret,l1_norm

def main():
    # origin a2c
    print("learning rate",args.learning_rate)
    print("env.action.space",env.action_space)
    running_reward = 0
    running_stepcount = 0
    print("modelloadpath",args.modelloadpath)
    if args.modelloadpath != None:
        model.load_state_dict(torch.load(args.modelloadpath))
    hard_update(model, old_model)
    # run inifinitely many episodes
    # args.max_episode = 2000000
    MAX_STEP = 200
    
    #  hinge
    running_reward = 0
    # past_prob_table = np.full((16 , 4 ),0.25)
    past_prob_table = np.full((env.observation_space.n , env.action_space.n ),0.25)
    # run inifinitely many episodes
    start_point_log = []
    reward_record = []
    hinge_reward_EWMA = []
    i_episode = 0
    hinge_max__episode = args.max_episode if args.classifier != 'a2c' else 0
    for i_episode in range(0,hinge_max__episode):
    # for i_episode in range(0,hinge_max__episode,4):
 
        # pdb.set_trace()
        # perform backprop
        # finish_episode(past_prob_table)
        finish_episode(i_episode,past_prob_table)
        # log results
        eval2ret , l1_norm = evaluation2(model,i_episode)
        if i_episode % args.log_interval == 0:
            # print('hinge Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f} \trunning_(stepcount - beststepcount): {} \t l1 norm: {} '.format(
            #       i_episode, ep_reward, running_reward,running_stepcount,l1_norm))
            # print("last_episode_log",last_episode_log)
            print("l1 norm episode",l1_norm, i_episode)
            # last_episode_log.clear()
            writer.add_scalar("Reward/l1_norm", l1_norm, i_episode)
        # pdb.set_trace()
    if args.modelsavepath:
        torch.save(model.state_dict(), args.modelsavepath)
    


if __name__ == '__main__':
    main()