import random
import argparse
import numpy as np

# from pylab import *
# import brewer2mpl
# import numpy as np
from matplotlib import pyplot as plt
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=123, help = 'randomseed')
args = parser.parse_args()
random.seed(args.seed)
# probs = [0.556,0.004,0.44]
# probs = [0.64,0.01,0.35]
# probs = [0.05,0.45,0.5]
probs = [0.15,0.45,0.4]
# probs = [0.1,0.4,0.5]
# probs = [0.25,0.25,0.5]
# probs = [0.546,0.044,0.41]
# probs = [0.63,0.005,0.365]
# probs = [0.6,0.2,0.2]
font_size = 10
adv = [ 1,-1,-1]
epsilon = 0.1
flipped_rate = 0.4
episode = 0 
timewindow_len = 5
timewindow = [[0]*timewindow_len for _ in range( len(probs) ) ] 
sa_counter = [0]*3
# rnlist = [2,1,1,2,1,1,1,2,2,2]
# rnlist = [1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1]
# rnlist = [2,2,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1]
# rnlist = [0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,0,0]
# rnlist = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
rnlist = [2]*50+[0]*35+[1,0,0,0]*20
# rnlist+[1]*2)
# timewindow =[ [1,1,0,0,0] ,[0,0,0,1,1],[1,1,0,0,0] ]
# timewindow =[ [1,1,0,0,0] ,[0,0,0,1,1],[0,0,0,0,0] ]
timewindow =[ [0,0,0,1,1] ,[1,1,0,0,0 ],[0,0,0,0,0] ]
Ilist = []
pithreshold1 = 0.5
# fig = plt.figure(figsize=(14,9))  # no frame
fig = plt.figure(figsize=(6,6))  # no frame
ax = fig.add_subplot(111)
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
# [[None]*5 for _ in range(5)]
plotx = []
ploty = []
th1y = []
th2y = []

# t1y = ploty
for idx in range( len( rnlist)):
    rn = rnlist[idx]
    plotx.append(idx)
    rn2 = timewindow[rn][sa_counter[rn] %timewindow_len ]
    # print("sa_counter[rn] %timewindow_len",sa_counter[rn] %timewindow_len)
    sa_counter[rn] = sa_counter[rn]+1
    old_p1 = probs[1]
    # print("rn",rn)
    rnadv = adv[rn]
    if rn2>flipped_rate:
        rnadv = rnadv* -1
    max_prob_index = np.argmax(np.array(probs))
    # if probs[rn]> 1/(epsilon+1) and rnadv==1 : #p
    # if probs[rn]> 0.99: #p
    # if rn == max_prob_index:
    if rn == max_prob_index and probs[rn] > pithreshold1:
        # pass
        optimal_probs_change = 0
    else : #q r
        # rnadv = 1
        optimal_probs_change = probs[0]
        
        for idx in range(len(probs)):
            if idx!= rn:
                probs[idx] = probs[idx] + rnadv*(-1)* probs[idx]*(epsilon*probs[rn]/(1-probs[rn]))
        probs[rn] = (1+ (rnadv)* epsilon)*probs[rn]
        optimal_probs_change = probs[0] - optimal_probs_change
        # else:
            # 
    if rn != 2:
        print("episode",episode,"probs",probs,"rn ",rn,"optiaml probs change",optimal_probs_change  )
        # print("timewindow",timewindow[rn])
        # print("timewindow",timewindow)
        # print("ratio of 0,1",probs[1]/probs[0])
    if rn == 1:
        print("I from probs1",1-(probs[1]-old_p1)/(1-old_p1))
        Ilist.append(1-(probs[1]-old_p1)/(1-old_p1))
    ploty.append(probs[0])
    th1y.append(0.5)
    th2y.append(0.5867768595)
    episode+=1
ax.plot(plotx,ploty,label='probability of the optimal action')
ax.plot(plotx,th1y,'--',label='threshold 1')
ax.plot(plotx,th2y,'--',label='threshold 2')
t1x = [101]*len(plotx)
ax.plot(t1x,ploty,'--',label='t1')
# ax.legend(  fontsize=12,loc = 'lower right')
plt.annotate("threshold 1" , (50, 0.5-0.015),label='threshold 1',fontsize=font_size )
plt.annotate("threshold 2" , (50, 0.5867768595-0.015),label='threshold 2',fontsize=font_size)
plt.annotate("t1" , (101+1, 0.5867768595+0.1),label='t1',fontsize=font_size)
# plt.ylabel('probability', fontsize=25)
# plt.xlabel('episodes ', fontsize=25)
plt.ylabel('Probability of the Optimal Action', fontsize=font_size)
# plt.ylabel('gradient descent counts', fontsize=25)
plt.xlabel('Training Episodes', fontsize=font_size)
# plt.title('the schematic diagram of pi threshold 1 and 2', fontsize=25)
# plt.show()
plt.savefig('./the schematic diagram of pi threshold2.png' ,format='png', dpi=300 )
# Iprod = 1
# for idx in range( len(Ilist)):
#     Iprod = Iprod*Ilist[idx]
# print("episode",episode,"probs",probs,"sa_counter",sa_counter,"Iprod",Iprod)