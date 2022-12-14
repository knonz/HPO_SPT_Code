import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import sys

old_prob = [0.2,0.6,0.2]
adv = [-1,-1,1]
# advw = [1,1,4]
advw = [1,1,1]
deltaYs =[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
ACTION_DIM = 3
epsilon = 0.1
def f(p0,p1):
    # p = [0.0]*3
    # p[0] = p0
    # p[1] = p1
    # p[2] = 1-p0-p1
    p2 = 1-p0-p1
    # if 1-p0-p1 <0:
    #     return 1
    # print(p0,p1)
    maxloss = 0.0
    for i in range(len(deltaYs)):
        # temploss = []
        # temploss = 0
        temploss = advw[0]*np.maximum(epsilon-adv[0]*(1-2*deltaYs[i][0])* (( p0 /old_prob[0])-1),0 ) +advw[1]*np.maximum(epsilon-adv[1]*(1-2*deltaYs[i][1])* (( p1 /old_prob[1])-1),0 ) +advw[2]*np.maximum(epsilon-adv[2]*(1-2*deltaYs[i][2])* (( p2 /old_prob[2])-1),0 )
        # for a in range(ACTION_DIM ):
        #     temploss += np.maximum(epsilon-adv[a]*deltaYs[i][a]* (( p[a] /old_prob[a])-1),0 )
            # temploss +=  epsilon-adv*deltaYs[i][a]* (( p[a] /old_prob[a])-1) 
        #     # temploss/.append(max(epsilon-adv*deltaYs[i][a]* (( p[a] /old_prob[a])-1),0 ))
        maxloss = np.maximum(temploss, maxloss)
    maxloss = np.where(p0+p1>1,-0.1,maxloss)
    return maxloss

xmin, xmax = 0.18,0.20
# ymin, ymax = 0.5945,0.602
# xmin, xmax = 0.0,1.0
# ymin, ymax = 0.0,1.0
x = np.arange(xmin, xmax, 0.00001)
y = 0.6*np.ones_like(x)
fm = f(x,y)
print("solution",fm)
print("solution set",np.max(fm))
# y = np.arange(ymin, ymax, 0.0001)