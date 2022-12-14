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

fig = plt.figure(figsize=(6,6))
np.set_printoptions(threshold=sys.maxsize)
ax = fig.add_subplot(111)
# ax = fig.add_subplot(111, projection='3d')
# x = np.linspace(0, 1, 1000 )
# y = np.linspace(0, 1 , 1000 )
# x = y = np.arange(0.0, 1.0, 0.001)
# xmin, xmax = 0.196,0.204
# ymin, ymax = 0.594,0.606
# xmin, xmax = 0.16,0.24
# ymin, ymax = 0.56,0.64
xmin, xmax = 0.1945,0.202
ymin, ymax = 0.5945,0.602
# xmin, xmax = 0.0,1.0
# ymin, ymax = 0.0,1.0
x = np.arange(xmin, xmax, 0.0001)
y = np.arange(ymin, ymax, 0.0001)
x, y = np.meshgrid(x, y)
# print(x,y,f(x,y))
# print( f(x,y))
# fm = f(x,y)

# print("minimum loss in p0xp1 mesh",np.amin(fm))
# ax.plot(f(x,y))
# ax.plot(x,y,f(x,y))
# surf = ax.plot_surface(x, y, f(x,y), cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# ax.set_xlabel('p[0] probability')
# ax.set_ylabel('p[1] probability')
ax.set_xlabel('p[0]')
ax.set_ylabel('p[1]')
# ax.set_zlabel('loss')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.plot([0.2],[0.6],f([0.2],[0.6]))
# surf = ax.plot_surface(x, y, 1-x-y ,
#                        linewidth=0, antialiased=False)
# plt.show()
# print(np.array([0.201]),np.array([0.599]),f( np.array([0.19]) , np.array([0.599]) ))
# square_side = 1/0.001
# x_min = x.min()
# y_min = y.min()

# def label_point(x, y):
#     # Double forward slash is integer (round down) division
#     # Add 1 here if you really want 1-based indexing
#     x_label = (x - x_min) // square_side

#     y_label = chr(ord('A') + (y - y_min) // square_side)
    
#     return f'{y_label}{x_label}'

# df['label'] = df[['X[mm]', 'Y[mm]']].apply(lambda coord: label_point(*coord), axis=1)

# plt.contourf(X, Y, f(X, Y), 10 , cmap=cm.coolwarm)
C=plt.contour(x,y, f(x,y), levels=30, alpha=1.0,zorder=2,cmap='binary' )
# C=plt.contourf(x,y, f(x,y), levels=30, alpha=1.0,zorder=2  )
plt.clabel(C, inline=True, fontsize=10)
plt.imshow(f(x,y), origin='lower', interpolation='none', extent=[xmin,xmax,ymin,ymax])
scale = 1
ax.scatter([0.2*scale],[0.6*scale  ], marker=".", color='red' ,zorder=2,label="(0.2,0.6,0.2)" )
# ax.scatter([ 0.18*scale],[0.6*scale  ], marker=".", color='orange' ,zorder=2,label="(0.18,0.6,0.22)" )
ax.scatter([ 0.195*scale],[0.6*scale ], marker=".", color='yellow' ,zorder=2,label="(0.195,0.6,0.205)" )
ax.legend()
# plt.xticks(x)
# plt.yticks(y)
plt.colorbar()  
plt.savefig('./color2dRCzoomin.png' ,format='png', dpi=300 )
# plt.show()