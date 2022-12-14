import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import sys
import ternary

old_prob = [0.2,0.6,0.2]
adv = [-1,-1,1]
# advw = [1,1,4]
advw = [1,1,1]
deltaYs =[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
ACTION_DIM = 3
epsilon = 0.1
# def f(p0,p1,p2):
def f(p):
    # print("p:",p)
    # p = p/scale
    # p = [0.0]*3
    # p[0] = p0
    # p[1] = p1
    # p[2] = 1-p0-p1
    # p2 = 1-p0-p1
    # if 1-p0-p1 <0:
    #     return 1
    # print(p0,p1)
    maxloss = 0.0
    for i in range(len(deltaYs)):
        # temploss = []
        # temploss = 0
        temploss = advw[0]*np.maximum(epsilon-adv[0]*(1-2*deltaYs[i][0])* (( p[0] /old_prob[0])-1),0 ) +advw[1]*np.maximum(epsilon-adv[1]*(1-2*deltaYs[i][1])* (( p[1] /old_prob[1])-1),0 ) +advw[2]*np.maximum(epsilon-adv[2]*(1-2*deltaYs[i][2])* (( p[2] /old_prob[2])-1),0 )
        # temploss = advw[0]*np.maximum(epsilon-adv[0]*(1-2*deltaYs[i][0])* (( p0 /old_prob[0])-1),0 ) +advw[1]*np.maximum(epsilon-adv[1]*(1-2*deltaYs[i][1])* (( p1 /old_prob[1])-1),0 ) +advw[2]*np.maximum(epsilon-adv[2]*(1-2*deltaYs[i][2])* (( p2 /old_prob[2])-1),0 )
        # for a in range(ACTION_DIM ):
        #     temploss += np.maximum(epsilon-adv[a]*deltaYs[i][a]* (( p[a] /old_prob[a])-1),0 )
            # temploss +=  epsilon-adv*deltaYs[i][a]* (( p[a] /old_prob[a])-1) 
        #     # temploss/.append(max(epsilon-adv*deltaYs[i][a]* (( p[a] /old_prob[a])-1),0 ))
        maxloss = np.maximum(temploss, maxloss)
    # maxloss = np.where(p0+p1>1,-0.1,maxloss)
    # return (p0,p1,p2,maxloss)
    return maxloss

# def generate_heatmap_data(scale=5):
#     from ternary.helpers import simplex_iterator
#     d = dict()
#     for (i, j, k) in simplex_iterator(scale):
#         d[(i , j , k )] = f(i, j, k)
#         # d[(i , j , k )] = f(i/scale, j/scale, k/scale)
#     return d


# scale = 80
# data = generate_heatmap_data(scale)
# figure, tax = ternary.figure(scale=scale)
# tax.heatmap(data, style="hexagonal", use_rgba=False, colorbar=True)
# tax.boundary()
# tax.set_title("RGBA Heatmap")
# plt.show()
font_size = 10
scale = 200
figure, tax = ternary.figure(scale=scale)
figure.set_size_inches(7.5, 6)
tax.scatter([(0.2*scale,0.6*scale,0.2*scale) ], marker=".", color='red' ,zorder=2,label="(0.2,0.6,0.2)" )
tax.scatter([ (0.18*scale,0.6*scale,0.22*scale) ], marker=".", color='orange' ,zorder=2,label="(0.18,0.6,0.22)" )
tax.scatter([ (0.195*scale,0.6*scale,0.205*scale)], marker=".", color='yellow' ,zorder=2,label="(0.195,0.6,0.205)" )
# tax.scatter([(0.2*scale,0.6*scale,0.2*scale),(0.18*scale,0.6*scale,0.22*scale),(0.195*scale,0.6*scale,0.205*scale)], marker=".", color='red' ,zorder=2 )
tax.heatmapf(f, boundary=True, style="triangular")
tax.boundary(linewidth=2.0)
tax.legend(loc='upper left')
# tax.annotate("(0.2,0.6,0.2)" , (0.17*scale,0.6*scale ))
# tax.annotate("(0.18,0.6,0.22)" , (0.18*scale,0.63*scale ))
# tax.annotate("(0.195,0.6,0.205)" , (0.195*scale,0.57*scale))
# tax.set_title("Shannon Entropy Heatmap")
# tax.ticks(axis='lbr',scale=1.0, linewidth=1, multiple=10),locations=np.arange(0, 1.0, 0.1)
# tax.ticks( locations=np.arange(0, 1.0, 0.1), linewidth=1, multiple=10,tick_formats="%.3f")
a = list(range(11))
# b = 

ticks_string =[str(x/10) for x in a]
tax.ticks(   locations=np.arange(0, scale, scale/10).tolist(),fontsize = font_size, ticks=ticks_string ,linewidth=1, multiple=scale/10 )
# tax.ticks(   locations=np.arange(0, scale, scale/10).tolist(), ticks=["0","1","2","3","4","5", "6", "7", "8", "9", "10"] ,linewidth=1, multiple=scale/10,tick_formats="%.3f")
tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')
tax.bottom_axis_label(label='p[0]',fontsize = font_size)
tax.left_axis_label(label='p[2]',fontsize = font_size)
tax.right_axis_label(label='p[1]',fontsize = font_size)

# ax = tax.get_axes()
# ax.set_xlabel('p[0] probability')
# ax.set_ylabel('p[1] probability')
# tax.show()
tax.savefig("color2dRC3.png", format='png', dpi=300)
# fig = plt.figure(figsize=(7.5,3.5))
# np.set_printoptions(threshold=sys.maxsize)
# ax = fig.add_subplot(111)
# ax = fig.add_subplot(111, projection='3d')
# x = np.linspace(0, 1, 1000 )
# y = np.linspace(0, 1 , 1000 )
# x = y = np.arange(0.0, 1.0, 0.001)
# xmin, xmax = 0.196,0.204
# ymin, ymax = 0.594,0.606
# x = np.arange(xmin, xmax, 0.0001)
# y = np.arange(ymin, ymax, 0.0001)
# x, y = np.meshgrid(x, y)
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
# plt.imshow(f(x,y), origin='lower', interpolation='none', extent=[xmin,xmax,ymin,ymax])
# plt.xticks(x)
# plt.yticks(y)
# plt.colorbar()  
# plt.show()