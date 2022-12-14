#min_x max_y x*y
from matplotlib import pyplot as plt
import random
import argparse
import numpy as np

font_size = 10
etax = 0.5

etay = 0.5
x_his = []
y_his = []
x = 1
y = 1
episode = 15
x_his.append(x)
y_his.append(y)
for i in range(episode):
    x = x - etax*y
    x_his.append(x)
    y = y + etay*x
    y_his.append(y)
    print(x,y,x*y)
fig = plt.figure(figsize=(7,6))  # no frame
ax = fig.add_subplot(111)
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
# ax.plot(x_his,y_his,label='xy')
plt.scatter(x_his, y_his)
plt.scatter([0], [0])
plt.ylabel('Y-Coordinate', fontsize=font_size)
plt.xlabel('X-Coordinate', fontsize=font_size)
plt.annotate("saddle point(0,0)" , (0, 0+0.1))
plt.annotate("p_0 starting point(1,1)" , (1-0.2, 1+0.1))

def f(a,b):
    return np.dot(a, b)

#建立網格
n = 100
a = np.linspace(-1.5, 1.5, n)
b = np.linspace(-1.5, 1.5, n)
a,b = np.meshgrid(a, b)
z = a*b
# print("z",a,b,z)
C = plt.contourf(a, b, z, levels=15, alpha=.6 )
# ax.legend()
# plt.contourf(  f(a, b), 10, alpha=.6, cmap=plt.cm.jet)
# plt.show()
plt.clabel(C, inline=True, fontsize=10)
for i in range(1,len(x_his)):
    # plt.annotate("p_{}=({:.2f},{:.2f})".format(i,x_his[i],y_his[i]), (x_his[i], y_his[i]+0.05), fontsize=font_size)
    plt.annotate("p_{} ".format(i ), (x_his[i], y_his[i]+0.05), fontsize=font_size)
# plt.show()
plt.savefig('./gda f xy.png' ,format='png', dpi=300 )