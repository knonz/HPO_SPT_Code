from pylab import *
import brewer2mpl
import numpy as np
from matplotlib import pyplot as plt
import csv
import os, sys
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d
# Folder = 'NNslt40flippedduration0'
# root_folder ='sptfullsa040advFlip'
# root_folder ='Cartpole200Epoch' #basic Cartpole
# root_folder = 'acrobot200epoch'
# root_folder = 'Mountaincar40epoch' #basic Mountaincar
# root_folder = 'lunarlander4Mfr20' #basic LunarLander-
# root_folder = 'acrobotfr20re' #basic acrobot
# root_folder = 'mountaincarent20'
# root_folder ='sptfullsa040advFlipSAindependentAndMoreEpoch'
# root_folder ='sptfullsa045advFlipSAindependentAndMoreEpoch'
root_folder = 'LunarlanderPPOT' #basic PPO lunarlander
# root_folder = 'AcrobotPPOT' #basic PPO Acrobot
# root_folder = 'CartpolePPOT' #basic PPO Cartpole

# root_folder = 'RnoiseLunarlander' #rnoise HPO lunarlander
# root_folder = 'RnoiseAcrobot' #rnoise HPO Acrobot
# root_folder = 'RnoiseMountaincar' #rnoise HPO Cartpole

# root_folder = 'acrobotfr10sub789'

root_folder_list = [root_folder] #basic 
# root_folder_list = ['lunarlander4Mfr10','lunarlander4Mfr20','lunarlander4Mfr30'] #fr cmp
# root_folder_list = ['acrobotfr10sub789','acrobot200epoch','acrobot30'] #fr cmp dont use
# root_folder_list = ['acrobotfr10sub789','acrobotfr20re','acrobot30'] #fr cmp
# root_folder_list = ['Cartpole200Epoch'] 

# comapare_folder = ["lunarlander_sptthresholdcompare_fr20_epsilon1"]
# root_folder_list = ["lunarlander_sptthresholdcompare_fr20_epsilon1"] #threshold cmp
# root_folder_list = ["acrobot_sptthresholdcompare"] #threshold cmp
# root_folder_list = ["acrobotfr20reepsilon1","acrobotepsilon2","acrobotepsilon3"] #epsilon cmp
# root_folder_list = ["lunarlander4Mfr20epsilon1","sptLunarlander20epsilon2","sptLunarlander20epsilon3"] #epsilon cmp
if root_folder_list[0] == "acrobotfr20reepsilon1" or root_folder_list[0] =="lunarlander4Mfr20epsilon1":
    comapare_folder = ["lunarlander_epsiloncompare"] #epsilon cmp
    # comapare_folder = ["acrobot_epsiloncompare"] #epsilon cmp
gamename = 'LunarLander-v2'
# gamename = 'CartPole-v1'
# gamename = 'Acrobot-v1'
# gamename = 'MountainCar-v0'
font_size = 9
# folder_list = ['vanilla','spt090'] #old dont use
if gamename== 'CartPole-v1':
    # folder_list = ['vanilla','spt90','vanilla2','spt902'] #cartpole
    folder_list = ['vanilla','spt90']
else:
    folder_list = ['vanilla','spt90'] # normal case
# folder_list = ['vanilla','spt90','spt80','spt70'] #threshold cmp
# folder_list = ['vanilla','spt80'] # rnoise case
# 
merge_flag = True
std_flag = True
# std_flag = False
median_flag = False
# median_flag = True
# folder_list = [gamename+'vanilla',gamename+'spt080',gamename+'slt010dY050']
# folder_list = [gamename+'vanilla',gamename+'spt060']
# folder_list = ['NNslt40flippedduration0','NNrc40flippedduration0','NN0flippedduration0']
# folder_list = ['NNsadependent_slt','NNsadependent_rc','NNsadependent_vanilla']
# seed_list = ['289','666','517','789']
# seed_list = ['1','2','3','4','5','6']
# seed_list = ['1','2','3','4','5']
# seed_list = ['1','2','3' ]
# seed_list = ['1','2','3','4']
# seed_list = ['123','196','285','517','789']
# seed_list = ['123','196','285','517' ] # acrobot
# seed_list = ['123','456','789','8565464','16842464' ] #mountaincar
seed_list = ['123','285','789','78949','16842464' ] #lunarlander 4M fr10
# DataLength = [1] * len(seed_list)
def load_csv(file_path, x_scale = 1):
    print("file_path",file_path)
    with open(file_path,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        x = []
        y = []
        is_skipped_scheme = False
        for row in plots:
            if not is_skipped_scheme:
                is_skipped_scheme = True
                continue
            # x.append(int(row[1])//x_scale)
            x.append(int(row[1]) )
            y.append(float(row[2]))
            # if Target == "Lambda" and y[-1] < 0:
            #     y[-1] = 0
        # average_x = []
        # average_y = []
        # for i in range(len(x)):
        #     x_sum = 0.0
        #     y_sum = 0.0
        #     avg = 50 # DMLab
        #     if Folder == "Data":
        #         avg = 2 # ML agent

        #     if i % avg == 0 and i+avg < len(x) and y[i+1] < 2e8:
        #         for idx in range(avg):
        #             x_sum += x[i + idx]
        #             y_sum += y[i + idx]
        #         average_x.append(x_sum / avg)
        #         average_y.append(y_sum / avg)
        #     #if i % 2 == 0 and  i+1 < len(x) and y[i+1] < 10000000: #  for ML-Agents
        #     #    average_x.append((x[i] + x[i+1])/2)
        #     #    average_y.append((y[i] + y[i+1])/2)
        #     #if i % 2 == 0 and  i+1 < len(x) and y[i+1] < 2e8:
        #     #    average_x.append((x[i] + x[i+1])/2)
        #     #    average_y.append((y[i] + y[i+1])/2)
        #     # average_x.append(x[i])
        #     # average_y.append(y[i])
        # return (x, y, average_x, average_y)
        return (x, y)
def refine_data(datas):
    refined_x = []
    refined_y = []
    min_y = []
    max_y = []

    max_len = 0
    for data in datas:
        print("data0",len(data[0]))
        if len(data[0]) > max_len:
            max_len = len(data[0])
            refined_x = data[0]
    for i in range(max_len):
        ys = []
        for data in datas:
            if i < len(data[1]):
                ys.append(data[1][i])
        if median_flag:
            refined_y.append(np.median(ys))
        else:
            refined_y.append(np.mean(ys))
        # min_y.append(np.min(ys))
        # max_y.append(np.max(ys))
        if std_flag:
            min_y.append(np.mean(ys) - np.std(ys))
            max_y.append(np.mean(ys) + np.std(ys))
        else:
            min_y.append(np.min(ys))
            max_y.append(np.max(ys))
        
    return (refined_x, refined_y, min_y, max_y)

bmap = brewer2mpl.get_map('Set1', 'qualitative', 8) # RBGPOY
bmap2 = brewer2mpl.get_map('Set2', 'qualitative', 8)
colors = bmap.mpl_colors

legend_colors = bmap.mpl_colors

human_color = bmap2.mpl_colors[1]

# fig = figure(figsize=(14,9))  # no frame
# fig = figure(figsize=(6,6))  # no frame 1*1
fig = figure(figsize=(4,3))  # no frame 2*2
# fig = figure(figsize=(4,6))  # no frame 2*2
ax = fig.add_subplot(111)


data_sets = []
# for x in range(DataLength[idx]):
for root_folder in root_folder_list:
    for idx in range( len(folder_list) ):
        Folder =  folder_list[idx]
        samples = []
        # autopath = './{0}/{1}/'.format(root_folder ,Folder)
        if root_folder_list[0] == "acrobotfr20reepsilon1" or root_folder_list[0] =="lunarlander4Mfr20epsilon1":
            autopath = './{2}/{0}/{1}/'.format(root_folder ,Folder,comapare_folder[0]) #epsilon cmp
        else:
            autopath = './{0}/{1}/'.format(root_folder ,Folder)
        dirs = os.listdir( autopath )
        for x in dirs:
            if root_folder_list[0] == "acrobotfr20reepsilon1" or root_folder_list[0] =="lunarlander4Mfr20epsilon1":
                samples.append(load_csv("{3}/{0}/{1}/{2}".format(root_folder ,Folder, x,comapare_folder[0] ), x_scale=1)) #epsilon cmp
            else:
                samples.append(load_csv("{0}/{1}/{2}".format(root_folder ,Folder, x ), x_scale=1))
        # for x in range( len(seed_list) ):
            # samples.append(load_csv("{0}/{1}/{2}.csv".format(root_folder ,Folder, seed_list[x] ), x_scale=1))
        print( len(samples) )
        print( len(samples[0]) )
        print( len(samples[0][0]) )
        # Sample = refine_data(samples)
        # data_sets.append(Sample)
        if merge_flag:
            samples = refine_data(samples)
        data_sets.append(samples)
        # data_sets = [Sample]
        print(  len(data_sets) )
        print(  len(data_sets[0]) )
        print(  len(data_sets[0][0]) )

# idx = 0
# print([len(a) for a in data_sets ])
# print("data_sets",data_sets)
# if gamename == 'CartPole-v1':
#     data_sets[0][0].extend( data_sets[2][0] )
#     data_sets[0][1].extend( data_sets[2][1] )
#     data_sets[0][2].extend( data_sets[2][2] )
#     data_sets[0][3].extend( data_sets[2][3] )
#     data_sets[1][0].extend( data_sets[3][0] )
#     data_sets[1][1].extend( data_sets[3][1] )
#     data_sets[1][2].extend( data_sets[3][2] )
#     data_sets[1][3].extend( data_sets[3][3] )
    # ax.set_xticks([ z*200000 for z in range(7)])
    # ax.set_xticklabels(["%.1f"%z+'M' for z in range(7)])
# elif gamename == 'MountainCar-v0':
#     pass
# elif gamename == 'LunarLander-v2':
#     pass
# elif gamename == 'Acrobot-v1':
#     pass
# gamename = 'Acrobot-v1'
# gamename = 'MountainCar-v0'
# for idx in range( len(folder_list) )
# legend_list = [ 'vanilla HPO','HPO with SPT ignoring pi(s,a)>0.9 state action pair' ]
# legend_list = [ 'Vanilla','SPT Th1 = 0.9' ]
# legend_list = [ 'Vanilla','SPT Th1 = 0.8' ]
legend_list = [ 'Vanilla PPO','PPO SPT Th1 = 0.9' ]
# legend_list = [ 'Vanilla','SPT' ]
# legend_list = [ 'Vanilla','SPT Threshold1 = 0.9','SPT Threshold1 = 0.8','SPT Threshold1 = 0.7' ]
# legend_list = [ 'Vanilla HPO','HPO with SPT ignoring pi(s,a)>0.9 state action pair','HPO with SPT ignoring pi(s,a)>0.8 state action pair','HPO with SPT ignoring pi(s,a)>0.7 state action pair' ]
# legend_list = [ 'Vanilla Epsilon=0.1','SPT Epsilon=0.1','Vanilla Epsilon=0.2','SPT Epsilon=0.2','Vanilla Epsilon=0.3','SPT Epsilon=0.3' ]
# 
# legend_list = ['Vanilla FR = 0.1','SPT FR = 0.1','Vanilla FR = 0.2','SPT FR = 0.2','Vanilla FR = 0.3','SPT FR = 0.3']
# legend_list = ['Vanilla Under a Sign-Flipping Probability of 0.1','SPT Under a Sign-Flipping Probability of 0.1','Vanilla  Under a Sign-Flipping Probability of 0.2','SPT Under a Sign-Flipping Probability of 0.2','Vanilla Under a Sign-Flipping Probability of 0.3','SPT Under a Sign-Flipping Probability of 0.3']
# for i in range(len(data_sets)):
from scipy.ndimage.filters import gaussian_filter1d

if gamename == 'CartPole-v1':
    imax = 2
else:
    imax = len(data_sets)
# for i in range(2):#cartpole
# for i in range(len(data_sets)):
for i in range(imax):
    if merge_flag:
        if gamename == 'CartPole-v1':
            ysmoothedmin = gaussian_filter1d(data_sets[i][2], sigma=3)
            ysmoothedmax = gaussian_filter1d(data_sets[i][3], sigma=3)
            # ax.fill_between(data_sets[i][0], data_sets[i][2], data_sets[i][3], alpha=0.25, linewidth=0, color=colors[i%2 ])
            ax.fill_between(data_sets[i][0], ysmoothedmin, ysmoothedmax, alpha=0.25, linewidth=0, color=colors[i%2 ])
            

            ysmoothed = gaussian_filter1d(data_sets[i][1], sigma=3)
            # cubic_interpolation_model = interp1d(data_sets[i][0], data_sets[i][1], kind = "cubic")
            # ax.plot(data_sets[i][0], data_sets[i][1], linewidth=3, color=colors[i%2 ])
            # X_=np.linspace(data_sets[i][0][0], data_sets[i][0][-1], 500)
            # Y_=cubic_interpolation_model(X_)
            ax.plot(data_sets[i][0], ysmoothed, linewidth=1, color=colors[i%2 ],label=legend_list[i%2])
            
        else:
            # ax.fill_between(data_sets[i][0], data_sets[i][2], data_sets[i][3], alpha=0.25, linewidth=0, color=colors[i ])
            # colori = i if i <5 else i+1
            colori = i 
            ysmoothedmin = gaussian_filter1d(data_sets[i][2], sigma=1)
            ysmoothedmax = gaussian_filter1d(data_sets[i][3], sigma=1)
            # ax.fill_between(data_sets[i][0], data_sets[i][2], data_sets[i][3], alpha=0.25, linewidth=0, color=colors[i%2 ])
            ax.fill_between(data_sets[i][0], ysmoothedmin, ysmoothedmax, alpha=0.25, linewidth=0, color=colors[colori])
            ysmoothed = gaussian_filter1d(data_sets[i][1], sigma=1)
            ax.plot(data_sets[i][0], ysmoothed, linewidth=1, color=colors[colori],label=legend_list[i])
            # ax.fill_between(data_sets[i][0], data_sets[i][2], data_sets[i][3], alpha=0.25, linewidth=0, color=colors[ colori ])
            # ax.plot(data_sets[i][0], data_sets[i][1], linewidth=3, color=colors[ colori ])
        # ax.plot(data_sets[i][0], data_sets[i][1], linewidth=3, color=colors[i ])
        print(legend_list[i],"lasty:",ysmoothed[-1],"std",(data_sets[i][3][-1]-data_sets[i][2][-1])/2 )
    else:
        # for x in range( len(seed_list) ):
        for x in range( len(data_sets[i]) ):
            # ax.fill_between(data_sets[i][0], data_sets[i][2], data_sets[i][3], alpha=0.25, linewidth=0, color=colors[i ])
            ax.plot(data_sets[i][x][0], data_sets[i][x][1], linewidth=1, color=colors[i ],label=legend_list[i] if x == 0 else "" )#, zorder = LineOrder[idx]) #2.5
    # ax.plot(data_sets[0][i][2], data_sets[0][i][3], linewidth=5, color=colors[idx])#, zorder = LineOrder[idx]) #2.5
    
    # ax.scatter(data_sets[i][0], data_sets[i][3], s=200, color=colors[idx])
    # ax.scatter(data_sets[0][i][2], data_sets[0][i][3], s=200, color=colors[idx])
# ax.legend(['small loss trick', 'robust classification','vanilla HPO'], fontsize=25)
# ax.legend(['SPT ignore pi(s,a)>80%', 'vanilla SPT HPO'], fontsize=25)
# ax.legend([ 'vanilla WCE HPO','WCE HPO with SPT ignore pi(s,a)>0.6 state action pair'], fontsize=12,loc = 'lower right')
# ax.legend([ 'vanilla WCE HPO','WCE HPO with SPT ignore pi(s,a)>0.8 state action pair'], fontsize=12,loc = 'lower right')
if merge_flag:
    # ax.legend([ 'vanilla CE HPO','CE HPO with SPT ignore pi(s,a)>0.9 state action pair' ], fontsize=12,loc = 'lower right')
    ax.legend( legend_list  ,fontsize=font_size,loc = 'lower right')
else:
    ax.legend(  fontsize=font_size,loc = 'lower right')
# ax.legend([ 'vanilla WCE HPO','WCE HPO with SPT ignore pi(s,a)>0.8 state action pair','WCE HPO with SLT 10% deltaY=0.5'], fontsize=12,loc = 'lower right')
scale = 1
ticks = ticker.FuncFormatter(lambda x, pos: '{0:g} '.format(x*scale))
ax.xaxis.set_major_formatter(ticks)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
# print("data_sets[:][1][-1]",data_sets[0][1][-1])
if gamename == 'CartPole-v1':
    order = [1,0]
else:
    tempdata_sets = np.array(data_sets )
    order = np.argsort(np.array(tempdata_sets[:,1,-1]), axis=0)[::-1]
h, l = ax.get_legend_handles_labels()
print("h",h)
ax.legend(handles=list(np.array(h)[order]),labels=list(np.array(legend_list)[order]),loc = 'lower right',fontsize=font_size)
print("order",order)
if gamename == 'CartPole-v1':
    ax.set_ylim([0, 510]) # for cartpole max rewards
    # ax.set_xticks([ z*200000 for z in range(7)])
    # ax.set_xticklabels(["{:.1f}".format(0.2*z)+'M' for z in range(7)])
    ax.set_xticks([ z*250000 for z in range(7)])
    ax.set_xticklabels(["{:.1f}".format(0.25*z)+'M' for z in range(7)])
elif gamename == 'LunarLander-v2':
    pass
    # ax.set_ylim([-100, 250]) 
    # ax.set_xticks([ z*1000000 for z in range(5)])
    # ax.set_xticklabels(["{:.1f}".format(z)+'M' for z in range(5)])
elif gamename == 'Acrobot-v1':
    # pass
    ax.set_ylim([-200, -80]) 
    # ax.set_xticks([ z*250000 for z in range(7)])
    # ax.set_xticklabels(["{:.1f}".format(0.25*z)+'M' for z in range(7)])
elif gamename == 'MountainCar-v0':
    ax.set_xticks([ z*200000 for z in range(6)])
    ax.set_xticklabels(["{:.1f}".format(0.2*z)+'M' for z in range(6)])

# if gamename == 'CartPole-v1':
#     ax.set_xticks([ z*200000 for z in range(7)])
#     ax.set_xticklabels(["%.1f"%z+'M' for z in range(7)])
# elif gamename == 'MountainCar-v0':
#     pass
# elif gamename == 'LunarLander-v2':
#     pass
# elif gamename == 'Acrobot-v1':
#     pass
# ax.set_xticks([ x*200000 for x in range(7)])
# ax.set_xticklabels([str(x*0.2)+'M' for x in range(7)])
# plt.ylabel('average returns of 100 eval ', fontsize=25)
plt.ylabel('Average Returns', fontsize=font_size)
plt.xlabel('Training Timesteps', fontsize=font_size)
# plt.title('NN policy + uniform flipping of advantage signs', fontsize=font_size)
# plt.title(gamename+' full sa with 0.2 uniform flipping of advantage signs  ', fontsize=font_size)
# plt.title(gamename+' Under a Sign-Flipping Probability of 20%', fontsize=font_size) #basic
plt.title(gamename+' Under Reward Noise With STD=0.5|R|', fontsize=font_size) #rnoise
# plt.title(gamename+' FR 20%', fontsize=font_size)
# plt.tight_layout()
# plt.title(gamename+' with threshold1 = 0.9,0.8,0.7 ', fontsize=font_size)
# plt.title(gamename+' with epsilon = 0.1,0.2,0.3 ', fontsize=font_size)
# plt.title(gamename+' Under a Sign-Flipping Probability of 10%, 20%, and 30%', fontsize=font_size) # fr cmp
# plt.title(gamename+' full sa with 0.45 uniform flipping of advantage signs  ', fontsize=font_size)
# plt.set_size_inches(1400,890)
# 
# plt.savefig('./rewardNoise_{rootfoldername}_{gamename}.png'.format(rootfoldername = root_folder_list[0] ,gamename = gamename ), format='png', dpi=300 , bbox_inches='tight')
# plt.savefig('./ppot_{rootfoldername}_{gamename}.png'.format(rootfoldername = root_folder_list[0] ,gamename = gamename ), format='png', dpi=300 , bbox_inches='tight')
# plt.savefig('./eps_{rootfoldername}_{gamename}.png'.format(rootfoldername = root_folder_list[0] ,gamename = gamename ), format='png', dpi=300 , bbox_inches='tight')
# plt.savefig('./fr_{rootfoldername}_{gamename}.png'.format(rootfoldername = root_folder_list[0] ,gamename = gamename ), format='png', dpi=300 , bbox_inches='tight')
# plt.savefig('./th_{rootfoldername}_{gamename}.png'.format(rootfoldername = root_folder_list[0] ,gamename = gamename ), format='png', dpi=300 , bbox_inches='tight')
# plt.savefig('./basic_{rootfoldername}_{gamename}.png'.format(rootfoldername = root_folder_list[0] ,gamename = gamename ), format='png', dpi=300 , bbox_inches='tight')
# plt.savefig('./full_sa_spt090_ablation3_{rootfoldername}_{gamename}_merge{mergeflag}_median{medianflag}_std{stdFlag}.png'.format(rootfoldername = root_folder_list[0] ,gamename = gamename, mergeflag = merge_flag, medianflag= median_flag, stdFlag= std_flag ), format='png', dpi=300 , bbox_inches='tight')
# plt.savefig('./full_sa_spt090_ablation2_{rootfoldername}_{gamename}_merge{mergeflag}_median{medianflag}_std{stdFlag}.png'.format(rootfoldername = comapare_folder[0] ,gamename = gamename, mergeflag = merge_flag, medianflag= median_flag, stdFlag= std_flag ), format='png' )
# plt.savefig('./full_sa_spt090_200epoch_adv020flip_{gamename}_merge{mergeflag}_median{medianflag}_std{stdFlag}.png'.format(gamename = gamename, mergeflag = merge_flag, medianflag= median_flag, stdFlag= std_flag ), format='png' )
# plt.savefig('./full_sa_spt090_20epoch_adv010flip_{gamename}_merge{mergeflag}_median{medianflag}_std{stdFlag}.png'.format(gamename = gamename, mergeflag = merge_flag, medianflag= median_flag, stdFlag= std_flag ), format='png' )
# plt.savefig('./full_sa_spt090_ablation2_{rootfoldername}_{gamename}_merge{mergeflag}_median{medianflag}_std{stdFlag}.png'.format(rootfoldername = root_folder ,gamename = gamename, mergeflag = merge_flag, medianflag= median_flag, stdFlag= std_flag ), format='png' )
# plt.savefig('./full_sa_spt080_slt010_adv040flip_{gamename}.png'.format(gamename = gamename), format='png' )
# plt.savefig('./full_sa_spt060_adv040flip_moreEpoch_{gamename}.png'.format(gamename = gamename), format='png' )
# plt.savefig('./full_sa_spt060_adv045flip_moreEpoch_{gamename}.png'.format(gamename = gamename), format='png' )
# plt.savefig('./full_sa_spt080_adv040flip_{gamename}.png'.format(gamename = gamename), format='png' )
plt.show()