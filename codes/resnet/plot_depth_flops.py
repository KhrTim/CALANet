from matplotlib import pyplot as plt
import pandas as pd

###opportunity

width = .5 # width of a bar

res_t = pd.DataFrame({
 'FLOPs' : [19.577,40.313,76.728,96.865],
 'F1-score' : [62.6,76.3,80.6,80.9],
 'L' : [2,3,6,9]})

plt.rcParams.update({'font.size':16})
plt.figure(figsize=[8,5.5])



res_t['FLOPs'].plot(kind='bar',width=width, color='tab:blue', ylabel="FLOPs (millon)", legend="FLOPs", title="T-ResNet", ylim=[0,100])
res_t['F1-score'].plot(marker='H', linewidth=2, markersize=12, color='tab:red', ylabel='F1-score (%)', xlabel='Depth of Network', legend='F1-score', secondary_y=True, ylim=[60,85])


ax = plt.gca()
plt.xlim([-width, len(res_t['F1-score'])-width])
#plt.legend(loc='upper left')
ax.set_xticklabels(('2', '3', '6', '9'))

plt.savefig('resnet/plot/fg_res.png',bbox_inches='tight', pad_inches=0)
plt.savefig('resnet/plot/fg_res.eps',bbox_inches='tight', pad_inches=0)


#### HT-AggNet

width = .5 # width of a bar

pro_t = pd.DataFrame({
 'FLOPs' : [21.915,21.961,22.054,22.238],
 'F1-score' : [71.8,79.6,80.4,81.4]})

plt.rcParams.update({'font.size':16})
plt.figure(figsize=[8,5.5])

pro_t['FLOPs'].plot(kind='bar',width=width, color='tab:blue', ylabel="FLOPs (millon)", legend="FLOPs", title="HT-AggNet", ylim=[0,100])
pro_t['F1-score'].plot(marker='H', linewidth=2, markersize=12, color='tab:red', ylabel='F1-score (%)', xlabel='Depth of Network', legend='F1-score', secondary_y=True, ylim=[60,85])


ax = plt.gca()
plt.xlim([-width, len(res_t['F1-score'])-width])
#plt.legend(loc='upper left')
ax.set_xticklabels(('2', '3', '5', '9'))

plt.savefig('resnet/plot/fg_pro.png',bbox_inches='tight', pad_inches=0)
plt.savefig('resnet/plot/fg_pro.eps',bbox_inches='tight', pad_inches=0)

#HT_AggNet = 