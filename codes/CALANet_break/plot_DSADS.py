from matplotlib import pyplot as plt
from matplotlib import ticker as mticker


pro_acc = [0.872, 0.845, 0.860, 0.900, 0.894]
pro_mac = [7.192, 8.493, 8.442, 8.512, 8.740]

res_acc = [0.875, 0.874, 0.876, 0.888]
res_mac = [6.176, 11.287, 19.010, 34.457]


N = [8, 16, 32, 48, 64]



res_acc = [a*100 for a in res_acc]
#res_params = [a*100 for a in res_params] # 10^4

pro_acc = [a*100 for a in pro_acc]
#pro_params = [a*100 for a in pro_params] 

plt.rcParams.update({'font.size':16})
#plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

plt.figure(figsize=[8,5.5])
plt.plot(pro_mac, pro_acc, '-', marker='o', linewidth=2, markersize=12, label='Ours', color= 'tab:red')
plt.plot(res_mac, res_acc, '-', marker='d', linewidth=2, markersize=12, label='Backbone', color= 'tab:blue')
plt.xlabel('FLOPs (millon)')
plt.ylabel('F1-score')
plt.legend(['CALANet', 'CALANet without LCTMs + SLAP'], loc='lower right')
plt.grid(True)
plt.xlim([0,65])
plt.ylim([80,90])
plt.xticks(N)

plt.savefig('MALANET_break/plot/dsads_mac.png',bbox_inches='tight', pad_inches=0)

