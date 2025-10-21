from matplotlib import pyplot as plt
from matplotlib import ticker as mticker


pro_acc = [0.728, 0.760, 0.780, 0.783, 0.775]
pro_mac = [4.691, 6.911, 8.390, 8.778, 9.440]

res_acc = [0.687, 0.695, 0.787]
res_mac = [3.446, 9.715, 25.358]


N = [4, 8, 16, 32]



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
plt.xlim([0,33])
plt.ylim([60,80])
plt.xticks(N)

plt.savefig('MALANET_break/plot/unimib_mac.png',bbox_inches='tight', pad_inches=0)
plt.savefig('MALANET_break/plot/unimib_mac.eps',bbox_inches='tight', pad_inches=0)

