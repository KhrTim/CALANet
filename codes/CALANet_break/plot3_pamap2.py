from matplotlib import pyplot as plt
from matplotlib import ticker as mticker


pro_acc = [0.725, 0.734, 0.765, 0.794, 0.790]
pro_mac = [24.974, 32.447, 65.750, 74.923, 88.162]

res_acc = [0.720, 0.744, 0.743]
res_mac = [20.780, 41.887, 168.006]


N = [16, 32, 48, 64, 96, 128, 192]



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
plt.xlim([15,193])
plt.ylim([70,80])
plt.xticks(N)

plt.savefig('MALANET_break/plot/pamap_mac.png',bbox_inches='tight', pad_inches=0)
plt.savefig('MALANET_break/plot/pamap_mac.eps',bbox_inches='tight', pad_inches=0)

