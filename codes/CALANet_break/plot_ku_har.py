from matplotlib import pyplot as plt
from matplotlib import ticker as mticker


pro_acc = [0.900, 0.947, 0.927, 0.975, 0.977]
pro_mac = [9.835, 14.099, 24.722, 29.649, 33.793]

res_acc = [0.838, 0.915, 0.934]#, 0.966]
res_mac = [7.377, 19.582, 90.518]#, 180.687]

N = [8, 16, 32, 48, 64, 96]

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
plt.xlim([0,97])
plt.ylim([80,100])
plt.xticks(N)

plt.savefig('MALANET_break/plot/ku_har_mac.png',bbox_inches='tight', pad_inches=0)
plt.savefig('MALANET_break/plot/ku_har_mac.eps',bbox_inches='tight', pad_inches=0)
