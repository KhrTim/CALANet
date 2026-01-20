from matplotlib import pyplot as plt


###ku_har
MALANet = [9.835, 14.099, 30.272, 29.649]
MALANet_RDR = [9.835, 26.894, 116.731, 224.909]
L = [2,3,5,9]

plt.rcParams.update({'font.size':16})
plt.figure(figsize=[8,5.5])
plt.plot(L, MALANet, '-', marker='*', linewidth=2, markersize=12, color= 'tab:red')
plt.plot(L, MALANet_RDR, '-', marker='D', linewidth=2, markersize=12, color= 'tab:blue')
plt.xlabel('Depth of Network')
plt.ylabel('FLOPs (millon)')
plt.legend(['CALANet with LCTMs + SLAP', 'CALANet with LCTMs'], loc='upper left')
plt.grid(True)
plt.xlim([1.8,9.2])
plt.xticks(L)
plt.ylim([0,250])

plt.savefig('HT-AggNet_v2/increase_rate/df_ku_har.png',bbox_inches='tight', pad_inches=0)
plt.savefig('HT-AggNet_v2/increase_rate/df_ku_har.eps',bbox_inches='tight', pad_inches=0)

###pamap2
MALANet = [24.974, 32.447, 65.750, 74.923]
MALANet_RDR = [24.974, 54.474, 230.952, 520.937]
L = [2,3,5,9]

plt.rcParams.update({'font.size':16})
plt.figure(figsize=[8,5.5])
plt.plot(L, MALANet, '-', marker='*', linewidth=2, markersize=12, color= 'tab:red')
plt.plot(L, MALANet_RDR, '-', marker='D', linewidth=2, markersize=12, color= 'tab:blue')
plt.xlabel('Depth of Network')
plt.ylabel('FLOPs (millon)')
plt.legend(['CALANet with LCTMs + SLAP', 'CALANet with LCTMs'], loc='upper left')
plt.grid(True)
plt.xlim([1.8,9.2])
plt.xticks(L)
plt.ylim([0,600])

plt.savefig('HT-AggNet_v2/increase_rate/df_pamap2.png',bbox_inches='tight', pad_inches=0)
plt.savefig('HT-AggNet_v2/increase_rate/df_pamap2.eps',bbox_inches='tight', pad_inches=0)
