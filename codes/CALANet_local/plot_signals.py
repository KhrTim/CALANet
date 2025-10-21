import warnings
warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from utils import input_pipeline
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os

from models import HTAggNet
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import classification_report
from utils import data_info
from sklearn.metrics import accuracy_score

#dataset = "UCI_HAR"
#dataset = "UniMiB-SHAR"
#dataset = "PAMAP2"
#dataset = "DSADS"
dataset = "KU-HAR"
#dataset = "REALDISP"

#memo = "_C2"
memo = ""
L = 8

input_nc, segment_size, class_num = data_info(dataset)

act_map = [
        'Stand', 'Sit', 'Talk-sit', 'Talk-stand', 'Stand-sit', 'Lay', 'Lay-stand', "Pick", 'Jump', 'Push-up', 'Sit-up', 'Walk',
        'Walk-backward', 'Walk-circle', 'Run', 'Stair-up', 'Stair-down', 'Table-tennis'
    ]

_, eval_queue, y_test_unary = input_pipeline(dataset, input_nc, 1)

torch.cuda.set_device(0)
cudnn.benchmark = True 

model = HTAggNet(input_nc, class_num, segment_size, L).cuda()
model.load_state_dict(torch.load('HT-AggNet_v2/save/'+dataset+ memo +'.pt'))
model.eval()

#### test
plt.rcParams.update({'font.size':15})


complete = []

#map = plt.get_cmap('gray')

proposed = pd.read_csv("HT-AggNet_v2/results/KU-HAR/proposed.csv").values
standard = pd.read_csv("HT-AggNet_v2/results/KU-HAR/standard.csv").values

idx_list = []
for i in range(proposed.shape[0]):
    # print(proposed[i,4])
    # break
    if (proposed[i,4] == True) and (standard[i,4]==False):
        idx_list.append(i)

print(len(idx_list))

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


for step, (x, y) in enumerate(eval_queue):
    # if step not in idx_list:
    #     continue

    # if step in [25, 57, 94]:
    #     continue
    
    act = int(y.numpy()[0])
    print("Activity:", act_map[act])
    print("index:", step)

    if act != 16:
        continue
    
    createDirectory("HT-AggNet_v2/fm_plot/proposed/"+act_map[act])
    

    # if len(os.listdir("HT-AggNet_v2/fm_plot/standard/"+act_map[act])) >= 2:
    #     continue


    createDirectory("HT-AggNet_v2/fm_plot/proposed/"+act_map[act] + "/" + str(step))
    createDirectory("HT-AggNet_v2/fm_plot/proposed/"+act_map[act] + "/" + str(step) + "/Layer_0")
    for i in range(x.size()[1]):
        plt.figure(figsize=[8,5.5])
        t = [i for i in range(x.size()[2])]
        plt.plot(t, x[0][i], '-', linewidth=2, color= 'tab:red')
        plt.savefig("HT-AggNet_v2/fm_plot/proposed/"+act_map[act] + "/" + str(step) + "/Layer_0/" + str(i) + ".png",bbox_inches='tight', pad_inches=0)
        plt.close()

    x = Variable(x).cuda().float()
    _, fms = model(x)

    for l in range(len(fms)):
        createDirectory("HT-AggNet_v2/fm_plot/proposed/"+act_map[act] + "/" + str(step) + "/Layer_"+str(l+1))
        fm = fms[l].cpu().detach().numpy()

        for i in range(fm.shape[1]):
            plt.figure(figsize=[8,5.5])
            t = [i for i in range(fm.shape[2])]
            plt.plot(t, fm[0][i], '-', linewidth=2, color= 'tab:red')
            plt.savefig("HT-AggNet_v2/fm_plot/proposed/"+act_map[act] + "/" + str(step) + "/Layer_"+str(l+1) + "/" + str(i) + ".png",bbox_inches='tight', pad_inches=0)
            plt.close()


