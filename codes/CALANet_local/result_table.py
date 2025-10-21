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

#from models import HTAggNet
from models_standard import HTAggNet
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
model.load_state_dict(torch.load('HT-AggNet_v2/save/standard/'+dataset+ memo +'.pt'))
model.eval()

#### test
plt.rcParams.update({'font.size':15})


complete = []

#map = plt.get_cmap('gray')

# print(len(eval_queue))
result_summary = pd.DataFrame(None, columns=["id",'actual','predict',"correct"])


for step, (x, y) in enumerate(eval_queue):
    act = int(y.numpy()[0])
    x = Variable(x).cuda().float()
    pred, _ = model(x)
    _, pred = torch.max(pred.cpu().data, 1)
    pred = pred.numpy()[0]
    result_summary.loc[len(result_summary)] = [step, act_map[act], act_map[pred], act==pred]

#print(result_summary)
result_summary.to_csv('HT-AggNet_v2/results/KU-HAR/standard.csv')
