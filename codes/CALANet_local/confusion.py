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
import seaborn as sn

from models import HTAggNet
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import confusion_matrix
from utils import data_info
from sklearn.metrics import accuracy_score

dataset = "UCI_HAR"
#dataset = "UniMiB-SHAR"
#dataset = "DSADS"
#dataset = "OPPORTUNITY"
#dataset = "KU-HAR"
#dataset = "PAMAP2"
#dataset = "REALDISP"

memo = ""
#memo = ""
L = 8

input_nc, segment_size, class_num = data_info(dataset)

act_map = None
if dataset == "UCI_HAR":
    act_map = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Lying'] # UCI_HAR
elif dataset == "WISDM":
    act_map = ['Jogging', 'Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
elif dataset == "PAMAP2":
    act_map = ['A' + str(i) for i in range(1,19)]
    # act_map = [ 'Lying','Sitting','Standing','Walking','Running','Cycling','Nordic walking', #PAMAP2
    #         'Watching TV', 'Computer work', 'Car driving', 'Ascending stairs',
    #         'Descending stairs','Vacuum cleaning','Ironing', 'Folding laundry', 
    #         'House cleaning', 'Playing soccer', 'Rope jumping']
elif dataset == "OPPORTUNITY":
    act_map = ['A' + str(i) for i in range(1,18)]
    # act_map = ['Open Door 1', 'Open Door 2', 'Close Door 1',  #OPPORTUNITY
    #         'Close Door 2', 'Open Fridge', 'Close Fridge', 'Open Dishwasher',
    #         'Close Dishwasher', 'Open Drawer 1', 'Close Drawer 1', 'Open Drawer 2',
    #         'Close Drawer 2', 'Open Drawer 3', 'Close Drawer 3', 'Clean Table',
    #         'Drink from Cup', 'Toggle Switch']
elif dataset == "UniMiB-SHAR":
    act_map = ['StandingUpFS', 'StandingUpFL', 'Walking', 'Running',
            'GoingUpS', 'Jumping', 'GoingDownS', 'LyingDownFS','SittingDown',
            'FallingForw', 'FallingRight', 'FallingBack', 'HittingObstacle',
            'FallingWithPS', 'FallingBackSC','Syncope', 'FallingLeft']
elif dataset == "DSADS":
    act_map = ['A' + str(i) for i in range(1,20)]
    # act_map = ['Sitting', 'Standing', 'LyingBack', 'LyingRight',
    #         'Upstairs', 'Downstairs', 'StandingElevator', 'MovingElevator','Walking',
    #         'WalkingFlatTreadmill', 'WalkingInclinedTreadmill', 'RunningTreadmill', 'ExercisingStepper', 'ExercisingCross-trainer',
    #         'CyclingHorizontal', 'CyclingVertical','Rowing', 'Jumping', 'PlayingBasketball']
elif dataset == "REALDISP":
    act_map = ['A' + str(i) for i in range(1,34)]
    # act_map = ['Walking', 'Jogging', 'Running', 'Jump-up', 'Jump-front-back', 'Jump-sideways',
    #         'Jump-leg/arms-open/closed', 'Jump-rope', 'Trunk-twist-arms', 'Trunk-twist-elbows','Waist-bends-forward',
    #         'Waist-rotation', 'Waist-bends', 'Reach-heels-backwards', 'Lateral-bend',
    #         'Lateral-bend-arm-up', 'Repetitive-forward-stretching', 'Upper-trunk-and-lower-body-opposite-twist',
    #         'Arms-lateral-elevation', 'Arms-frontal-elevation','Frontal-hand-claps', 'Arms-frontal-crossing', 'Shoulders-high-amplitude-rotation',
    #         'Shoulders-low-amplitude-rotation', 'Arms-inner-rotation', 'Knees-alternatively-breast', 'Heels-alternatively-backside',
    #         'Knees-bending-crouching', 'Knees-alternatively-bend-forward', 'Rotation-on-the-knees', 'Rowing', 'Elliptical-bike', 'Cycling'
    #         ]
elif dataset == "KU-HAR":
    act_map = ['A' + str(i) for i in range(1,19)]
    # act_map = [
    #     'Stand', 'Sit', 'Talk-sit', 'Talk-stand', 'Stand-sit', 'Lay', 'Lay-stand', "Pick", 'Jump', 'Push-up', 'Sit-up', 'Walk',
    #     'Walk-backward', 'Walk-circle', 'Run', 'Stair-up', 'Stair-down', 'Table-tennis'
    # ]

_, eval_queue, y_test_unary = input_pipeline(dataset, input_nc, 1)

torch.cuda.set_device(0)
cudnn.benchmark = True 

model = HTAggNet(input_nc, class_num, segment_size, L).cuda()
model.load_state_dict(torch.load('CALANet_local/save/'+dataset+ memo +'.pt'))
model.eval()

#### test
plt.rcParams.update({'font.size':15})


complete = []

#map = plt.get_cmap('gray')

actual=[]
predict = []

for step, (x, y) in enumerate(eval_queue):
    act = y.numpy()[0]
    actual.append(act)
    x = Variable(x).cuda().float()
    pred = model(x)
    _, pred = torch.max(pred.cpu().data, 1)
    predict.append(pred.numpy()[0])


#classes = [act_map[i] for i in range(class_num-1)]
f = confusion_matrix(actual, predict)

df_cm = pd.DataFrame(f, index = [act_map[i] for i in range(class_num)],
                  columns = [act_map[i] for i in range(class_num)])

fig = plt.figure(figsize=[10,7])
#fig = plt.figure(figsize=[30,20])
sn.heatmap(df_cm, annot=True, fmt="d", cmap="RdPu")
plt.ylabel('Actual', fontsize=21)
plt.xlabel('Predicted', fontsize=21)
plt.xticks(rotation=45)
plt.yticks(rotation=45)

fig.subplots_adjust(left=0.3,
                bottom=0.43, 
                right=0.99, 
                top=0.98,)

plt.savefig('CALANet_local/confMatrix/cm_'+dataset.lower()+memo+'.png', bbox_inches='tight')
plt.savefig('CALANet_local/confMatrix/cm_'+dataset.lower()+memo+'.eps', bbox_inches='tight')



#y = model(x)

#print(x.size(), fm1.size())
