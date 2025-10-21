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

from models_lctm_only import HTAggNet
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import confusion_matrix
from utils import data_info
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#dataset = "UCI_HAR"
#dataset = "DSADS"
#dataset = "OPPORTUNITY"
#dataset = "KU-HAR"
dataset = "PAMAP2"
#dataset = "REALDISP"

memo = "_RDR"
#memo = ""
L = 3

input_nc, segment_size, class_num = data_info(dataset)

_, eval_queue, y_test_unary = input_pipeline(dataset, input_nc, 1)

torch.cuda.set_device(0)
cudnn.benchmark = True 

model = HTAggNet(input_nc, class_num, segment_size, L).cuda()
model.load_state_dict(torch.load('MALANet_break/save/'+dataset+ memo +'.pt'))
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


#results = classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True)
#weighted_avg_f1 = results['weighted avg']['f1-score']

results = classification_report(y_test_unary, predict, digits=4, output_dict=True)
weighted_avg_f1 = results['weighted avg']['f1-score']
print("weighted f1 %f" % (weighted_avg_f1))



#y = model(x)

#print(x.size(), fm1.size())
