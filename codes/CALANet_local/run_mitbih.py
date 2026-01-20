import warnings
warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import time
import math
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import glob
import logging
from sklearn.metrics import classification_report, confusion_matrix

from utils import input_pipeline, count_parameters_in_MB, AvgrageMeter, accuracy, data_info
from models_gts import HTAggNet
import pandas as pd
from torch.utils.data import Dataset, DataLoader

epoches = 500
batch_size = 128
seed = 243 #143
L = 8

train_path = "Data/MIT_BIH/mitbih_train.csv"
test_path = "Data/MIT_BIH/mitbih_test.csv"

df_train = pd.read_csv(train_path, header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv(test_path, header=None)

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis].transpose(0,2,1)



Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis].transpose(0,2,1)
y_test_unary = Y_test



input_nc = 1 
segment_size = 187
class_num = 5

class To_DataSet(Dataset):
    def __init__(self, X, Y):
        self.data_num = Y.shape[0]
        self.x = torch.as_tensor(X)
        self.y = torch.as_tensor(Y)#torch.max(torch.as_tensor(Y), 1)[1]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.data_num


train_queue = DataLoader(
    To_DataSet(X, Y), batch_size=batch_size,shuffle=True,
    pin_memory=True, num_workers=0)
eval_queue = DataLoader(
    To_DataSet(X_test, Y_test), batch_size=batch_size,shuffle=False,
    pin_memory=True, num_workers=0)


def weight_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(
            m.weight, std=0.01)
        #nn.init.kaiming_normal(m.weight, mode='fan_out')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant(m.weight,1)
        nn.init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant(m.bias, 0)

def train(train_queue, model, criterion, optimizer):
    cl_loss = AvgrageMeter()
    cl_acc = AvgrageMeter()
    model.train() # mode change

    for step, (x_train, y_train) in enumerate(train_queue):
        
        n = x_train.size(0)
        x_train = Variable(x_train, requires_grad=False).cuda().float()
        y_train = Variable(y_train, requires_grad=False).cuda().long() # It can handle the compute of model and memory transfer on GPU at the same time.

        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)

        loss.backward() # weight compute
        optimizer.step() # weight update


        prec1 = accuracy(logits.cpu().detach(), y_train.cpu())
        cl_loss.update(loss.data.item(), n)
        cl_acc.update(prec1, n)

    return cl_loss.avg, cl_acc.avg

def infer(eval_queue, model, criterion):
    cl_loss = AvgrageMeter()
    model.eval()

    preds = []
    with torch.no_grad():
        for step, (x, y) in enumerate(eval_queue):
            x = Variable(x).cuda().float()
            y = Variable(y).cuda().long()

            logits = model(x)
            loss = criterion(logits, y)
            preds.extend(logits.cpu().numpy())

            n = x.size(0)
            cl_loss.update(loss.data.item(), n)


    return cl_loss.avg, np.asarray(preds)

#train_queue, eval_queue, y_test_unary = input_pipeline(dataset, input_nc, batch_size)

if not torch.cuda.is_available():
    print('no gpu device available')
    sys.exit(1)

np.random.seed(seed)
torch.cuda.set_device(0)
torch.manual_seed(seed)
cudnn.benchmark = True # This automatically benchmarks each possible algorithm on your GPU and chooses the fastest one.
torch.cuda.manual_seed(seed)
#logging.info('gpu device = %d' % params['gpu_id'])

criterion = nn.CrossEntropyLoss().cuda()
model = HTAggNet(input_nc, class_num, segment_size, L).cuda()
model.apply(weight_init)
print("param size = %fMB" % count_parameters_in_MB(model))

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-4,
    betas=(0.9,0.999),
    weight_decay=5e-4,
    eps=1e-08
)
max_f1 = 0
weighted_avg_f1 = 0

for epoch in range(epoches):
    
    # training
    train_loss, train_acc = train(train_queue, model, criterion, optimizer)

    # evaluating
    eval_loss, y_pred = infer(eval_queue, model, criterion)
    results = classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True)
    weighted_avg_f1 = results['weighted avg']['f1-score']
    if (epoch+1) % 50 == 0:
        print('training... ', epoch+1)
    if max_f1 < weighted_avg_f1:
        torch.save(model.state_dict(), 'HT-AggNet_v2/save/with_gts/ecg.pt')
        print("epoch %d, loss %e, weighted f1 %f, best_f1 %f" % (epoch+1, eval_loss, weighted_avg_f1, max_f1))
        max_f1 = weighted_avg_f1
        print(classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4))
 

#model.cpu()
#example = torch.rand(1,input_nc,segment_size)
#traced_script_module = torch.jit.trace(model, (example))
#traced_script_module.save('resnet_S2Conv_final/save/'+dataset+'.pt')