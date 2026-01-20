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

from utils import input_pipeline, count_parameters_in_MB, AvgrageMeter, accuracy, save, data_info, To_DataSet
from models_gts import HTAggNet

from sklearn.model_selection import KFold
import pandas as pd
from torch.utils.data import DataLoader
from sklearn import preprocessing

epoches = 500
batch_size = 128
seed = 243 #143
L = 8
k_folds = 5

kf = KFold(n_splits=k_folds, shuffle=True)

parent_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
data_path = os.path.join(parent_path, 'Data','KU_HAR')

seg_size = 300 # 3s 

# data preprocessing
fn = os.path.join(data_path,'3.Time_domain_subsamples', 'KU-HAR_time_domain_subsamples_20750x300.csv')
data=pd.read_csv(fn, header=None) 
data = data.values

X = data[:,:-3]
Y = data[:,-3]

indexes = [
    6587,
    6588,
    6589,
    6590,
    6591,
    6592,
    6593,
    6594,
    6595,
    6596,
    6597,
    6598,
    6599,
    6600,
    6601,
    6602,
    6603,
    6604,
    6605,
    6606,
    6607,
    6660,
    6661,
    6662,
    6663,
    6664,
    6665,
    6666,
    6667,
    6668,
    6669,
    6670,
    6671,
    6672,
    6673,
    6674,
    6675,
    6676,
    6677,
    6678,
    6679,
    6680,
    6681,
    6682,
    6683,
    6684,
    6685,
    6686,
    6687,
    6716,
    6717,
    6718,
    6719,
    6720,
    6721,
    6722,
    6723,
    6724,
    6725,
    6726,
    6727,
    6728,
    6729,
    6730,
    6731,
    6732,
    6733,
    6734,
    6735,
    6736,
    6737,
    6738,
    6739,
    6740,
    6741,
    6742,
    6743,
    6750,
    6751,
    6752,
    6753,
    6754,
    6755,
    6756,
    6757,
    6758,
    6759,
    6760,
    6761,
    6762,
    6763,
    6764,
    6765,
    6766,
    6767,
]

# delete the bad samples
X = np.delete(X, indexes, 0)
Y = np.delete(Y, indexes, 0)

X = X.reshape([X.shape[0],6,300])

train_dataset = To_DataSet(X, Y)

#all_indices = list(range(X.shape[0]))
#train_ind, test_ind = train_test_split(all_indices, test_size=0.2)


#memo = "_D" + str(L)
memo=""

input_nc, segment_size, class_num = data_info("KU-HAR")



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

if not torch.cuda.is_available():
    print('no gpu device available')
    sys.exit(1)

np.random.seed(seed)
torch.cuda.set_device(0)
torch.manual_seed(seed)
cudnn.benchmark = True # This automatically benchmarks each possible algorithm on your GPU and chooses the fastest one.
torch.cuda.manual_seed(seed)
#logging.info('gpu device = %d' % params['gpu_id'])


max_f1 = 0
weighted_avg_f1 = 0

for fold, (train_idx, test_idx) in enumerate(kf.split(train_dataset)):
    print(f"Fold {fold + 1}")
    print("-------")

    train_X = np.asarray(X[train_idx], dtype=np.float32)
    train_Y = np.asarray(Y[train_idx], dtype=np.float32)
    test_X = np.asarray(X[test_idx], dtype=np.float32)
    test_Y = np.asarray(Y[test_idx], dtype=np.float32)

    y_test_unary = test_Y

    ## normalizaiton
    for i in range(6):
        scalers = preprocessing.StandardScaler()
        train_X[:,i,:] = scalers.fit_transform(train_X[:,i,:])
        test_X[:,i,:] = scalers.transform(test_X[:,i,:])

    train_data = To_DataSet(train_X, train_Y)
    eval_data = To_DataSet(test_X, test_Y)

    train_queue = DataLoader(
        train_data, batch_size=batch_size,shuffle=True,
        pin_memory=True, num_workers=0)
    eval_queue = DataLoader(
        eval_data, batch_size=batch_size,shuffle=False,
        pin_memory=True, num_workers=0)

    criterion = nn.CrossEntropyLoss().cuda()
    model = HTAggNet(input_nc, class_num, segment_size, L).cuda()
    model.apply(weight_init)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,
        betas=(0.9,0.999),
        weight_decay=5e-4,
        eps=1e-08
    )
    

    for epoch in range(epoches):
        
        # training
        train_loss, train_acc = train(train_queue, model, criterion, optimizer)

        # evaluating
        eval_loss, y_pred = infer(eval_queue, model, criterion)
        results = classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True)
        weighted_avg_f1 = results['weighted avg']['f1-score']
        #if (epoch+1) % 50 == 0:
        #    print('training... ', epoch+1)
        if max_f1 < weighted_avg_f1:
            torch.save(model.state_dict(), 'HT-AggNet_v2/save/kfold_ku_har.pt')
            #print("epoch %d, loss %e, weighted f1 %f, best_f1 %f" % (epoch+1, eval_loss, weighted_avg_f1, max_f1))
            max_f1 = weighted_avg_f1
            #print(classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4))
        # if dataset=='UniMiB-SHAR':
        #     print('ADL:',classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=list(range(9)))['weighted avg']['f1-score'])
        #     print('Falls:',classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=list(range(9,17)))['weighted avg']['f1-score'])
        # elif dataset=='PAMAP2':
        #     print('ADL:',classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=[0,1,2,3,4,5,6,10,11,12,13,17])['weighted avg']['f1-score'])
        #     print('Complex:',classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=[7,8,9,14,15,16])['weighted avg']['f1-score'])
        
    print("epoch %d, loss %e, weighted f1 %f, best_f1 %f" % (epoch+1, eval_loss, weighted_avg_f1, max_f1))
    print(classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4))

#model.cpu()
#example = torch.rand(1,input_nc,segment_size)
#traced_script_module = torch.jit.trace(model, (example))
#traced_script_module.save('resnet_S2Conv_final/save/'+dataset+'.pt')