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
import sys
import glob
import logging
from sklearn.metrics import classification_report, confusion_matrix

from utils import input_pipeline, count_parameters_in_MB, AvgrageMeter, accuracy, save, data_info
from models import DeepConvLSTM

# Import shared metrics collector
import importlib.util
codes_dir_for_metrics = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("shared_metrics",
                                              os.path.join(codes_dir_for_metrics, 'shared_metrics.py'))
shared_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared_metrics)
MetricsCollector = shared_metrics.MetricsCollector


epoches = 1000
batch_size = 128
seed = 2


#dataset = "UCI_HAR"
#dataset = "WISDM"
dataset = "OPPORTUNITY"
#dataset = "PAMAP2"
#dataset = "UniMiB-SHAR"
#dataset = "DSADS"
#dataset = "KU-HAR"
#dataset = "REALDISP"

input_nc, segment_size, class_num = data_info(dataset)

def weight_init(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Conv1d or type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

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

train_queue, eval_queue, y_test_unary = input_pipeline(dataset, input_nc, batch_size)

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
model = DeepConvLSTM(input_nc, class_num).cuda()
model.apply(weight_init)
print("param size = %fMB" % count_parameters_in_MB(model))

optimizer = torch.optim.Adam(

# Initialize metrics collector
metrics_collector = MetricsCollector(
    model_name='DeepConvLSTM',
    dataset=dataset,
    task_type='HAR',
    save_dir='results'
)
    model.parameters(),
    lr=1e-4,
    betas=(0.9,0.999),
    weight_decay=1e-4,
    eps=1e-08
)
max_f1 = 0
weighted_avg_f1 = 0


# Track training time
with metrics_collector.track_training():
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
            #torch.save(model.state_dict(), 'RNNs/save/'+dataset+'.pt')
            print("epoch %d, loss %e, weighted f1 %f, best_f1 %f" % (epoch+1, eval_loss, weighted_avg_f1, max_f1))
            max_f1 = weighted_avg_f1
            print(classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4))
            if dataset=='UniMiB-SHAR':
                print('ADL:',classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=list(range(9)))['weighted avg']['f1-score'])
                print('Falls:',classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=list(range(9,17)))['weighted avg']['f1-score'])
            elif dataset=='PAMAP2':
                print('ADL:',classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=[0,1,2,3,4,5,6,10,11,12,13,17])['weighted avg']['f1-score'])
                print('Complex:',classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=[7,8,9,14,15,16])['weighted avg']['f1-score'])
        
 

#model.cpu()
#example = torch.rand(1,input_nc,segment_size)
#traced_script_module = torch.jit.trace(model, (example))
#traced_script_module.save('resnet_S2Conv_final/save/'+dataset+'.pt')


# ============================================================================
# COMPREHENSIVE METRICS COLLECTION
# ============================================================================
print("\n" + "="*70)
print("COLLECTING COMPREHENSIVE METRICS")
print("="*70)

# Track inference time
with metrics_collector.track_inference():
    # Re-run inference for timing
    if 'eval_queue' in locals():
        eval_loss, y_pred = infer(eval_queue, model, criterion)
    elif 'test_queue' in locals():
        eval_loss, y_pred = infer(test_queue, model, criterion)
    else:
        y_pred = model(X_test_torch if 'X_test_torch' in locals() else torch.FloatTensor(X_test).to(device))

# Compute throughput
test_samples = len(y_test_unary) if 'y_test_unary' in locals() else (len(test_Y) if 'test_Y' in locals() else (len(y_test) if 'y_test' in locals() else len(eval_data)))
metrics_collector.compute_throughput(test_samples, phase='inference')

# Compute classification metrics
if hasattr(y_pred, 'cpu'):
    y_pred_np = y_pred.cpu().numpy() if hasattr(y_pred, 'cpu') else y_pred
else:
    y_pred_np = y_pred

y_pred_labels = np.argmax(y_pred_np, axis=1) if len(y_pred_np.shape) > 1 else y_pred_np

y_true_labels = y_test_unary if 'y_test_unary' in locals() else (test_Y if 'test_Y' in locals() else y_test)
metrics_collector.compute_classification_metrics(y_true_labels, y_pred_labels)

# Compute model complexity
input_shape = (1, input_nc, segment_size)
if input_shape is not None:
    try:
        metrics_collector.compute_model_complexity(model, input_shape, device=device if 'device' in locals() else 'cuda')
    except Exception as e:
        print(f"Could not compute model complexity: {e}")

# Save comprehensive metrics
metrics_collector.save_metrics()
metrics_collector.print_summary()

