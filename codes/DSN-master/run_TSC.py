import warnings
warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import time
import math
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import glob
import logging
from sklearn.metrics import classification_report, confusion_matrix

from aeon.datasets import load_from_arff_file
from utils import To_DataSet
from torch.utils.data import DataLoader

from utils import input_pipeline, count_parameters_in_MB, AvgrageMeter, accuracy, save, data_info
from model.SCNN import SCNN
from sparselearning.core_kernel import Masking, CosineDecay, str2bool
from sklearn.preprocessing import LabelEncoder

# Import shared metrics collector
import importlib.util
codes_dir_for_metrics = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("shared_metrics",
                                              os.path.join(codes_dir_for_metrics, 'shared_metrics.py'))
shared_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared_metrics)
MetricsCollector = shared_metrics.MetricsCollector


epoches = 500
batch_size = 128
seed = 243

# Dataset selection (uncomment the one you want to run)
#dataset = "AtrialFibrillation"
#dataset = "MotorImagery"
#dataset = "Heartbeat"
#dataset = "PhonemeSpectra"
#dataset = "LSST"
dataset = "PEMS-SF"
DATA_PATH = os.path.join('Data', 'TSC', dataset)

train_X, train_Y = load_from_arff_file(os.path.join(DATA_PATH, dataset + "_TRAIN.arff"))
_, train_Y = np.unique(train_Y, return_inverse=True)

test_X, test_Y = load_from_arff_file(os.path.join(DATA_PATH, dataset + "_TEST.arff"))
_, test_Y = np.unique(test_Y, return_inverse=True)

# Infer input_nc, seq_len, and class_num from data
input_nc = train_X.shape[1]
seq_len = train_X.shape[2]
class_num = len(np.unique(train_Y))

train_data = To_DataSet(train_X, train_Y)
test_data = To_DataSet(test_X, test_Y)


train_queue = DataLoader(
        train_data, batch_size=batch_size,shuffle=True,
        pin_memory=True, num_workers=0)
eval_queue = DataLoader(
        test_data, batch_size=batch_size,shuffle=False,
        pin_memory=True, num_workers=0)

y_test_unary = test_Y


def train(train_queue, model,  optimizer, mask=None, weights=None):
    cl_loss = AvgrageMeter()
    cl_acc = AvgrageMeter()
    model.train() # mode change

    for step, (x_train, y_train) in enumerate(train_queue):
        
        n = x_train.size(0)
        x_train = Variable(x_train, requires_grad=False).cuda().float()
        y_train = Variable(y_train, requires_grad=False).cuda().long() # It can handle the compute of model and memory transfer on GPU at the same time.

        optimizer.zero_grad()
        logits = model(x_train)
        loss = F.nll_loss(logits, y_train.long())

        loss.backward() # weight compute
        if mask is not None: mask.step()
        else: optimizer.step()


        prec1 = accuracy(logits.cpu().detach(), y_train.cpu())
        cl_loss.update(loss.data.item(), n)
        cl_acc.update(prec1, n)

    return cl_loss.avg, cl_acc.avg


def infer(eval_queue, model, weights=None):
    cl_loss = AvgrageMeter()
    model.eval()

    preds = []
    with torch.no_grad():
        for step, (x, y) in enumerate(eval_queue):
            x = Variable(x).cuda().float()
            y = Variable(y).cuda().long()

            logits = model(x)
            preds.extend(logits.cpu().numpy())

            n = x.size(0)
            #cl_loss.update(loss.data.item(), n)


    return np.asarray(preds)


if not torch.cuda.is_available():
    print('no gpu device available')
    sys.exit(1)

np.random.seed(seed)
torch.cuda.set_device(0)
torch.manual_seed(seed)
cudnn.benchmark = True # This automatically benchmarks each possible algorithm on your GPU and chooses the fastest one.
torch.cuda.manual_seed(seed)
#logging.info('gpu device = %d' % params['gpu_id'])

## classes weights
classes = np.unique(train_Y)
le = LabelEncoder()
y_ind = le.fit_transform(train_Y.ravel())
recip_freq = len(train_Y)/(len(le.classes_)*np.bincount(y_ind).astype(np.float64))
class_weight = recip_freq[le.transform(classes)]
print('Class weights: ', class_weight)


model = SCNN(input_nc, class_num, nf=47, depth=4, kernel=39, pad_zero=False).cuda()



optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Initialize metrics collector
metrics_collector = MetricsCollector(
    model_name='DSN-master',
    dataset=dataset,
    task_type='TSC',
    save_dir='results'
)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoches, 1e-4, last_epoch=-1)

mask = None
death_rate = 0.5


decay = CosineDecay(death_rate, len(train_queue) * epoches * 0.8)
mask = Masking(optimizer, death_rate=death_rate, death_mode="magnitude", death_rate_decay=decay, growth_mode="random", redistribution_mode="none", train_loader=train_queue)
mask.add_module(model, sparse_init='remain_random', density=0.2)



max_acc = 0
acc = 0


# Track training time
with metrics_collector.track_training():
    for epoch in range(epoches):
    
        # training
        train_loss, train_acc = train(train_queue, model, optimizer, mask, weights=class_weight)

        if epoch >= epoches * 0.8:
           mask.death_decay_update(decay_flag=False)

        # evaluating
        y_pred = infer(eval_queue, model)
        results = classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True)
        acc = results['accuracy']
        if (epoch+1) % 50 == 0:
            print('training... ', epoch+1)
        if max_acc < acc:
            #if weighted_avg_f1 < 0.88:
            #    torch.save(model.state_dict(), 'DSN-master/save/'+dataset + '.pt')
            print("epoch %d, Acc %f, best_Acc %f" % (epoch+1,  acc, max_acc))
            max_acc = acc
            print(classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4))

print("epoch %d, Acc %f, best_Acc %f" % (epoch+1,  acc, max_acc))

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
        y_pred = infer(eval_queue, model)
    elif 'test_queue' in locals():
        y_pred = infer(test_queue, model)
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
input_shape = (1, input_nc, seq_len)
if input_shape is not None:
    try:
        metrics_collector.compute_model_complexity(model, input_shape, device=device if 'device' in locals() else 'cuda')
    except Exception as e:
        print(f"Could not compute model complexity: {e}")

# Save comprehensive metrics
metrics_collector.save_metrics()
metrics_collector.print_summary()

