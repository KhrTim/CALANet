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

from utils import input_pipeline, count_parameters_in_MB, AvgrageMeter, accuracy, save, data_info
from model.millet_model import MILLETModel
from model import backbone, pooling
from util import get_gpu_device_for_os

# Import shared metrics collector
import importlib.util
codes_dir_for_metrics = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("shared_metrics",
                                              os.path.join(codes_dir_for_metrics, 'shared_metrics.py'))
shared_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared_metrics)
MetricsCollector = shared_metrics.MetricsCollector


epoches = 200
batch_size = 128
seed = 4



#dataset = "UCI_HAR"
#dataset = "UniMiB-SHAR"
#dataset = "DSADS"
#dataset = "OPPORTUNITY"
#dataset = "KU-HAR"
dataset = "PAMAP2"
#dataset = "REALDISP"

#memo = "_D" + str(L)
memo=""

input_nc, segment_size, class_num = data_info(dataset)

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

#criterion = nn.CrossEntropyLoss().cuda()

# Pooling config
d_in = 128  # Output size of feature extractor
n_clz = class_num
dropout = 0.1
apply_positional_encoding = True

device = torch.device("cuda")

# Example network
class ExampleNet(nn.Module):
    def __init__(self, feature_extractor, pool):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.pool = pool

    def forward(self, bags, pos=None):
        timestep_embeddings = self.feature_extractor(bags)
        return self.pool(timestep_embeddings, pos=pos)

# Create network using InceptionTime feature extractor and Conjunctive Pooling
net = ExampleNet(
    backbone.InceptionTimeFeatureExtractor(input_nc),
    pooling.MILConjunctivePooling(
        d_in,
        n_clz,
        dropout=dropout,
        apply_positional_encoding=apply_positional_encoding,
    ),
)

model = MILLETModel("ExampleNet", device, n_clz, net)
model.fit(train_queue, n_epochs=epoches)


test_results_dict = model.evaluate(eval_queue)


print(test_results_dict["f1_score"])



# print("param size = %fMB" % count_parameters_in_MB(model))

# optimizer = torch.optim.Adam(

# Initialize metrics collector
metrics_collector = MetricsCollector(
    model_name='millet',
    dataset=dataset,
    task_type='HAR',
    save_dir='results'
)
#     model.parameters(),
#     lr=5e-4,
#     betas=(0.9,0.999),
#     weight_decay=5e-4,
#     eps=1e-08
# )
# max_f1 = 0
# weighted_avg_f1 = 0

# for epoch in range(epoches):
    
#     # training
#     train_loss, train_acc = train(train_queue, model, criterion, optimizer)

#     # evaluating
#     eval_loss, y_pred = infer(eval_queue, model, criterion)
#     results = classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True)
#     weighted_avg_f1 = results['weighted avg']['f1-score']
#     if (epoch+1) % 50 == 0:
#         print('training... ', epoch+1)
#     if max_f1 < weighted_avg_f1:
#         if weighted_avg_f1 < 0.88:
#             torch.save(model.state_dict(), 'millet/save/'+dataset + memo + '.pt')
#         print("epoch %d, loss %e, weighted f1 %f, best_f1 %f" % (epoch+1, eval_loss, weighted_avg_f1, max_f1))
#         max_f1 = weighted_avg_f1
#         print(classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4))
#         if dataset=='UniMiB-SHAR':
#             print('ADL:',classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=list(range(9)))['weighted avg']['f1-score'])
#             print('Falls:',classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=list(range(9,17)))['weighted avg']['f1-score'])
#         elif dataset=='PAMAP2':
#             print('ADL:',classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=[0,1,2,3,4,5,6,10,11,12,13,17])['weighted avg']['f1-score'])
#             print('Complex:',classification_report(y_test_unary, np.argmax(y_pred, axis=1), digits=4, output_dict=True, labels=[7,8,9,14,15,16])['weighted avg']['f1-score'])
        
 

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

# Track inference time and re-evaluate
with metrics_collector.track_inference():
    test_results_dict = model.evaluate(eval_queue)

# Get predictions for metrics
y_pred_labels = test_results_dict.get('y_pred', None)
y_true_labels = test_results_dict.get('y_true', None)

# If predictions not in results, get them separately
if y_pred_labels is None or y_true_labels is None:
    # Fallback: re-run evaluation to get predictions
    test_results_dict = model.evaluate(eval_queue)
    y_pred_labels = test_results_dict.get('y_pred', y_test_unary)
    y_true_labels = y_test_unary

# Compute throughput
test_samples = len(y_true_labels)
metrics_collector.compute_throughput(test_samples, phase='inference')

# Compute classification metrics
metrics_collector.compute_classification_metrics(y_true_labels, y_pred_labels)

# Compute model complexity (millet model structure is different, skip for now)
try:
    input_shape = (1, input_nc, segment_size)
    metrics_collector.compute_model_complexity(model.net, input_shape, device=device)
except Exception as e:
    print(f"Could not compute model complexity: {e}")

# Save comprehensive metrics
metrics_collector.save_metrics()
metrics_collector.print_summary()

