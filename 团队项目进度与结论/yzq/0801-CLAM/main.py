from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--feature_dir', type=str, default='ESCC_uni2h_features',
                    help='feature subdirectory under data_root_dir for ESCC tasks')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--early_stopping_metric', type=str, choices=['loss', 'auc'], default='loss',
                    help='metric for selecting the best checkpoint when early stopping is enabled (default: loss)')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'mean_pool', 'max_pool'], default='clam_sb',
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'ESCC_pCR', 'ESCC_MPR'])
parser.add_argument('--fusion_mode', type=str,
                    choices=['none', 'dual_branch', 'dual_attention', 'gated_modality', 'film',
                             'cross_attention', 'modality_attention', 'external_attention',
                             'adaptive_fusion', 'low_rank_fusion'],
                    default='none',
                    help='optional CLAM fusion mode for concatenated multimodal patch features')
parser.add_argument('--path_dim', type=int, default=1536,
                    help='pathology feature dimension for dual_branch fusion')
parser.add_argument('--gene_dim', type=int, default=30,
                    help='gene score feature dimension for dual_branch fusion')
parser.add_argument('--path_proj_dim', type=int, default=512,
                    help='pathology branch projection dimension for dual_branch fusion')
parser.add_argument('--gene_proj_dim', type=int, default=128,
                    help='gene branch projection dimension for dual_branch fusion')
parser.add_argument('--cross_attn_heads', type=int, default=4,
                    help='number of heads for cross_attention fusion')
parser.add_argument('--cross_attn_direction', type=str, choices=['path_to_gene', 'gene_to_path'],
                    default='path_to_gene',
                    help='cross_attention direction: path queries gene, or gene queries path')
parser.add_argument('--cross_attn_output', type=str,
                    choices=['fused', 'attended', 'fused_plus_path_raw', 'fused_plus_query'],
                    default='fused',
                    help='features sent to CLAM after cross_attention fusion')
parser.add_argument('--external_attn_size', type=int, default=64,
                    help='number of external memory slots for external_attention fusion')
parser.add_argument('--fusion_output_dim', type=int, default=512,
                    help='output feature dimension for adaptive_fusion and low_rank_fusion')
parser.add_argument('--fusion_hidden_dim', type=int, default=256,
                    help='hidden dimension for adaptive_fusion weight generator')
parser.add_argument('--fusion_rank', type=int, default=64,
                    help='rank for low_rank_fusion')
parser.add_argument('--fusion_projector', type=str, choices=['linear', 'identity'], default='linear',
                    help='use linear projection branches or identity branches before fusion')
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--aem_weight', type=float, default=0.0,
                    help='AEM attention entropy maximization weight; 0 disables AEM (default: 0.0)')
parser.add_argument('--psemix', action='store_true', default=False,
                    help='enable same-label train-split PseMix pseudo-bag augmentation during training')
parser.add_argument('--psemix_pseudo_bags', type=int, default=30,
                    help='number of pseudo-bags to split each slide into for PseMix (default: 30)')
parser.add_argument('--psemix_dividing', type=str, choices=['proto', 'random'], default='proto',
                    help='pseudo-bag dividing method for PseMix (default: proto)')
parser.add_argument('--psemix_l', type=int, default=8,
                    help='number of phenotypes/prototype clusters for PseMix proto dividing (default: 8)')
parser.add_argument('--psemix_iter_tuning', type=int, default=8,
                    help='number of prototype cluster fine-tuning iterations for PseMix (default: 8)')
parser.add_argument('--psemix_alpha', type=float, default=1.0,
                    help='beta distribution alpha for PseMix pseudo-bag mixing ratio (default: 1.0)')
parser.add_argument('--psemix_prob', type=float, default=1.0,
                    help='probability of mixing selected pseudo-bags with a same-label train slide (default: 1.0)')
parser.add_argument('--psemix_weight', type=float, default=1.0,
                    help='weight for the additional PseMix augmented-bag loss (default: 1.0)')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt,
            'early_stopping_metric': args.early_stopping_metric,
            'psemix': args.psemix}
if args.psemix:
    settings.update({'psemix_pseudo_bags': args.psemix_pseudo_bags,
                     'psemix_dividing': args.psemix_dividing,
                     'psemix_l': args.psemix_l,
                     'psemix_iter_tuning': args.psemix_iter_tuning,
                     'psemix_alpha': args.psemix_alpha,
                     'psemix_prob': args.psemix_prob,
                     'psemix_weight': args.psemix_weight,
                     'psemix_pairing': 'same_label_train_split'})
if args.task in ['ESCC_pCR', 'ESCC_MPR']:
    settings.update({'feature_dir': args.feature_dir})

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B,
                    'aem_weight': args.aem_weight})

if args.model_type in ['clam_sb', 'clam_mb', 'mean_pool', 'max_pool'] and args.fusion_mode != 'none':
   settings.update({'fusion_mode': args.fusion_mode,
                    'path_dim': args.path_dim,
                    'gene_dim': args.gene_dim,
                    'path_proj_dim': args.path_proj_dim,
                    'gene_proj_dim': args.gene_proj_dim,
                    'cross_attn_heads': args.cross_attn_heads,
                    'cross_attn_direction': args.cross_attn_direction,
                    'cross_attn_output': args.cross_attn_output,
                    'fusion_projector': args.fusion_projector})
   if args.model_type in ['clam_sb', 'clam_mb']:
       settings.update({'external_attn_size': args.external_attn_size,
                        'fusion_output_dim': args.fusion_output_dim,
                        'fusion_hidden_dim': args.fusion_hidden_dim,
                        'fusion_rank': args.fusion_rank})

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping 

elif args.task == 'ESCC_pCR':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/ESCC_pCR_clam.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_dir),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'non_pCR':0, 'pCR':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'ESCC_MPR':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/ESCC_MPR_clam.csv',
                            data_dir= os.path.join(args.data_root_dir, args.feature_dir),
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {'non_MPR':0, 'MPR':1},
                            patient_strat=False,
                            ignore=[])
        
else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")
