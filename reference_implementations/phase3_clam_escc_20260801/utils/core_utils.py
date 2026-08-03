import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_pooling import FixedPoolMIL
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, WeightedRandomSampler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False, monitor='loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            monitor (str): Which validation metric to use for checkpoint selection.
                            'loss' minimizes validation loss, 'auc' maximizes validation AUC.
        """
        if monitor not in ['loss', 'auc']:
            raise ValueError("monitor must be either 'loss' or 'auc'")
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_value = np.Inf if monitor == 'loss' else -np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt', val_auc = None):

        if self.monitor == 'loss':
            current_value = val_loss
            score = -val_loss
        else:
            if val_auc is None:
                raise ValueError("val_auc must be provided when monitor='auc'")
            current_value = val_auc
            score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(current_value, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(current_value, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, metric_value, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            direction = 'decreased' if self.monitor == 'loss' else 'increased'
            print(f'Validation {self.monitor} {direction} ({self.best_value:.6f} --> {metric_value:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.best_value = metric_value

def attention_entropy_maximization_loss(A_raw):
    log_attention = F.log_softmax(A_raw, dim=1)
    attention = torch.exp(log_attention)
    return (attention * log_attention).sum(dim=1).mean()

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        features, label = self.dataset[idx][:2]
        return features, label, idx


def collate_MIL_with_index(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    index = torch.LongTensor([item[2] for item in batch])
    return [img, label, index]


def get_split_loader_with_index(split_dataset, training=False, testing=False, weighted=False):
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    indexed_dataset = IndexedDataset(split_dataset)
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                sampler = WeightedRandomSampler(weights, len(weights))
            else:
                sampler = RandomSampler(split_dataset)
        else:
            sampler = SequentialSampler(split_dataset)
        return DataLoader(indexed_dataset, batch_size=1, sampler=sampler, collate_fn=collate_MIL_with_index, **kwargs)

    ids = np.random.choice(np.arange(len(split_dataset)), int(len(split_dataset) * 0.1), replace=False)
    return DataLoader(indexed_dataset, batch_size=1, sampler=SubsetSequentialSampler(ids),
                      collate_fn=collate_MIL_with_index, **kwargs)


class SameLabelPseMix:
    def __init__(self, dataset, pseudo_bags=30, dividing='proto', phenotypes=8, iter_tuning=8,
                 alpha=1.0, mixup_prob=1.0, min_instances=1):
        self.dataset = dataset
        self.pseudo_bags = max(1, int(pseudo_bags))
        self.dividing = dividing
        self.phenotypes = max(1, int(phenotypes))
        self.iter_tuning = max(0, int(iter_tuning))
        self.alpha = float(alpha)
        self.mixup_prob = float(mixup_prob)
        self.min_instances = max(1, int(min_instances))
        self.candidates = self._build_same_label_candidates()
        self.num_available = sum(1 for idxs in self.candidates.values() if len(idxs) > 0)

    def _build_same_label_candidates(self):
        slide_data = self.dataset.slide_data.reset_index(drop=True)
        grouped = {}
        for idx, row in slide_data.iterrows():
            key = int(row['label'])
            grouped.setdefault(key, []).append(idx)

        candidates = {}
        for idx, row in slide_data.iterrows():
            key = int(row['label'])
            candidates[idx] = [mate_idx for mate_idx in grouped.get(key, []) if mate_idx != idx]
        return candidates

    def _split_pseudo_bags(self, bag):
        n_bags = min(self.pseudo_bags, bag.size(0))
        if self.dividing == 'random' or n_bags == 1:
            perm = torch.randperm(bag.size(0), device=bag.device)
            return list(torch.tensor_split(bag[perm], n_bags, dim=0))

        assignments = self._proto_assignments(bag)
        pseudo_indices = [[] for _ in range(n_bags)]
        for cluster_id in range(int(assignments.max().item()) + 1):
            cluster_indices = torch.nonzero(assignments == cluster_id, as_tuple=False).flatten()
            if cluster_indices.numel() == 0:
                continue
            cluster_indices = cluster_indices[torch.randperm(cluster_indices.numel(), device=bag.device)]
            offset = int(torch.randint(n_bags, (1,), device=bag.device).item())
            for pos, index in enumerate(cluster_indices):
                pseudo_indices[(pos + offset) % n_bags].append(index)

        if any(len(indices) == 0 for indices in pseudo_indices):
            perm = torch.randperm(bag.size(0), device=bag.device)
            return list(torch.tensor_split(bag[perm], n_bags, dim=0))

        return [bag[torch.stack(indices).long()] for indices in pseudo_indices]

    def _proto_assignments(self, bag):
        features = bag.float()
        n_clusters = min(self.phenotypes, features.size(0))
        if n_clusters == 1:
            return torch.zeros(features.size(0), dtype=torch.long, device=features.device)

        with torch.no_grad():
            normalized = F.normalize(features, dim=1)
            prototype = F.normalize(normalized.mean(dim=0, keepdim=True), dim=1)
            similarity = torch.matmul(normalized, prototype.t()).flatten()
            sorted_indices = torch.argsort(similarity)
            center_positions = torch.linspace(0, sorted_indices.numel() - 1, n_clusters, device=features.device).long()
            centers = features[sorted_indices[center_positions]].clone()

            assignments = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
            for _ in range(max(1, self.iter_tuning)):
                distances = torch.cdist(features, centers)
                assignments = torch.argmin(distances, dim=1)
                updated_centers = []
                for cluster_id in range(n_clusters):
                    mask = assignments == cluster_id
                    if mask.any():
                        updated_centers.append(features[mask].mean(dim=0))
                    else:
                        updated_centers.append(centers[cluster_id])
                centers = torch.stack(updated_centers, dim=0)
            return assignments

    def _select_pseudo_bags(self, pseudo_bags, count):
        if count <= 0:
            return pseudo_bags[0].new_empty((0, pseudo_bags[0].size(1)))
        count = min(count, len(pseudo_bags))
        selected = np.random.choice(len(pseudo_bags), count, replace=False)
        return torch.cat([pseudo_bags[idx] for idx in selected], dim=0)

    def _load_features(self, idx, device):
        features, label = self.dataset[idx][:2]
        return features.to(device), int(label)

    def __call__(self, data, label, slide_index):
        if self.alpha <= 0 or data.size(0) == 0:
            return data, 1.0, False

        current_idx = int(slide_index.item())
        candidates = self.candidates.get(current_idx, [])
        if not candidates:
            return data, 1.0, False

        mate_idx = int(np.random.choice(candidates))
        mate_data, mate_label = self._load_features(mate_idx, data.device)
        if mate_label != int(label.item()) or mate_data.size(0) == 0:
            return data, 1.0, False

        pseudo_a = self._split_pseudo_bags(data)
        pseudo_b = self._split_pseudo_bags(mate_data)
        n_bags = min(len(pseudo_a), len(pseudo_b))
        pseudo_a = pseudo_a[:n_bags]
        pseudo_b = pseudo_b[:n_bags]

        lam = np.random.beta(self.alpha, self.alpha)
        lam = min(lam, 1.0 - 1e-5)
        lam_discrete = int(lam * (n_bags + 1))
        part_a = self._select_pseudo_bags(pseudo_a, lam_discrete)

        if np.random.rand() <= self.mixup_prob:
            part_b = self._select_pseudo_bags(pseudo_b, n_bags - lam_discrete)
            mixed = torch.cat([part_a, part_b], dim=0)
            mix_ratio = lam_discrete / n_bags
        else:
            mixed = part_a
            mix_ratio = 1.0

        if mixed.size(0) < self.min_instances:
            return data, 1.0, False
        return mixed, mix_ratio, True

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb', 'mean_pool', 'max_pool'] and args.fusion_mode != 'none':
        model_dict.update({
            'fusion_mode': args.fusion_mode,
            'path_dim': args.path_dim,
            'gene_dim': args.gene_dim,
            'path_proj_dim': args.path_proj_dim,
            'gene_proj_dim': args.gene_proj_dim,
            'cross_attn_heads': args.cross_attn_heads,
            'cross_attn_direction': args.cross_attn_direction,
            'cross_attn_output': args.cross_attn_output,
            'fusion_projector': args.fusion_projector,
        })

    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.fusion_mode != 'none':
            model_dict.update({
                'external_attn_size': args.external_attn_size,
                'fusion_output_dim': args.fusion_output_dim,
                'fusion_hidden_dim': args.fusion_hidden_dim,
                'fusion_rank': args.fusion_rank,
            })

        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    elif args.model_type == 'mil':
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    elif args.model_type == 'mean_pool':
        model = FixedPoolMIL(pooling='mean', **model_dict)
    elif args.model_type == 'max_pool':
        model = FixedPoolMIL(pooling='max', **model_dict)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")
    
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    psemix_augmentor = None
    if args.psemix:
        train_loader = get_split_loader_with_index(train_split, training=True, testing=args.testing,
                                                   weighted=args.weighted_sample)
        min_instances = args.B if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster else 1
        psemix_augmentor = SameLabelPseMix(train_split, pseudo_bags=args.psemix_pseudo_bags,
                                           dividing=args.psemix_dividing,
                                           phenotypes=args.psemix_l,
                                           iter_tuning=args.psemix_iter_tuning,
                                           alpha=args.psemix_alpha, mixup_prob=args.psemix_prob,
                                           min_instances=min_instances)
    else:
        train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 10, stop_epoch=10, verbose = True,
                                       monitor=args.early_stopping_metric)

    else:
        early_stopping = None
    print('Done!')
    if args.model_type in ['clam_sb', 'clam_mb'] and getattr(args, 'aem_weight', 0.0) > 0:
        print('AEM enabled: weight={}'.format(args.aem_weight))
    if args.psemix:
        print('PseMix enabled: pseudo_bags={}, dividing={}, l={}, iter_tuning={}, alpha={}, mixup_prob={}, weight={}, same-label train slides with mates={}/{}'.format(
            args.psemix_pseudo_bags, args.psemix_dividing, args.psemix_l, args.psemix_iter_tuning,
            args.psemix_alpha, args.psemix_prob, args.psemix_weight,
            psemix_augmentor.num_available, len(train_split)))

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn,
                            args.aem_weight, psemix_augmentor, args.psemix_weight)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        else:
            aem_weight = args.aem_weight if args.model_type in ['clam_sb', 'clam_mb'] else 0.0
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn,
                       aem_weight, psemix_augmentor, args.psemix_weight)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None,
                    aem_weight = 0.0, psemix_augmentor = None, psemix_weight = 1.0):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    train_aem_loss = 0.
    train_psemix_loss = 0.
    train_psemix_count = 0
    train_psemix_ratio = 0.
    inst_count = 0

    print('\n')
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 3:
            data, label, slide_index = batch
        else:
            data, label = batch
            slide_index = None
        data, label = data.to(device), label.to(device)
        psemix_ratio = 1.0

        logits, Y_prob, Y_hat, A_raw, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 
        aem_loss_value = 0.
        if aem_weight > 0:
            aem_loss = attention_entropy_maximization_loss(A_raw)
            aem_loss_value = aem_loss.item()
            train_aem_loss += aem_loss_value
            total_loss = total_loss + aem_weight * aem_loss

        psemix_loss_value = 0.
        if psemix_augmentor is not None and slide_index is not None and psemix_weight > 0:
            psemix_data, psemix_ratio, psemix_applied = psemix_augmentor(data, label, slide_index)
            if psemix_applied:
                psemix_logits, _, _, psemix_A_raw, psemix_instance_dict = model(
                    psemix_data, label=label, instance_eval=True)
                psemix_bag_loss = loss_fn(psemix_logits, label)
                psemix_instance_loss = psemix_instance_dict['instance_loss']
                psemix_total_loss = bag_weight * psemix_bag_loss + (1-bag_weight) * psemix_instance_loss
                if aem_weight > 0:
                    psemix_total_loss = psemix_total_loss + aem_weight * attention_entropy_maximization_loss(psemix_A_raw)
                psemix_loss_value = psemix_total_loss.item()
                train_psemix_loss += psemix_loss_value
                train_psemix_count += 1
                train_psemix_ratio += psemix_ratio
                total_loss = total_loss + psemix_weight * psemix_total_loss

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, aem_loss: {:.4f}, psemix_loss: {:.4f}, psemix_ratio: {:.3f}, weighted_loss: {:.4f}, '.format(
                batch_idx, loss_value, instance_loss_value, aem_loss_value, psemix_loss_value, psemix_ratio, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    if aem_weight > 0:
        train_aem_loss /= len(loader)
    if train_psemix_count > 0:
        train_psemix_ratio /= train_psemix_count
        train_psemix_loss /= train_psemix_count
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    if aem_weight > 0:
        print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_aem_loss: {:.4f}, train_error: {:.4f}'.format(
            epoch, train_loss, train_inst_loss, train_aem_loss, train_error))
    else:
        print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    if psemix_augmentor is not None:
        print('PseMix added {}/{} augmented bags, mean current-slide pseudo-bag ratio: {:.3f}, mean psemix_loss: {:.4f}'.format(
            train_psemix_count, len(loader), train_psemix_ratio, train_psemix_loss))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
        if aem_weight > 0:
            writer.add_scalar('train/aem_loss', train_aem_loss, epoch)
        if psemix_augmentor is not None:
            writer.add_scalar('train/psemix_count', train_psemix_count, epoch)
            writer.add_scalar('train/psemix_current_ratio', train_psemix_ratio, epoch)
            writer.add_scalar('train/psemix_loss', train_psemix_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None,
               aem_weight = 0.0, psemix_augmentor = None, psemix_weight = 1.0):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    train_aem_loss = 0.
    train_psemix_loss = 0.
    train_psemix_count = 0
    train_psemix_ratio = 0.

    print('\n')
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 3:
            data, label, slide_index = batch
        else:
            data, label = batch
            slide_index = None
        data, label = data.to(device), label.to(device)
        psemix_ratio = 1.0

        logits, Y_prob, Y_hat, A_raw, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        total_loss = loss
        aem_loss_value = 0.
        if aem_weight > 0:
            aem_loss = attention_entropy_maximization_loss(A_raw)
            aem_loss_value = aem_loss.item()
            train_aem_loss += aem_loss_value
            total_loss = total_loss + aem_weight * aem_loss

        psemix_loss_value = 0.
        if psemix_augmentor is not None and slide_index is not None and psemix_weight > 0:
            psemix_data, psemix_ratio, psemix_applied = psemix_augmentor(data, label, slide_index)
            if psemix_applied:
                psemix_logits, _, _, psemix_A_raw, _ = model(psemix_data)
                psemix_total_loss = loss_fn(psemix_logits, label)
                if aem_weight > 0:
                    psemix_total_loss = psemix_total_loss + aem_weight * attention_entropy_maximization_loss(psemix_A_raw)
                psemix_loss_value = psemix_total_loss.item()
                train_psemix_loss += psemix_loss_value
                train_psemix_count += 1
                train_psemix_ratio += psemix_ratio
                total_loss = total_loss + psemix_weight * psemix_total_loss
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            if aem_weight > 0:
                print('batch {}, loss: {:.4f}, aem_loss: {:.4f}, psemix_loss: {:.4f}, psemix_ratio: {:.3f}, weighted_loss: {:.4f}, label: {}, bag_size: {}'.format(
                    batch_idx, loss_value, aem_loss_value, psemix_loss_value, psemix_ratio, total_loss.item(), label.item(), data.size(0)))
            else:
                print('batch {}, loss: {:.4f}, psemix_loss: {:.4f}, psemix_ratio: {:.3f}, label: {}, bag_size: {}'.format(
                    batch_idx, loss_value, psemix_loss_value, psemix_ratio, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    if aem_weight > 0:
        train_aem_loss /= len(loader)
    if train_psemix_count > 0:
        train_psemix_ratio /= train_psemix_count
        train_psemix_loss /= train_psemix_count

    if aem_weight > 0:
        print('Epoch: {}, train_loss: {:.4f}, train_aem_loss: {:.4f}, train_error: {:.4f}'.format(
            epoch, train_loss, train_aem_loss, train_error))
    else:
        print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    if psemix_augmentor is not None:
        print('PseMix added {}/{} augmented bags, mean current-slide pseudo-bag ratio: {:.3f}, mean psemix_loss: {:.4f}'.format(
            train_psemix_count, len(loader), train_psemix_ratio, train_psemix_loss))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        if aem_weight > 0:
            writer.add_scalar('train/aem_loss', train_aem_loss, epoch)
        if psemix_augmentor is not None:
            writer.add_scalar('train/psemix_count', train_psemix_count, epoch)
            writer.add_scalar('train/psemix_current_ratio', train_psemix_ratio, epoch)
            writer.add_scalar('train/psemix_loss', train_psemix_loss, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)),
                       val_auc=auc)
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)),
                       val_auc=auc)
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger
