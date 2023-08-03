# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os, time, random
import datetime
from tqdm import tqdm

from sklearn.metrics import auc, roc_auc_score, precision_recall_curve

#####我加的1130
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support
from sklearn import metrics ###########################################3333


from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from noam_opt import *


id_name = 'ID' # column name in dataframe
label_name = ['label'] # some task may have mutiple labels
metal_list = ["PRO", "CA", "MG", "MN"]
sequence_name = "sequence"
gpus = list(range(torch.cuda.device_count()))
print("Available GPUs", gpus)

print_pred_labels=True

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def Metric(preds, labels):
    labels = np.array(labels).reshape(-1)
    preds = np.array(preds).reshape(-1)

    tn, fp, fn, tp = confusion_matrix(labels, preds > 0.5).ravel()
    #tn, fp, fn, tp = confusion_matrix(labels, preds > 0).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    pr = tp / (tp + fp)
    mcc = ((tp * tn) - (fn * fp)) / ((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn)) ** 0.5
    
    AUC = roc_auc_score(labels,preds)
    precisions, recalls, _ = precision_recall_curve(labels, preds)  #######
    AUPRC = auc(recalls, precisions)
    return AUC, AUPRC    ###, mcc, mccL, precisionL, recallL  ########


def Write_log(logFile,text,isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write('\n')
    return None


class TaskDataset:
    def __init__(self, df, protein_data, label_name):
        self.df = df
        self.protein_data = protein_data
        self.label_name = label_name

    def __len__(self):
        return (self.df.shape[0])

    def __getitem__(self, idx):
        pdb_id = self.df.loc[idx,'ID']
        protein_X, protein_node_features, protein_masks, labels = self.protein_data[pdb_id]

        return {
            'PDB_ID': pdb_id,
            'PROTEIN_X': protein_X,
            'PROTEIN_NODE_FEAT': protein_node_features,
            'PROTEIN_MASK': protein_masks,
            'LABEL': labels,
        }

    def collate_fn(self, batch):
        pdb_ids = [item['PDB_ID'] for item in batch]
        protein_X = torch.stack([item['PROTEIN_X'] for item in batch], dim=0)
        protein_node_features = torch.stack([item['PROTEIN_NODE_FEAT'] for item in batch], dim=0)
        protein_masks = torch.stack([item['PROTEIN_MASK'] for item in batch], dim=0)
        labels = torch.stack([item['LABEL'] for item in batch], dim=0)

        return pdb_ids, protein_X, protein_node_features, protein_masks, labels


# main function
def NN_train_and_predict(train, test, protein_data, model_class, config, logit=False, output_root='./output/', run_id=None, args=None):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if not run_id:
        run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        while os.path.exists(output_root+run_id+'/'):
            time.sleep(1)
            run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    output_path = output_root + run_id + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)


    node_features = config['node_features']
    edge_features = config['edge_features']
    hidden_dim = config['hidden_dim']
    num_encoder_layers = config['num_encoder_layers']
    k_neighbors = config['k_neighbors']
    augment_eps = config['augment_eps']
    dropout = config['dropout']

    id_name = config['id_name']
    obj_max = config['obj_max']
    epochs = config['epochs']
    patience = config['patience']
    batch_size = config['batch_size']
    num_samples = config['num_samples']
    folds = config['folds']
    seed = config['seed']

    task = args.task
    
    oof = None

    if test is not None:
        if train is None:
            log = open(output_path + 'test.log','w', buffering=1)
            Write_log(log,str(config)+'\n')
        sub = test[[id_name, sequence_name]]

        if task == "Metal":
            for m in metal_list:
                sub[m] = 0.0
        elif isinstance(label_name,list):
            for l in label_name:
                sub[l] = 0.0
        else:
            sub[label_name] = 0.0

        test_dataset = TaskDataset(test, protein_data, label_name)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn, shuffle=False, drop_last=False, num_workers=args.num_workers, prefetch_factor=2)
        models = []
        print(folds)
        for fold in range(folds):
            if not os.path.exists(output_path + 'fold%s.ckpt'%fold):
                continue

            if model_class.__name__ in ['GraphTrans', "MetalSite"]:  #Transformer_YQM
                model = model_class(node_features, edge_features, hidden_dim, num_encoder_layers, k_neighbors, augment_eps, dropout)

            model.cuda()
            state_dict = torch.load(output_path + 'fold%s.ckpt'%fold, torch.device('cuda') )
            model.load_state_dict(state_dict)
            if len(gpus) > 1:
                model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

            model.eval()
            models.append(model)
        print('model count:',len(models))

        test_preds = []
        test_outputs = [] # 用来导出结果到submission.csv
        test_Y = []
        test_preds4 = [[], [], [], []]
        test_Y4 = [[], [], [], []]
        k = 0
        with torch.no_grad():
            for data in tqdm(test_dataloader):
                # print(data)
                k += 1
                protein_X, protein_node_features, protein_masks, y = [d.cuda() for d in data[1:]]
                
                if logit:
                    outputs = [model(protein_X, protein_node_features, protein_masks).sigmoid() for model in models]
                else:
                    outputs = [model(protein_X, protein_node_features, protein_maskss) for model in models]
                # print(outputs)
                outputs = torch.stack(outputs,0).mean(0) # 5个模型预测结果求平均,最终shape=(bsize, max_len)
                test_outputs.append(outputs.detach().cpu().numpy())

                if task == "Metal":
                    for i in range(4):
                        test_seq_y = torch.masked_select(y[:,i*1000:(i+1)*1000], protein_label_masks[:,i*1000:(i+1)*1000].bool()) # maxlen 1000
                        test_seq_preds = torch.masked_select(outputs[:,i*1000:(i+1)*1000], protein_label_masks[:,i*1000:(i+1)*1000].bool())
                        test_Y4[i].append(test_seq_y.detach().cpu().numpy())
                        test_preds4[i].append(test_seq_preds.detach().cpu().numpy())
                else:
                    test_seq_y = torch.masked_select(y, protein_masks.bool())
                    test_seq_preds = torch.masked_select(outputs, protein_masks.bool())
                    test_preds.append(test_seq_preds.cpu().detach().numpy())
                    test_Y.append(test_seq_y.cpu().detach().numpy())
#                     test_seq_pd_labels = []
#                     test_seq_gt_labels = []
#                     if print_pred_labels:
#                         for pred in test_seq_preds:
#                             if pred > 0.5:
#                                 test_seq_pd_labels.append(1)
#                             else:
#                                 test_seq_pd_labels.append(0)
#                         test_seq_gt_labels = [int(label) for label in test_seq_y.cpu().detach().numpy()]
#                         print('test_seq_gt_labels: ', test_seq_gt_labels)
#                         print('test_seq_pd_labels: ', test_seq_pd_labels)
#                         pos = 0
#                         for gt,pd in zip(test_seq_gt_labels, test_seq_pd_labels):
#                             if gt != pd:
#                                 print('the uncorrect prediction pos is {}, it predicts {} while the gt is {}.'.format(pos, pd, gt))
#                             pos += 1
#                     if k >= 5:
#                         break

        if task == "Metal":
            test_metrics = []
            for i in range(4):
                test_preds4[i] = np.concatenate(test_preds4[i])
                test_Y4[i] = np.concatenate(test_Y4[i])
                test_metrics.append(Metric(test_preds4[i], test_Y4[i]))
            Write_log(log,'test_auc: %.6f,%.6f,%.6f,%.6f, test_auprc: %.6f,%.6f,%.6f,%.6f'\
            %(test_metrics[0][0],test_metrics[1][0],test_metrics[2][0],test_metrics[3][0],test_metrics[0][1],test_metrics[1][1],test_metrics[2][1],test_metrics[3][1]))

        else:
            test_preds = np.concatenate(test_preds)
            test_Y = np.concatenate(test_Y)
#             print('test_preds: ', test_preds)
#             print('test_Y: ', test_Y)
#             import pdb; pdb.set_trace()
            test_metric = Metric(test_preds, test_Y)

            Write_log(log,'test_auc:%.6f, test_auprc:%.6f'%(test_metric[0],test_metric[1]))

        test_outputs = np.concatenate(test_outputs) # shape = (num_samples, max_len) or (num_samples,  4 * max_len)

        if task == "Metal":
            for i in range(4):
                sub[metal_list[i]] = [test_outputs[:,i*1000:(i+1)*1000][j, :len(sub.loc[j,sequence_name])].tolist() for j in range(len(sub))]
        else:
            sub[label_name] = [test_outputs[i, :len(sub.loc[i,sequence_name])].tolist() for i in range(len(sub))]

        sub.to_csv(output_path + 'submission.csv',index=False)
        log.close()
