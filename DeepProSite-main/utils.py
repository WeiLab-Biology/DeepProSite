# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os, time, random
import datetime
from tqdm import tqdm

from sklearn.metrics import auc, roc_auc_score, precision_recall_curve

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support
from sklearn import metrics 


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
    return AUC, AUPRC, mcc    ###, mcc, mccL, precisionL, recallL  ########


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
        protein_X, protein_node_features, protein_masks = self.protein_data[pdb_id]

        return {
            'PDB_ID': pdb_id,
            'PROTEIN_X': protein_X,
            'PROTEIN_NODE_FEAT': protein_node_features,
            'PROTEIN_MASK': protein_masks,
#             'LABEL': labels,
        }

    def collate_fn(self, batch):
        pdb_ids = [item['PDB_ID'] for item in batch]
        protein_X = torch.stack([item['PROTEIN_X'] for item in batch], dim=0)
        protein_node_features = torch.stack([item['PROTEIN_NODE_FEAT'] for item in batch], dim=0)
        protein_masks = torch.stack([item['PROTEIN_MASK'] for item in batch], dim=0)
        # labels = torch.stack([item['LABEL'] for item in batch], dim=0)

        return pdb_ids, protein_X, protein_node_features, protein_masks # , labels


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

    if train is not None:
        os.system(f'cp ./*.py {output_path}')

        oof = train[[id_name, sequence_name]]
        oof['fold'] = -1
        if task == "Metal":
            for m in metal_list:
                oof[m] = 0.0
                oof[m] = oof[m].astype(np.float32)
        elif isinstance(label_name, list):
            for l in label_name:
                oof[l] = 0.0
                oof[l] = oof[l].astype(np.float32)
        else:
            oof[label_name] = 0.0
            oof[label_name] = oof[label_name].astype(np.float32)
    else:
        oof = None

    if train is not None:
        log = open(output_path + 'train.log','w', buffering=1)
        log.write(str(config) + '\n')

        all_valid_metric = []

        kf = KFold(n_splits = folds, shuffle=True, random_state=seed)

        train_folds = []
        for fold, (train_index, val_index) in enumerate(kf.split(train, train[label_name])):
            print("\n========== Fold " + str(fold + 1) + " ==========")

            train_dataset = TaskDataset(train.loc[train_index].reset_index(drop=True), protein_data, label_name)
            sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_samples)   # when set to sampler, shuffle is False.
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, sampler=sampler, shuffle=False, drop_last=True, num_workers=args.num_workers, prefetch_factor=2)

            valid_dataset = TaskDataset(train.loc[val_index].reset_index(drop=True), protein_data, label_name)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn, shuffle=False, drop_last=False, num_workers=args.num_workers, prefetch_factor=2)
            
            if model_class.__name__ in ['GraphTrans', 'MetalSite']: 
                model = model_class(node_features, edge_features, hidden_dim, num_encoder_layers, k_neighbors, augment_eps, dropout)

            model.cuda()

            optimizer = get_std_opt(task, model.parameters(), hidden_dim)

            if len(gpus) > 1:
                model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

            loss_tr = nn.BCEWithLogitsLoss(reduction='none')

            if obj_max == 1:
                best_valid_metric = 0
            else:
                best_valid_metric = 1e9
            not_improve_epochs = 0
            if args.train:
                for epoch in range(epochs):
                    train_loss = 0.0
                    train_num = 0
                    model.train()

                    train_pdb_ids = []
                    train_preds = []
                    train_Y = []
                    bar = tqdm(train_dataloader)
                    for i, data in enumerate(bar):
                        optimizer.zero_grad()
                        protein_X, protein_node_features, protein_masks, y = [d.cuda() for d in data[1:]]
 
                        outputs = model(protein_X, protein_node_features, protein_masks)
                        if task == "Metal": # multi-task:
                            loss = loss_tr(outputs, y) * protein_label_masks
                        else:
                            loss = loss_tr(outputs, y) * protein_masks
                        loss = loss.sum() / protein_masks.sum()
                        loss.backward()
                        optimizer.step()

                        if logit:
                            outputs = outputs.sigmoid() # outputs.shape = (batch_size, max_len)

                        train_pdb_ids.extend(data[0])

                        if task == "Metal":
                            train_seq_preds = torch.masked_select(outputs, protein_label_masks.bool()) 
                            train_seq_y = torch.masked_select(y, protein_label_masks.bool())
                        else:
                            train_seq_preds = torch.masked_select(outputs, protein_masks.bool()) 
                            train_seq_y = torch.masked_select(y, protein_masks.bool())

                        train_preds.append(train_seq_preds.detach().cpu().numpy())
                        train_Y.append(train_seq_y.clone().detach().cpu().numpy())

                        train_num += len(train_seq_y)
                        train_loss += len(train_seq_y) * loss.item()

                        bar.set_description('loss: %.4f' % (loss.item()))

                    train_loss /= train_num
                    train_preds = np.concatenate(train_preds)
                    train_Y = np.concatenate(train_Y)

                    train_metric = Metric(train_preds, train_Y) 

                    # eval
                    model.eval()
                    valid_preds = []
                    valid_Y = []
                    valid_preds4 = [[], [], [], []]
                    valid_Y4 = [[], [], [], []]
                    for data in tqdm(valid_dataloader):
                        protein_X, protein_node_features, protein_masks, y = [d.cuda() for d in data[1:]]
                        with torch.no_grad():
                            if logit:
                                outputs = model(protein_X, protein_node_features, protein_masks).sigmoid()
                            else:
                                outputs = model(protein_X, protein_node_features, protein_masks)

                        if task == "Metal":
                            for i in range(4):
                                valid_seq_y = torch.masked_select(y[:,i*1000:(i+1)*1000], protein_label_masks[:,i*1000:(i+1)*1000].bool()) # maxlen 1000
                                valid_seq_preds = torch.masked_select(outputs[:,i*1000:(i+1)*1000], protein_label_masks[:,i*1000:(i+1)*1000].bool())
                                valid_Y4[i].append(valid_seq_y.detach().cpu().numpy())
                                valid_preds4[i].append(valid_seq_preds.detach().cpu().numpy())
                        else:
                            valid_seq_y = torch.masked_select(y, protein_masks.bool())
                            valid_seq_preds = torch.masked_select(outputs, protein_masks.bool())

                            valid_Y.append(valid_seq_y.detach().cpu().numpy())
                            valid_preds.append(valid_seq_preds.detach().cpu().numpy())

                    if task == "Metal":
                        valid_metrics = []
                        for i in range(4):
                            valid_preds4[i] = np.concatenate(valid_preds4[i])
                            valid_Y4[i] = np.concatenate(valid_Y4[i])
                            valid_metrics.append(Metric(valid_preds4[i], valid_Y4[i]))
                        valid_metric = np.array(valid_metrics).mean(axis = 0) # [average_AUC, average_AUPR]
                    else:
                        valid_preds = np.concatenate(valid_preds)
                        valid_Y = np.concatenate(valid_Y)
                        valid_metric = Metric(valid_preds, valid_Y)

                    if obj_max * (valid_metric[1]) > obj_max * best_valid_metric: # use AUPRC
                        if len(gpus) > 1:
                            torch.save(model.module.state_dict(), output_path + 'fold%s.ckpt'%fold)
                        else:
                            torch.save(model.state_dict(), output_path + 'fold%s.ckpt'%fold)
                        not_improve_epochs = 0
                        best_valid_metric = valid_metric[1]
                        if task == "Metal":
                            Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, train_auc: %.6f, train_auprc: %.6f, valid_auc: %.6f,%.6f,%.6f,%.6f, valid_auprc:%.6f,%.6f,%.6f,%.6f,'\
                            %(epoch,optimizer._rate,train_loss,train_metric[0],train_metric[1],valid_metrics[0][0],valid_metrics[1][0],valid_metrics[2][0],valid_metrics[3][0],valid_metrics[0][1],valid_metrics[1][1],valid_metrics[2][1],valid_metrics[3][1]))
                        else:
                            Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, train_auc: %.6f, train_auprc: %.6f, valid_auc: %.6f, valid_auprc:%.6f'\
                            %(epoch,optimizer._rate,train_loss,train_metric[0],train_metric[1],valid_metric[0],valid_metric[1]))
                    else:
                        not_improve_epochs += 1
                        if task == "Metal":
                            Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, train_auc: %.6f, train_auprc: %.6f, valid_auc: %.6f,%.6f,%.6f,%.6f, valid_auprc:%.6f,%.6f,%.6f,%.6f, NIE +1 ---> %s'\
                            %(epoch,optimizer._rate,train_loss,train_metric[0],train_metric[1],valid_metrics[0][0],valid_metrics[1][0],valid_metrics[2][0],valid_metrics[3][0],valid_metrics[0][1],valid_metrics[1][1],valid_metrics[2][1],valid_metrics[3][1],not_improve_epochs))
                        else:
                            Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, train_auc: %.6f, train_auprc: %.6f, valid_auc: %.6f, valid_auprc:%.6f, NIE +1 ---> %s'\
                            %(epoch,optimizer._rate,train_loss,train_metric[0],train_metric[1],valid_metric[0],valid_metric[1],not_improve_epochs))
                        if not_improve_epochs >= patience:
                            break


            # 用最好的epoch再测试一下validation，并存下一些结果
            state_dict = torch.load(output_path + 'fold%s.ckpt'%fold, torch.device('cuda' if torch.cuda.is_available() else 'cpu') )

            if model_class.__name__ in ['GraphTrans', "MetalSite"]:  
                model = model_class(node_features, edge_features, hidden_dim, num_encoder_layers, k_neighbors, augment_eps, dropout)

            model.cuda()
            model.load_state_dict(state_dict)
            if len(gpus) > 1:
                model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

            model.eval()

            valid_preds = []
            valid_outputs = [] 
            valid_Y = []
            valid_preds4 = [[], [], [], []]
            valid_Y4 = [[], [], [], []]
            for data in tqdm(valid_dataloader):
                protein_X, protein_node_features, protein_masks, y = [d.cuda() for d in data[1:]]
                with torch.no_grad():
                    if logit:
                        outputs = model(protein_X, protein_node_features, protein_masks).sigmoid()
                    else:
                        outputs = model(protein_X, protein_node_features, protein_masks)

                valid_outputs.append(outputs.detach().cpu().numpy()) # outputs = bz * max_len

                if task == "Metal":
                    for i in range(4):
                        valid_seq_y = torch.masked_select(y[:,i*1000:(i+1)*1000], protein_label_masks[:,i*1000:(i+1)*1000].bool()) # maxlen 1000
                        valid_seq_preds = torch.masked_select(outputs[:,i*1000:(i+1)*1000], protein_label_masks[:,i*1000:(i+1)*1000].bool())
                        valid_Y4[i].append(valid_seq_y.detach().cpu().numpy())
                        valid_preds4[i].append(valid_seq_preds.detach().cpu().numpy())
                else:
                    valid_seq_y = torch.masked_select(y, protein_masks.bool())
                    valid_seq_preds = torch.masked_select(outputs, protein_masks.bool())

                    valid_Y.append(valid_seq_y.detach().cpu().numpy())
                    valid_preds.append(valid_seq_preds.detach().cpu().numpy())

            if task == "Metal":
                valid_metrics = []
                for i in range(4):
                    valid_preds4[i] = np.concatenate(valid_preds4[i])
                    valid_Y4[i] = np.concatenate(valid_Y4[i])
                    valid_metrics.append(Metric(valid_preds4[i], valid_Y4[i]))
                valid_metric = np.array(valid_metrics).mean(axis = 0) # [average_AUC, average_AUPR]
                Write_log(log,'[fold %s] best_valid_auc_avg: %.6f, best_valid_auprc_avg: %.6f, best_valid_auc: %.6f,%.6f,%.6f,%.6f, best_valid_auprc: %.6f,%.6f,%.6f,%.6f,'\
                %(fold, valid_metric[0], valid_metric[1], valid_metrics[0][0],valid_metrics[1][0],valid_metrics[2][0],valid_metrics[3][0],valid_metrics[0][1],valid_metrics[1][1],valid_metrics[2][1],valid_metrics[3][1]))
                all_valid_metric.append(np.array(valid_metrics)[:,1]) # AUPRC
            else:
                valid_preds = np.concatenate(valid_preds)
                valid_Y = np.concatenate(valid_Y)
                valid_metric = Metric(valid_preds, valid_Y)
                Write_log(log,'[fold %s] best_valid_auc: %.6f, best_valid_auprc: %.6f'%(fold, valid_metric[0], valid_metric[1]))
                all_valid_metric.append(valid_metric[1]) # AUPRC

            valid_outputs = np.concatenate(valid_outputs) # shape = (num_samples, max_len) or (num_samples,  4 * max_len)
            if task == "Metal":
                for i in range(4):
                    oof.loc[val_index,metal_list[i]] = [valid_outputs[:,i*1000:(i+1)*1000][j, :len(oof.loc[val_idx,sequence_name])].tolist() for j,val_idx in enumerate(val_index)]
            else:
                oof.loc[val_index,label_name] = [valid_outputs[i, :len(oof.loc[val_idx,sequence_name])].tolist() for i,val_idx in enumerate(val_index)]
            oof.loc[val_index,'fold'] = fold
            train_folds.append(fold)

        if task == "Metal":
            mean_valid_metric = np.mean(all_valid_metric, axis = 0)
            Write_log(log,'all valid mean metric:%.6f,%.6f,%.6f,%.6f, avg metric over tasks:%.6f'%(mean_valid_metric[0],mean_valid_metric[1],mean_valid_metric[2],mean_valid_metric[3],np.mean(mean_valid_metric)))
        else:
            mean_valid_metric = np.mean(all_valid_metric)
            Write_log(log,'all valid mean metric:%.6f'%(mean_valid_metric))
        oof.loc[oof['fold'].isin(train_folds)].to_csv(output_path + 'oof.csv',index=False)


        log_df = pd.DataFrame({'run_id':[run_id],'folds':folds,'metric':[mean_valid_metric],'remark':[config['remark']]})
        if not os.path.exists(output_root + 'experiment_log.csv'):
            log_df.to_csv(output_root + 'experiment_log.csv', index=False)
        else:
            log_df.to_csv(output_root + 'experiment_log.csv',index=False, mode='a', header=None)


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
        for fold in range(folds):
            if not os.path.exists(output_path + 'fold%s.ckpt'%fold):
                continue

            if model_class.__name__ in ['GraphTrans', "MetalSite"]:  
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
        test_outputs = [] 
        test_Y = []
        test_preds4 = [[], [], [], []]
        test_Y4 = [[], [], [], []]
        with torch.no_grad():
            for data in tqdm(test_dataloader):
                protein_X, protein_node_features, protein_masks = [d.cuda() for d in data[1:]]

                if logit:
                    outputs = [model(protein_X, protein_node_features, protein_masks).sigmoid() for model in models]
                else:
                    outputs = [model(protein_X, protein_node_features, protein_maskss) for model in models]

                outputs = torch.stack(outputs,0).mean(0) # 5个模型预测结果求平均,最终shape=(bsize, max_len)
                test_outputs.append(outputs.detach().cpu().numpy())

                if task == "Metal":
                    for i in range(4):
                        test_seq_y = torch.masked_select(y[:,i*1000:(i+1)*1000], protein_label_masks[:,i*1000:(i+1)*1000].bool()) # maxlen 1000
                        test_seq_preds = torch.masked_select(outputs[:,i*1000:(i+1)*1000], protein_label_masks[:,i*1000:(i+1)*1000].bool())
                        test_Y4[i].append(test_seq_y.detach().cpu().numpy())
                        test_preds4[i].append(test_seq_preds.detach().cpu().numpy())
                else:
                    # test_seq_y = torch.masked_select(y, protein_masks.bool())
                    test_seq_preds = torch.masked_select(outputs, protein_masks.bool())

                    test_preds.append(test_seq_preds.cpu().detach().numpy())
                    # test_Y.append(test_seq_y.cpu().detach().numpy())

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
            # test_Y = np.concatenate(test_Y)
            # test_metric = Metric(test_preds, test_Y)



            # Write_log(log,'test_auc:%.6f, test_auprc:%.6f, testFYT_mccL:%.6f'%(test_metric[0],test_metric[1],test_metric[2]))


            


        test_outputs = np.concatenate(test_outputs) # shape = (num_samples, max_len) or (num_samples,  4 * max_len)

        if task == "Metal":
            for i in range(4):
                sub[metal_list[i]] = [test_outputs[:,i*1000:(i+1)*1000][j, :len(sub.loc[j,sequence_name])].tolist() for j in range(len(sub))]
        else:
            sub[label_name] = [test_outputs[i, :len(sub.loc[i,sequence_name])].tolist() for i in range(len(sub))]

        sub.to_csv(output_path + 'submission.csv',index=False)
        log.close()
