import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

Dataset_Path = './datasets/'
Feature_Path = './feature/'

NODE_DIM = 1024 + 14
max_len = 869 # within train & tests


def get_pdb_xyz(pdb_file):
    current_pos = -1000
    X = []
    current_aa = {} # 'N', 'CA', 'C', 'O'
    for line in pdb_file:
        if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
            if current_aa != {}:
                X.append(current_aa["CA"]) # X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"]])
                current_aa = {}
            if line[0:4].strip() != "TER":
                current_pos = int(line[22:26].strip())

        if line[0:4].strip() == "ATOM":
            atom = line[13:16].strip()
            if atom in ['N', 'CA', 'C', 'O']:
                xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                current_aa[atom] = xyz
    return np.array(X)


def prepare_features(pdb_id, dataset):
    with open(Dataset_Path + "pdb/" + pdb_id + ".pdb", "r") as f:
        X = get_pdb_xyz(f.readlines()) # [L, 3]

    protrans = np.load(Feature_Path + f'ProtTrans/{pdb_id}.npy')
    dssp = np.load(Feature_Path + f'dssp/{pdb_id}.npy') ## 107,14
#     print(dssp, dssp.shape)
#     print('####')
#     print(protrans, protrans.shape)
    node_features = np.hstack([protrans, dssp])

    # Padding
    padded_X = np.zeros((max_len, 3))
    padded_X[:X.shape[0]] = X
    padded_X = torch.tensor(padded_X, dtype = torch.float)

    padded_node_features = np.zeros((max_len, NODE_DIM))
    padded_node_features[:node_features.shape[0]] = node_features
    padded_node_features = torch.tensor(padded_node_features, dtype = torch.float)

    masks = np.zeros(max_len)
    masks[:X.shape[0]] = 1
    masks = torch.tensor(masks, dtype = torch.long)

#     padded_y = np.zeros(max_len)
#     y = dataset[pdb_id][1] ## 
#     padded_y[:X.shape[0]] = y
#     padded_y = torch.tensor(padded_y, dtype = torch.float)

    # Save
    torch.save(padded_X, Feature_Path + f'input_ProtTrans_dssp/{pdb_id}_X.tensor')
    torch.save(padded_node_features, Feature_Path + f'input_ProtTrans_dssp/{pdb_id}_node_feature.tensor')
    torch.save(masks, Feature_Path + f'input_ProtTrans_dssp/{pdb_id}_mask.tensor')
    # torch.save(padded_y, Feature_Path + f'input_ProtTrans_dssp/{pdb_id}_label.tensor')


def pickle2csv(metal_name):
    with open(Dataset_Path + metal_name + "_train.pkl", "rb") as f:
        train = pickle.load(f)

    train_IDs, train_sequences, train_labels = [], [], []
    for ID in train:
        train_IDs.append(ID)
        item = train[ID]
        train_sequences.append(item[0])
        train_labels.append(item[1])

    train_dic = {"ID": train_IDs, "sequence": train_sequences, "label": train_labels}
    train_dataframe = pd.DataFrame(train_dic)
    train_dataframe.to_csv(Dataset_Path + metal_name + '_train.csv', index=False)

    with open(Dataset_Path + metal_name + "_test.pkl", "rb") as f:
        test = pickle.load(f)

    test_IDs, test_sequences, test_labels = [], [], []
    for ID in test:
        test_IDs.append(ID)
        item = test[ID]
        test_sequences.append(item[0])
        test_labels.append(item[1])

    test_dic = {"ID": test_IDs, "sequence": test_sequences, "label": test_labels}
    test_dataframe = pd.DataFrame(test_dic)
    test_dataframe.to_csv(Dataset_Path + metal_name + '_test.csv', index=False)



if __name__ == '__main__':
    # pickle2csv("ZN")

#     with open(Dataset_Path + "ZN_train.pkl", "rb") as f:
#         ZN_train = pickle.load(f)
#     for ID in tqdm(ZN_train):
#         prepare_features(ID, ZN_train)

#     with open(Dataset_Path + "ZN_test.pkl", "rb") as f:
#         ZN_test = pickle.load(f)
#     for ID in tqdm(ZN_test):
#         prepare_features(ID, ZN_test)

    metal_train_native = {}
    with open("./datasets/test_protein.fa", "rb") as f:
        lines = f.readlines()
        length = len(lines)
        N = int(length/2)
        for i in range(N):
            line = str(lines[2*i]).rstrip('\n')
            idx = line[3:-3]
            seq = str(lines[2*i+1])
            metal_train_native[idx] = seq

    for ID in metal_train_native:
        prepare_features(ID, metal_train_native[ID][0])