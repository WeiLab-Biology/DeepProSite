# -*- coding: utf-8 -*-
###YQM
# run on gpu2
import pickle
import numpy as np
import torch
from tqdm import tqdm

raw_protrans_path = "./feature/rawembedd/"
protrans_output_path = "./feature/ProtTrans/"

Max_protrans = []
Min_protrans = []


# ######### Get Min and Max ##########

# with open("./Dataset_335_60/Train_335.pkl", "rb") as f:
#     GO_train = pickle.load(f)

# for i, ID in tqdm(enumerate(GO_train)):
#     raw_protrans = np.load(raw_protrans_path + ID + ".npy")
#     Max_protrans.append(np.max(raw_protrans, axis = 0))
#     Min_protrans.append(np.min(raw_protrans, axis = 0))
#     if i == len(GO_train) - 1:
#         Max_protrans = np.max(np.array(Max_protrans), axis = 0)
#         Min_protrans = np.min(np.array(Min_protrans), axis = 0)
#     elif i % 5000 == 0:
#         Max_protrans = [np.max(np.array(Max_protrans), axis = 0)]
#         Min_protrans = [np.min(np.array(Min_protrans), axis = 0)]

# np.save("Max_ProtTrans_repr", Max_protrans)
# np.save("Min_ProtTrans_repr", Min_protrans)
Max_protrans = np.load("./Max_ProtTrans_repr.npy")
Min_protrans = np.load("./Min_ProtTrans_repr.npy")

# ######### Normalize Feature ##########

# with open("../../others/multi-task/Dataset/GO_valid.pkl", "rb") as f:
#     GO_valid = pickle.load(f)

# with open("../../example/test_protein.fa      Dataset_335_60/Test_60.pkl", "rb") as f:

all_ID = []
with open("./datasets/test_protein.fa", "rb") as f:
    lines = f.readlines()
    length = len(lines)
    N = int(length/2)
    for i in range(N):
        line = str(lines[2*i]).rstrip('\n')
        idx = line[3:-3]
        all_ID.append(idx)
    
#all_ID = list(GO_train.keys()) + list(GO_valid.keys()) + list(GO_test.keys())
# all_ID = list(GO_train.keys()) + list(GO_test.keys())    ####只留下train和test
for ID in tqdm(all_ID):
    raw_protrans = np.load(raw_protrans_path + ID + ".npy")
    protrans = (raw_protrans - Min_protrans) / (Max_protrans - Min_protrans)
    torch.save(torch.tensor(protrans, dtype = torch.float), protrans_output_path + ID + '.tensor')
    
    #1129我自己加的
    np.save(protrans_output_path + ID + '.npy', protrans)
