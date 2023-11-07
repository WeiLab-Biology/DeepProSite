import os, pickle
import numpy as np
from Bio import pairwise2

software_path = "./Software/"


def process_dssp(dssp_file):
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    with open(dssp_file, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        seq += aa
        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        SS_vec = np.zeros(9) # The last dim represents "Unknown" for missing residues
        SS_vec[SS_type.find(SS)] = 1
        PHI = float(lines[i][103:109].strip())
        PSI = float(lines[i][109:115].strip())
        ACC = float(lines[i][34:38].strip())
        ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
        dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

    return seq, dssp_feature


def match_dssp(seq, dssp, ref_seq):
    alignments = pairwise2.align.globalxx(ref_seq, seq)
    ref_seq = alignments[0].seqA
    seq = alignments[0].seqB

    SS_vec = np.zeros(9) # The last dim represent "Unknown" for missing residues
    SS_vec[-1] = 1
    padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))

    new_dssp = []
    for aa in seq:
        if aa == "-":
            new_dssp.append(padded_item)
        else:
            new_dssp.append(dssp.pop(0))

    matched_dssp = []
    for i in range(len(ref_seq)):
        if ref_seq[i] == "-":
            continue
        matched_dssp.append(new_dssp[i])

    return matched_dssp


def transform_dssp(dssp_feature):
    dssp_feature = np.array(dssp_feature)
    angle = dssp_feature[:,0:2]
    ASA_SS = dssp_feature[:,2:]

    radian = angle * (np.pi / 180)
    dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis = 1)

    return dssp_feature


def get_dssp(ID, ref_seq):
    os.system("{}dssp-2.0.4/mkdssp -i ./datasets/pdb/{}.pdb -o ./feature/dssp/{}.dssp".format(software_path, ID, ID))
    dssp_seq, dssp_matrix = process_dssp("./feature/dssp/" + ID + ".dssp")
    if dssp_seq != ref_seq:
        dssp_matrix = match_dssp(dssp_seq, dssp_matrix, ref_seq)

    np.save("./feature/dssp/" + ID, transform_dssp(dssp_matrix))
    os.system("rm ./feature/dssp/" + ID + ".dssp")

metal_train_native = {}
with open("./datasets/test_protein.fa", "rb") as f:
    lines = f.readlines()
    length = len(lines)
    N = int(length/2)
    for i in range(N):
        line = str(lines[2*i]).rstrip('\n')
        idx = line[3:-3]
        seq = str(lines[2*i+1])
        metal_train_native[idx] = seq[2:-3]

for ID in metal_train_native:
    get_dssp(ID, metal_train_native[ID])
