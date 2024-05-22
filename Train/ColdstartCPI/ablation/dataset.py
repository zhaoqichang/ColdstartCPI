import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pandas as pd

import os,pickle
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN,dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN,np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)

class WOPretrain_collater():
    """"""
    def __init__(self,drug_f,drug_smiles,protein_f, protein_aas,d_max = 100,p_max = 1000):
        self.drug_f = drug_f
        self.drug_smiles = drug_smiles
        self.protein_f = protein_f
        self.protein_aas = protein_aas
        self.d_max = d_max
        self.p_max = p_max

    def __call__(self,batch_data):
        batch_size = len(batch_data)
        d_g_tensor = []
        d_m_tensor = torch.zeros((batch_size, self.d_max), dtype=torch.long)
        p_g_tensor = []
        p_m_tensor = torch.zeros((batch_size, self.p_max), dtype=torch.long)
        labels_tensor = torch.zeros(batch_size, dtype=torch.long)
        for num, sample in enumerate(batch_data):
            d_id, p_id, label = sample
            d_g_tensor.append(self.drug_f[d_id])
            compoundint = torch.from_numpy(label_smiles(self.drug_smiles[d_id], CHARISOSMISET, self.d_max))
            d_m_tensor[num] = compoundint

            p_g_tensor.append(self.protein_f[p_id])
            proteinint = torch.from_numpy(label_sequence(self.protein_aas[p_id], CHARPROTSET, self.p_max))
            p_m_tensor[num] = proteinint

            labels_tensor[num] = np.int(float(label))

        d_g_tensor = torch.from_numpy(np.array(d_g_tensor)).float()
        p_g_tensor = torch.from_numpy(np.array(p_g_tensor)).float()

        return [d_g_tensor, d_m_tensor,p_g_tensor, p_m_tensor], labels_tensor

def load_WOPretrain1(DATASET, batch_size,i_fold = 0,setting="warm_start"):
    if DATASET=="BindingDB_AIBind":
        drug_without_feature = []
        with open("./../../../Datasets/{}/drug_without_feature.txt".format(DATASET)) as file:
            lines = file.readlines()
            for line in lines:
                drug_without_feature.append(line.split()[0])
        protein_without_feature = []
        with open("./../../../Datasets/{}/protein_without_feature.txt".format(DATASET)) as file:
            lines = file.readlines()
            for line in lines:
                protein_without_feature.append(line.split()[0])
    columns = ['head', 'tail', 'label']
    if setting in ["warm_start", "blind_start"]:
        train_df = pd.read_csv("./../../../Datasets/{}/{}/train_set.csv".format(DATASET, setting))[columns].values
        valid_df = pd.read_csv("./../../../Datasets/{}/{}/val_set.csv".format(DATASET, setting))[columns].values
        test_df = pd.read_csv("./../../../Datasets/{}/{}/test_set.csv".format(DATASET, setting))[columns].values
        train_data_list = train_df
        val_data_list = valid_df
        test_data_list = test_df
    else:
        train_df = pd.read_csv("./../../../Datasets/{}/{}/train_set{}.csv".format(DATASET, setting, i_fold))[
            columns].values
        valid_df = pd.read_csv("./../../../Datasets/{}/{}/val_set{}.csv".format(DATASET, setting, i_fold))[
            columns].values
        test_df = pd.read_csv("./../../../Datasets/{}/{}/test_set{}.csv".format(DATASET, setting, i_fold))[
            columns].values
        if DATASET == "BindingDB_AIBind":
            train_data_list = [pair for pair in train_df if
                               pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
            val_data_list = [pair for pair in valid_df if
                             pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
            test_data_list = [pair for pair in test_df if
                              pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]

    train_set = CustomDataSet(train_data_list)
    val_set = CustomDataSet(val_data_list)
    test_set = CustomDataSet(test_data_list)

    drug_features = load_pickle("./../../../Datasets/{}/feature/compound_ECFP1024.pkl".format(DATASET))
    drug_smiles = load_pickle("./../../../Datasets/{}/feature/compound_smiles.pkl".format(DATASET))
    protein_features = load_pickle("./../../../Datasets/{}/feature/protein_fingerprint787.pkl".format(DATASET))
    protein_aas = load_pickle("./../../../Datasets/{}/feature/protein_aas.pkl".format(DATASET))
    collate_fn = WOPretrain_collater(drug_features, drug_smiles, protein_features, protein_aas)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,
                                    collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4,collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4,collate_fn=collate_fn)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader

class WODecouple_collater():
    """"""
    def __init__(self,drug_f,drug_m,protein_m,d_max = 100,p_max = 1000):
        self.drug_f = drug_f
        self.drug_m = drug_m
        self.protein_m = protein_m
        self.d_max = d_max
        self.p_max = p_max
        self.dim_max = 1024

    def __call__(self,batch_data):
        batch_size = len(batch_data)
        d_g_tensor = []
        d_m_tensor = []
        p_g_tensor = []
        p_m_tensor = []
        labels_tensor = torch.zeros(batch_size, dtype=torch.long)
        for num, sample in enumerate(batch_data):
            d_id, p_id, label = sample
            drug_feature = np.zeros([self.dim_max])
            drug_feature[:300] = self.drug_f[d_id]
            d_g_tensor.append(drug_feature)
            drug_matrix = np.zeros([self.d_max,self.dim_max])
            d_dim = self.drug_m[d_id].shape[0]
            if d_dim <= self.d_max:
                drug_matrix[:d_dim,:300] = self.drug_m[d_id]
            else:
                drug_matrix[:,:300] = self.drug_m[d_id][:self.d_max]
            d_m_tensor.append(drug_matrix)

            p_g_tensor.append(self.protein_m[p_id][0])
            protein_matrix = np.zeros([self.p_max, self.protein_m[p_id][1].shape[1]])
            p_dim = self.protein_m[p_id][1].shape[0]
            if p_dim <= self.p_max:
                protein_matrix[:p_dim] = self.protein_m[p_id][1]
            else:
                protein_matrix = self.protein_m[p_id][1][:self.p_max]
            p_m_tensor.append(protein_matrix)

            labels_tensor[num] = int(float(label))

        d_g_tensor = torch.from_numpy(np.array(d_g_tensor)).float()
        d_m_tensor = torch.from_numpy(np.array(d_m_tensor)).float()
        p_g_tensor = torch.from_numpy(np.array(p_g_tensor)).float()
        p_m_tensor = torch.from_numpy(np.array(p_m_tensor)).float()

        return [d_g_tensor, d_m_tensor,p_g_tensor, p_m_tensor], labels_tensor

def load_WODecouple1(DATASET, batch_size,i_fold = 0,setting="warm_start"):
    columns = ['head', 'tail', 'label']
    if setting in ["warm_start", "blind_start"]:
        train_df = pd.read_csv("./../../../Datasets/{}/{}/train_set.csv".format(DATASET,setting))[columns]
        valid_df = pd.read_csv("./../../../Datasets/{}/{}/val_set.csv".format(DATASET,setting))[columns]
        test_df = pd.read_csv("./../../../Datasets/{}/{}/test_set.csv".format(DATASET,setting))[columns]
    else:
        train_df = pd.read_csv("./../../../Datasets/{}/{}/train_set{}.csv".format(DATASET, setting,i_fold))[columns]
        valid_df = pd.read_csv("./../../../Datasets/{}/{}/val_set{}.csv".format(DATASET, setting,i_fold))[columns]
        test_df = pd.read_csv("./../../../Datasets/{}/{}/test_set{}.csv".format(DATASET, setting,i_fold))[columns]
    train_set = CustomDataSet(train_df.values)
    val_set = CustomDataSet(valid_df.values)
    test_set = CustomDataSet(test_df.values)
    print("Loading features")
    drug_features = load_pickle("./../../../Datasets/{}/feature/compound_Mol2Vec300.pkl".format(DATASET))
    drug_pretrain = load_pickle("./../../../Datasets/{}/feature/compound_Atom2Vec300.pkl".format(DATASET))
    protein_pretrain = load_pickle("./../../../Datasets/{}/feature/aas_ProtTransBertBFD1024.pkl".format(DATASET))
    print("Loaded features")
    collate_fn = WODecouple_collater(drug_features, drug_pretrain, protein_pretrain)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader

class WOTransformer_collater():
    """"""
    def __init__(self,drug_f,protein_m):
        self.drug_f = drug_f
        self.protein_m = protein_m

    def __call__(self,batch_data):
        batch_size = len(batch_data)
        d_g_tensor = []
        p_g_tensor = []
        labels_tensor = torch.zeros(batch_size, dtype=torch.long)
        for num, sample in enumerate(batch_data):
            d_id, p_id, label = sample
            d_g_tensor.append(self.drug_f[d_id])
            p_g_tensor.append(self.protein_m[p_id][0])

            labels_tensor[num] = int(float(label))

        d_g_tensor = torch.from_numpy(np.array(d_g_tensor)).float()
        p_g_tensor = torch.from_numpy(np.array(p_g_tensor)).float()

        return [d_g_tensor,p_g_tensor], labels_tensor

def load_WOTransformer1(DATASET, batch_size,i_fold = 0,setting="warm_start"):
    if DATASET=="BindingDB_AIBind":
        drug_without_feature = []
        with open("./../../../Datasets/{}/drug_without_feature.txt".format(DATASET)) as file:
            lines = file.readlines()
            for line in lines:
                drug_without_feature.append(line.split()[0])
        protein_without_feature = []
        with open("./../../../Datasets/{}/protein_without_feature.txt".format(DATASET)) as file:
            lines = file.readlines()
            for line in lines:
                protein_without_feature.append(line.split()[0])

    columns = ['head', 'tail', 'label']
    if setting in ["warm_start", "blind_start"]:
        train_df = pd.read_csv("./../../../Datasets/{}/{}/train_set.csv".format(DATASET,setting))[columns].values
        valid_df = pd.read_csv("./../../../Datasets/{}/{}/val_set.csv".format(DATASET,setting))[columns].values
        test_df = pd.read_csv("./../../../Datasets/{}/{}/test_set.csv".format(DATASET,setting))[columns].values
        train_data_list = train_df
        val_data_list = valid_df
        test_data_list = test_df

    else:
        train_df = pd.read_csv("./../../../Datasets/{}/{}/train_set{}.csv".format(DATASET, setting,i_fold))[columns].values
        valid_df = pd.read_csv("./../../../Datasets/{}/{}/val_set{}.csv".format(DATASET, setting,i_fold))[columns].values
        test_df = pd.read_csv("./../../../Datasets/{}/{}/test_set{}.csv".format(DATASET, setting,i_fold))[columns].values
        if DATASET=="BindingDB_AIBind":
            train_data_list = [pair for pair in train_df if
                               pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
            val_data_list = [pair for pair in valid_df if
                             pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
            test_data_list = [pair for pair in test_df if
                              pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]

    train_set = CustomDataSet(train_data_list)
    val_set = CustomDataSet(val_data_list)
    test_set = CustomDataSet(test_data_list)

    drug_features = load_pickle("./../../../Datasets/{}/feature/compound_Mol2Vec300.pkl".format(DATASET))
    protein_pretrain = load_pickle("./../../../Datasets/{}/feature/aas_ProtTransBertBFD1024.pkl".format(DATASET))
    collate_fn = WOTransformer_collater(drug_features, protein_pretrain)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,
                                    collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4,collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4,collate_fn=collate_fn)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader

class MolTrans_collater():
    """"""
    def __init__(self,drug_m,protein_m,d_max = 50,p_max = 545):
        self.drug_m = drug_m
        self.protein_m = protein_m
        self.d_max = d_max
        self.p_max = p_max

    def __call__(self,batch_data):
        batch_size = len(batch_data)
        d_m_tensor = []
        p_m_tensor = []
        labels_tensor = torch.zeros(batch_size, dtype=torch.long)
        for num, sample in enumerate(batch_data):
            d_id, p_id, label = sample
            drug_matrix = np.zeros([self.d_max,self.drug_m[d_id].shape[1]])
            d_dim = self.drug_m[d_id].shape[0]
            if d_dim <= self.d_max:
                drug_matrix[:d_dim] = self.drug_m[d_id]
            else:
                drug_matrix = self.drug_m[d_id][:self.d_max]
            d_m_tensor.append(drug_matrix)

            protein_matrix = np.zeros([self.p_max, self.protein_m[p_id][1].shape[1]])
            p_dim = self.protein_m[p_id][1].shape[0]
            if p_dim <= self.p_max:
                protein_matrix[:p_dim] = self.protein_m[p_id][1]
            else:
                protein_matrix = self.protein_m[p_id][1][:self.p_max]
            p_m_tensor.append(protein_matrix)

            labels_tensor[num] = int(float(label))

        d_m_tensor = torch.from_numpy(np.array(d_m_tensor)).float()
        p_m_tensor = torch.from_numpy(np.array(p_m_tensor)).float()

        return [d_m_tensor,p_m_tensor], labels_tensor

def load_MolTrans(DATASET, batch_size,i_fold = 0,setting="warm_start"):
    columns = ['head', 'tail', 'label']
    if setting in ["warm_start", "blind_start"]:
        train_df = pd.read_csv("./../../../Datasets/{}/{}/train_set.csv".format(DATASET,setting))[columns]
        valid_df = pd.read_csv("./../../../Datasets/{}/{}/val_set.csv".format(DATASET,setting))[columns]
        test_df = pd.read_csv("./../../../Datasets/{}/{}/test_set.csv".format(DATASET,setting))[columns]
    else:
        train_df = pd.read_csv("./../../../Datasets/{}/{}/train_set{}.csv".format(DATASET, setting,i_fold))[columns]
        valid_df = pd.read_csv("./../../../Datasets/{}/{}/val_set{}.csv".format(DATASET, setting,i_fold))[columns]
        test_df = pd.read_csv("./../../../Datasets/{}/{}/test_set{}.csv".format(DATASET, setting,i_fold))[columns]
    train_set = CustomDataSet(train_df.values)
    val_set = CustomDataSet(valid_df.values)
    test_set = CustomDataSet(test_df.values)

    drug_pretrain = load_pickle("./../../../Datasets/{}/feature/compound_Atom2Vec300.pkl".format(DATASET))
    protein_pretrain = load_pickle("./../../../Datasets/{}/feature/aas_ProtTransBertBFD1024.pkl".format(DATASET))
    collate_fn = MolTrans_collater(drug_pretrain, protein_pretrain)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,
                                    collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4,collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4,collate_fn=collate_fn)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader

if __name__ == "__main__":

    from prefetch_generator import BackgroundGenerator
    from tqdm import tqdm
    DATASET = "yamanishi_08"
    scenarios = "warm_start_1_1"
    data_path = "./../../Datasets/{}/data_folds/{}/test_fold_1.csv".format(DATASET,scenarios)
    columns = ['head', 'tail', 'label']
    data = pd.read_csv(data_path)[columns].values
    data = CustomDataSet(data)

    drug_features = load_pickle("./../../Datasets/{}/feature/compound_Mol2Vec300.pkl".format(DATASET))
    drug_pretrain = load_pickle("./../../Datasets/{}/feature/compound_Atom2Vec300.pkl".format(DATASET))
    protein_pretrain = load_pickle("./../../Datasets/{}/feature/aas_ProtTransBertBFD1024.pkl".format(DATASET))
    collate_fn = MolTrans_collater(drug_features,drug_pretrain,protein_pretrain)
    dataset_load = DataLoader(data, batch_size=2, shuffle=False, num_workers=0,
                                    collate_fn=collate_fn)
    data_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    for i_batch, i_data in data_pbar:
        '''data preparation '''
        input_tensors, labels_tensor = i_data
        d_g_tensor, d_m_tensor, p_g_tensor, p_m_tensor, \
        d_masks, p_masks = input_tensors
        print(d_m_tensor.shape)