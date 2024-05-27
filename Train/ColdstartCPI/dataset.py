import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)

class collater_embeding():
    """"""
    def __init__(self,drug_f,drug_m,protein_m,d_max = 100,p_max = 1000):
        self.drug_f = drug_f
        self.drug_m = drug_m
        self.protein_m = protein_m
        self.d_max = d_max
        self.p_max = p_max

    def __call__(self,batch_data):
        batch_size = len(batch_data)
        d_g_tensor = []
        d_m_tensor = []
        d_masks = []
        p_g_tensor = []
        p_m_tensor = []
        p_masks = []
        labels_tensor = torch.zeros(batch_size, dtype=torch.long)
        for num, sample in enumerate(batch_data):
            d_id, p_id, label = sample
            d_g_tensor.append(self.drug_f[d_id])
            drug_matrix = np.zeros([self.d_max,self.drug_m[d_id].shape[1]])
            d_mask = np.zeros([self.d_max])
            d_dim = self.drug_m[d_id].shape[0]
            if d_dim <= self.d_max:
                drug_matrix[:d_dim] = self.drug_m[d_id]
                d_mask[d_dim:] = 1
            else:
                drug_matrix = self.drug_m[d_id][:self.d_max]
            d_m_tensor.append(drug_matrix)
            d_masks.append(d_mask==1)

            p_g_tensor.append(self.protein_m[p_id][0])
            protein_matrix = np.zeros([self.p_max, self.protein_m[p_id][1].shape[1]])
            p_mask = np.zeros([self.p_max])
            p_dim = self.protein_m[p_id][1].shape[0]
            if p_dim <= self.p_max:
                protein_matrix[:p_dim] = self.protein_m[p_id][1]
                p_mask[p_dim:] = 1
            else:
                protein_matrix = self.protein_m[p_id][1][:self.p_max]
            p_m_tensor.append(protein_matrix)
            p_masks.append(p_mask==1)

            labels_tensor[num] = int(float(label))

        d_g_tensor = torch.from_numpy(np.array(d_g_tensor)).float()
        d_m_tensor = torch.from_numpy(np.array(d_m_tensor)).float()
        p_g_tensor = torch.from_numpy(np.array(p_g_tensor)).float()
        p_m_tensor = torch.from_numpy(np.array(p_m_tensor)).float()
        d_masks = torch.from_numpy(np.array(d_masks))
        p_masks = torch.from_numpy(np.array(p_masks))

        return [d_g_tensor, d_m_tensor,p_g_tensor, p_m_tensor,\
               d_masks, p_masks], labels_tensor

import os,pickle
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def split_train_valid(data_df, fold, val_ratio=0.1):
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=range(len(data_df)), y=data_df['label'])))

    train_df = data_df.iloc[train_index]
    val_df = data_df.iloc[val_index]

    return train_df, val_df

def load_scenario_dataset(DATASET,setting,i, batch_size):
    columns = ['head', 'tail', 'label']
    train_df = pd.read_csv("./../../Datasets/{}/{}/train_set{}.csv".format(DATASET,setting,i))[columns]
    valid_df = pd.read_csv("./../../Datasets/{}/{}/valid_set{}.csv".format(DATASET,setting,i))[columns]
    test_df = pd.read_csv("./../../Datasets/{}/{}/test_set{}.csv".format(DATASET,setting,i))[columns]
    train_set = CustomDataSet(train_df.values)
    val_set = CustomDataSet(valid_df.values)
    test_set = CustomDataSet(test_df.values)
    try:
        drug_features = load_pickle("./../../Datasets/{}/feature/compound_Mol2Vec300.pkl".format(DATASET))
        drug_pretrain = load_pickle("./../../Datasets/{}/feature/compound_Atom2Vec300.pkl".format(DATASET))
        protein_pretrain = load_pickle("./../../Datasets/{}/feature/aas_ProtTransBertBFD1024.pkl".format(DATASET))
    except:
        print("Pre-training features for compounds and proteins are not found in the {}/feature folder, \n\
        please check the file naming or run Mol2Vec.py and generator.py first.".format(DATASET))
        raise
    collate_fn = collater_embeding(drug_features, drug_pretrain, protein_pretrain)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                    collate_fn=collate_fn,pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,collate_fn=collate_fn)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader

def load_Miss_dataset(DATASET,miss_rate, batch_size, fold=0):
    columns = ['head', 'tail', 'label']
    full_df = pd.read_csv("./../../Datasets/{}/full_pair.csv".format(DATASET))[columns]
    train_df, valid_test_df = split_train_valid(full_df, fold=fold,val_ratio=miss_rate/100)
    test_df, val_df = split_train_valid(valid_test_df, fold=fold, val_ratio=0.1)
    train_set = CustomDataSet(train_df.values)
    val_set = CustomDataSet(val_df.values)
    test_set = CustomDataSet(test_df.values)
    try:
        drug_features = load_pickle("./../../Datasets/{}/feature/compound_Mol2Vec300.pkl".format(DATASET))
        drug_pretrain = load_pickle("./../../Datasets/{}/feature/compound_Atom2Vec300.pkl".format(DATASET))
        protein_pretrain = load_pickle("./../../Datasets/{}/feature/aas_ProtTransBertBFD1024.pkl".format(DATASET))
    except:
        print("Pre-training features for compounds and proteins are not found in the {}/feature folder, \n\
        please check the file naming or run Mol2Vec.py and generator.py first.".format(DATASET))
        raise
    collate_fn = collater_embeding(drug_features, drug_pretrain, protein_pretrain)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                    collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader

"""load BindingDB of AIBind datasets"""
def load_BindingDB_AIBind_dataset(DATASET,scenarios, batch_size, fold=0):
    drug_without_feature = []
    with open("./../../Datasets/{}/drug_without_feature.txt".format(DATASET)) as file:
        lines = file.readlines()
        for line in lines:
            drug_without_feature.append(line.split()[0])
    protein_without_feature = []
    with open("./../../Datasets/{}/protein_without_feature.txt".format(DATASET)) as file:
        lines = file.readlines()
        for line in lines:
            protein_without_feature.append(line.split()[0])
    columns = ['head', 'tail', 'label']
    print("load data")
    train_data_list = pd.read_csv("./../../Datasets/{}/{}/train_set{}.csv".format(DATASET, scenarios, fold))[columns].values
    val_data_list = pd.read_csv("./../../Datasets/{}/{}/val_set{}.csv".format(DATASET, scenarios, fold))[columns].values
    test_data_list = pd.read_csv("./../../Datasets/{}/{}/test_set{}.csv".format(DATASET, scenarios, fold))[columns].values
    train_data_list = [pair for pair in train_data_list if pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
    val_data_list = [pair for pair in val_data_list if pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
    test_data_list = [pair for pair in test_data_list if pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
    print("data process done")
    train_set = CustomDataSet(train_data_list)
    val_set = CustomDataSet(val_data_list)
    test_set = CustomDataSet(test_data_list)
    print("load feature")
    try:
        drug_features = load_pickle("./../../Datasets/{}/feature/compound_Mol2Vec300.pkl".format(DATASET))
        drug_pretrain = load_pickle("./../../Datasets/{}/feature/compound_Atom2Vec300.pkl".format(DATASET))
        protein_pretrain = load_pickle("./../../Datasets/{}/feature/aas_ProtTransBertBFD1024.pkl".format(DATASET))
    except:
        print("Pre-training features for compounds and proteins are not found in the {}/feature folder, \n\
        please check the file naming or run Mol2Vec.py and generator.py first.".format(DATASET))
        raise
    collate_fn = collater_embeding(drug_features, drug_pretrain, protein_pretrain)
    print("feature load done")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                    collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader

def load_BindingDB_AIBind_Miss_dataset(DATASET,miss_rate, batch_size, fold=0):
    drug_without_feature = []
    with open("./../../Datasets/{}/drug_without_feature.txt".format(DATASET)) as file:
        lines = file.readlines()
        for line in lines:
            drug_without_feature.append(line.split()[0])
    protein_without_feature = []
    with open("./../../Datasets/{}/protein_without_feature.txt".format(DATASET)) as file:
        lines = file.readlines()
        for line in lines:
            protein_without_feature.append(line.split()[0])
    columns = ['head', 'tail', 'label']
    full_df = pd.read_csv("./../../Datasets/{}/full_pair.csv".format(DATASET))[columns]
    train_df, valid_test_df = split_train_valid(full_df, fold=fold,val_ratio=miss_rate/100)
    test_df, val_df = split_train_valid(valid_test_df, fold=fold, val_ratio=0.1)
    train_data_list = train_df.values
    val_data_list = val_df.values
    test_data_list = test_df.values
    train_data_list = [pair for pair in train_data_list if
                       pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
    val_data_list = [pair for pair in val_data_list if
                     pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
    test_data_list = [pair for pair in test_data_list if
                      pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
    train_set = CustomDataSet(train_data_list)
    val_set = CustomDataSet(val_data_list)
    test_set = CustomDataSet(test_data_list)
    try:
        drug_features = load_pickle("./../../Datasets/{}/feature/compound_Mol2Vec300.pkl".format(DATASET))
        drug_pretrain = load_pickle("./../../Datasets/{}/feature/compound_Atom2Vec300.pkl".format(DATASET))
        protein_pretrain = load_pickle("./../../Datasets/{}/feature/aas_ProtTransBertBFD1024.pkl".format(DATASET))
    except:
        print("Pre-training features for compounds and proteins are not found in the {}/feature folder, \n\
        please check the file naming or run Mol2Vec.py and generator.py first.".format(DATASET))
        raise
    collate_fn = collater_embeding(drug_features, drug_pretrain, protein_pretrain)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                    collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    from prefetch_generator import BackgroundGenerator
    from tqdm import tqdm
    DATASET = "BindingDB_AIBind"
    scenarios = "warm_start"
    dataset_load,_,_ = load_BindingDB_AIBind_dataset(DATASET,scenarios, 32, fold=0)
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