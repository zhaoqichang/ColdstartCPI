import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

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

            labels_tensor[num] = float(label)

        d_g_tensor = torch.from_numpy(np.array(d_g_tensor)).float()
        d_m_tensor = torch.from_numpy(np.array(d_m_tensor)).float()
        p_g_tensor = torch.from_numpy(np.array(p_g_tensor)).float()
        p_m_tensor = torch.from_numpy(np.array(p_m_tensor)).float()
        d_masks = torch.from_numpy(np.array(d_masks))
        p_masks = torch.from_numpy(np.array(p_masks))

        return [d_g_tensor, d_m_tensor,p_g_tensor, p_m_tensor,\
               d_masks, p_masks], labels_tensor

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_dataset(batch_size):
    data_set = []
    with open("./Dataset/demo_data.txt", "r") as file:
        for pair in file.readlines():
            cid, pid, label = pair.strip().split()
            data_set.append([cid, pid, label])
    train_set = CustomDataSet(data_set)
    val_set = CustomDataSet(data_set)
    test_set = CustomDataSet(data_set)

    drug_features = load_pickle("./Dataset/feature/compound_Mol2Vec300.pkl")
    drug_pretrain = load_pickle("./Dataset/feature/compound_Atom2Vec300.pkl")
    protein_pretrain = load_pickle("./Dataset/feature/aas_ProtTransBertBFD1024.pkl")
    collate_fn = collater_embeding(drug_features, drug_pretrain, protein_pretrain)
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
    train_loader, val_loader, test_loader = load_dataset(32)
    data_pbar = tqdm(
        enumerate(
            BackgroundGenerator(train_loader)),
        total=len(train_loader))
    for i_batch, i_data in data_pbar:
        '''data preparation '''
        input_tensors, labels_tensor = i_data
        d_g_tensor, d_m_tensor, p_g_tensor, p_m_tensor, \
        d_masks, p_masks = input_tensors
        print(d_m_tensor.shape)