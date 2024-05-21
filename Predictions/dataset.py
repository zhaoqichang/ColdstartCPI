import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pandas as pd
import os,pickle
from rdkit import Chem
from Mol2Vec.mol2vec.features import mol2alt_sentence, MolSentence, Atom2Substructure
from gensim.models import word2vec
from bio_embeddings.embed import ProtTransBertBFDEmbedder

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
        d_ids = []
        p_ids = []
        d_g_tensor = []
        d_m_tensor = []
        d_masks = []
        p_g_tensor = []
        p_m_tensor = []
        p_masks = []
        for num, sample in enumerate(batch_data):
            d_id, p_id = sample
            d_ids.append(d_id)
            p_ids.append(p_id)
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

        d_g_tensor = torch.from_numpy(np.array(d_g_tensor)).float()
        d_m_tensor = torch.from_numpy(np.array(d_m_tensor)).float()
        p_g_tensor = torch.from_numpy(np.array(p_g_tensor)).float()
        p_m_tensor = torch.from_numpy(np.array(p_m_tensor)).float()
        d_masks = torch.from_numpy(np.array(d_masks))
        p_masks = torch.from_numpy(np.array(p_masks))

        return [d_ids, p_ids, d_g_tensor, d_m_tensor,p_g_tensor, p_m_tensor,\
               d_masks, p_masks]


def load_dataset(compound_path,protein_path, batch_size=1):
    drug_model = word2vec.Word2Vec.load('./../Feature_generation/Mol2Vec/model_300dim.pkl')
    keys = set(drug_model.wv.vocab.keys())
    unseen = 'UNK'
    unseen_vec = drug_model.wv.word_vec(unseen)
    drug_descriptor = {}
    drug_matrix = {}
    with open(compound_path,"r") as file:
        for line in file.readlines():
            cid, smiles = line.strip().split()
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol != None:
                    sentence = MolSentence(mol2alt_sentence(mol, 1))
                    matrix = Atom2Substructure(mol, 1, drug_model, keys, unseen_vec)
                    if matrix.shape[0] > 1:
                        vector = sum([drug_model.wv.word_vec(y) for y in sentence
                                      if y in set(sentence) & keys])
                    else:
                        vector = matrix[0]
                        print("{} is too small\n".format(cid))
                    if type(vector) == int:
                        print("{} has no feature\n".format(cid))
                    drug_descriptor[cid] = vector
                    drug_matrix[cid] = matrix
                else:
                    print("RDKit don't read {}".format(cid))
            except Exception as e:
                print(cid, e)

    protein_embed_dict = {}
    embedder = ProtTransBertBFDEmbedder()
    with open(protein_path,"r") as file:
        for line in file.readlines():
            pid, aas = line.strip().split()
            matrix = np.array(embedder.embed(aas))
            vector = np.array(embedder.reduce_per_protein(matrix))
            # print(vector.shape[0])
            # protein_embed_dict[pid] = [vector, matrix]
            protein_embed_dict[pid] = vector

    data_list = []
    for pid in protein_embed_dict.keys():
        for cid in drug_descriptor.keys():
            data_list.append([cid,pid])
    data = CustomDataSet(data_list)

    collate_fn = collater_embeding(drug_descriptor, drug_matrix, protein_embed_dict,p_max = 1000)

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4,
                                    collate_fn=collate_fn)
    print("Number of samples in the data: ", len(data_loader))

    return data_loader

if __name__ == "__main__":

    from prefetch_generator import BackgroundGenerator
    from tqdm import tqdm
    identifier = "default"
    compound_path = None
    protein_path = None
    dataset_load = load_dataset(compound_path,protein_path)
    data_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    for i_batch, i_data in data_pbar:
        '''data preparation '''
        cids,pids = i_data[0],i_data[1]
        print(cids.shape)