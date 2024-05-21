import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pickle
from tqdm import tqdm
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder
import argparse


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="BindingDB_AIBind",
                       choices=['BindingDB_AIBind', 'BioSNAP', 'BindingDB'],
                       help='the scenario of experiment setting')
    opt = parse.parse_args()
    Dataset = opt.dataset
    protein_embed_dict = {}
    embedder = ProtTransBertBFDEmbedder()
    path = "./../../Datasets/{}/feature/".format(Dataset)
    with open(path + "./protein_list.txt") as file:
        lines = file.readlines()
        lines = tqdm(lines, total=len(lines))
        for line in lines:
            pid, aas = line.strip().split()
            matrix = np.array(embedder.embed(aas))
            vector = np.array(embedder.reduce_per_protein(matrix))
            # print(vector.shape[0])
            # protein_embed_dict[pid] = [vector, matrix]
            protein_embed_dict[pid] = vector
    with open(path+'aas_ProtTransBertBFD1024.pkl', 'wb') as f:
        pickle.dump(protein_embed_dict, f)