import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pickle
from tqdm import tqdm
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder

if __name__ == "__main__":
    protein_embed_dict = {}
    embedder = ProtTransBertBFDEmbedder()
    protein_embed_dict = {}
	path = "./../../Datasets/BindingDB_AIBind/feature/"
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
    with open(path+'aas_ProtTransBertBFD1024.pkl'.format(matrix.shape[0]), 'wb') as f:
        pickle.dump(protein_embed_dict, f)