import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import sys, os
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec, Atom2Substructure
from gensim.models import word2vec
import copy,pickle
from tqdm import tqdm

Drugs ={}
with open("./drug_smiles.txt","r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip().split()
        Drugs[line[0]] = line[1]
model = word2vec.Word2Vec.load('./model_300dim.pkl')
keys = set(model.wv.vocab.keys())
unseen='UNK'
unseen_vec = model.wv.word_vec(unseen)
drug_descriptor= {}
drug_matrix= {}
max_num = 0
for drug_id in tqdm(Drugs.keys(),total=len(Drugs)):
    try:
        mol = Chem.MolFromSmiles(Drugs[drug_id])
        if mol != None:
            sentence = MolSentence(mol2alt_sentence(mol, 1))
            matrix = Atom2Substructure(mol, 1, model, keys, unseen_vec)
            if matrix.shape[0]>1:
                vector = sum([model.wv.word_vec(y) for y in sentence
                              if y in set(sentence) & keys])
            else:
                vector = matrix[0]
                print("{} is too small\n".format(drug_id))
            if type(vector)==int:
                print("{} has no feature\n".format(drug_id))
            drug_descriptor[drug_id] = vector
            drug_matrix[drug_id] = matrix
        else:
            print("RDKit don't read {}".format(drug_id))
    except Exception as e:
        print(drug_id, e)
with open('compound_Mol2Vec300.pkl', 'wb') as f:
    pickle.dump(drug_descriptor, f)
with open('compound_Atom2Vec300.pkl', 'wb') as f:
    pickle.dump(drug_matrix, f)
