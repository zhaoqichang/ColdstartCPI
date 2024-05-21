# -*- coding: utf-8 -*-
"""
@Time:Created on 2021/7/
@author: Qichang Zhao
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import os
import pandas as pd
from model import ColdstartCPI
from dataset import load_dataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import timeit
# from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from bio_embeddings.embed import ProtTransBertBFDEmbedder
import os,pickle


if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument('--batch_size', type=int,default=10,
                       help='Set the batch_size according to the size of your GPU memory.')
    parse.add_argument('--compound_path', type=str, default="./Custom_Data/default/drug_list.txt",
                       help='the path of compounds')
    parse.add_argument('--protein_path', type=str, default="./Custom_Data/default/protein_list.txt",
                       help='the path of proteins')
    parse.add_argument('--identifier', type=int,default="default",
                       help='The identifier of this run.')
    opt = parse.parse_args()
    batch_size = opt.batch_size
    identifier = opt.identifier
    compound_path = opt.compound_path
    protein_path = opt.protein_path

    save_path = "./Prediction/{}/".format(identifier)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("loading input files.")
    print("This will take some time as Mol2Vec and Bio_embedding need to be called to extract the features of compounds and proteins.")
    print("ProtTransBert loading")
    embedder = ProtTransBertBFDEmbedder()

    dataset_load = load_dataset(embedder,compound_path,protein_path,batch_size=batch_size)
    """ create model"""
    model = ColdstartCPI(unify_num=512,head_num=4)
    model.load_state_dict(torch.load('./checkpoint.pth'))
    model = model.cuda()
    data_pbar = tqdm(
            enumerate(
                BackgroundGenerator(dataset_load)),
            total=len(dataset_load))
    Results = []
    print("Predicting...")
    with torch.no_grad():
        for i_batch, input_batch in data_pbar:
            '''data preparation '''
            cids,pids,input_batch = input_batch[0],input_batch[1], input_batch[2:]
            input_batch = [d.cuda() for d in input_batch]
            predicted_scores = model(input_batch)
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_scores = predicted_scores[:, 1]

            for i_pair, cid in enumerate(cids):
                Results.append([cid,pids[i_pair],'%.3f' % predicted_scores[i_pair]])
    print("Saving...")
    Results = pd.DataFrame(Results, columns=["Compound_ID", "Protein_ID", "Score"])
    Results.to_csv(save_path + "{}.csv".format(identifier), index=False)
    print("The predictions are saved in {} folder.".format(save_path))







