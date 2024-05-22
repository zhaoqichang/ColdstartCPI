# -*- coding: utf-8 -*-
"""
@Time:Created on 2021/7/
@author: Qichang Zhao
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random
from model import MolTrans_pretrain
from dataset import load_MolTrans
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score,precision_recall_curve, auc
from sklearn import metrics
import argparse


def roc_auc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def pr_auc(y, pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc

import os,pickle
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def show_result(DATASET1, DATASET2, Loss_List, Accuracy_List,Precision_List,Recall_List,F1_score_List,AUC_List,AUPR_List):
    Loss_mean, Loss_std = np.mean(Loss_List), np.sqrt(np.var(Loss_List))
    Accuracy_mean, Accuracy_std = np.mean(Accuracy_List), np.sqrt(np.var(Accuracy_List))
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    F1_score_mean, F1_score_var = np.mean(F1_score_List), np.sqrt(np.var(F1_score_List))
    AUC_mean, AUC_std = np.mean(AUC_List), np.sqrt(np.var(AUC_List))
    PRC_mean, PRC_std = np.mean(AUPR_List), np.sqrt(np.var(AUPR_List))
    print("The results on {} of {}:".format(DATASET1,DATASET2))
    with open(save_path + 'results.txt', 'a') as f:
        f.write('{}:'.format(DATASET1) + '\n')
        f.write('Loss(std):{:.4f}({:.4f})'.format(Loss_mean, Loss_std) + '\n')
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_std) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('F1_score(std):{:.4f}({:.4f})'.format(F1_score_mean, F1_score_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_std) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_std) + '\n')
    print('Loss(std):{:.4f}({:.4f})'.format(Loss_mean, Loss_std))
    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_std))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('F1_score(std):{:.4f}({:.4f})'.format(F1_score_mean, F1_score_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_std))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_std))

def test_precess(model,pbar,LOSS):
    model.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            input_batch, labels = data
            labels = labels.cuda()
            input_batch = [d.cuda() for d in input_batch]
            predicted_scores = model(input_batch)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Reacll = recall_score(Y, P)
    F1_score = f1_score(Y, P)
    # AUC = roc_auc_score(Y, S)
    AUC = roc_auc(Y,S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    # PRC = auc(fpr, tpr)
    PRC = pr_auc(Y,S)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)
    return Y, P, test_loss, Accuracy, Precision, Reacll, F1_score, AUC, PRC

def test_model(dataset_load,save_path,DATASET, LOSS,save = False):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test = \
        test_precess(model,test_pbar, LOSS)
    if save:
        with open(save_path + "/{}_prediction.txt".format(DATASET), 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = 'Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};F1 score:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(loss_test, Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test)
    print(results)
    return results,loss_test, Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test

if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument('--scenarios', type=str, default="warm_start",
                       choices=['warm_start', 'compound_cold_start', 'protein_cold_start', 'blind_start'],
                       help='the scenario of experiment setting')
    opt = parse.parse_args()
    scenarios = opt.scenarios

    """select seed"""
    # torch.backends.cudnn.deterministic = True
    # device = torch.device('cuda:0')

    validation = True
    Epoch = 500
    Batch_size = 32
    Learning_rate = 0.0001
    Early_stopping_patience = 25
    """Load preprocessed data."""
    DATASET = "BindingDB_AIBind"

    print("Train on {},{}".format(DATASET,scenarios))
    save_path = "./Results/MolTrans/{}/{}/".format(DATASET,scenarios)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    K_Fold = 5
    Loss_List_train, Accuracy_List_train, Precision_List_train, Recall_List_train, F1_List_train, AUC_List_train, AUPR_List_train = [], [], [], [], [], [], []
    Loss_List_test, Accuracy_List_test, Precision_List_test, Recall_List_test, F1_List_test, AUC_List_test, AUPR_List_test = [], [], [], [], [], [], []

    for i_fold in range(K_Fold):
        SEED = i_fold
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        print('*' * 25, 'No.', i_fold + 1, 'Fold', '*' * 25)
        train_dataset_load, valid_dataset_load, test_dataset_load = load_MolTrans(DATASET, batch_size=Batch_size,i_fold = i_fold,setting=scenarios)

        """ create model"""
        model = MolTrans_pretrain()
        # model = nn.DataParallel(model)
        model = model.cuda()
        Loss = nn.CrossEntropyLoss(weight=None)
        patience = 0
        best_score = 0
        best_epoch = 0

        optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
        if not os.path.exists(save_path + 'valid_best_checkpoint{}.pth'.format(i_fold)):
            """Start training."""
            print('Training...')
            epoch_len = len(str(Epoch))
            for epoch in range(Epoch):
                trian_pbar = tqdm(
                    enumerate(
                        BackgroundGenerator(train_dataset_load)),
                    total=len(train_dataset_load))
                """train"""
                train_losses_in_epoch = []
                model.train()
                for trian_i, train_data in trian_pbar:
                    '''data preparation '''
                    input_batch, trian_labels = train_data
                    input_batch = [d.cuda() for d in input_batch]
                    trian_labels = trian_labels.cuda()

                    optimizer.zero_grad()

                    predicted_interaction = model(input_batch)
                    train_loss = Loss(predicted_interaction, trian_labels)
                    train_losses_in_epoch.append(train_loss.item())
                    train_loss.backward()

                    optimizer.step()
                train_loss_a_epoch = np.average(train_losses_in_epoch)

                """valid"""
                valid_pbar = tqdm(
                    enumerate(
                        BackgroundGenerator(valid_dataset_load)),
                    total=len(valid_dataset_load))
                _,_,valid_loss_a_epoch, _, _, _, _, AUC_dev, PRC_dev = test_precess(model,valid_pbar,Loss)
                valid_score = AUC_dev + PRC_dev


                print_msg = (f'[{epoch + 1:>{epoch_len}}/{Epoch:>{epoch_len}}] ' +
                             f'patience: {patience} ' +
                             f'train_loss: {train_loss_a_epoch:.5f} ' +
                             f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                             f'valid_AUC: {AUC_dev:.5f} ' +
                             f'valid_PRC: {PRC_dev:.5f} '

                             )
                print(print_msg)

                if valid_score > best_score:
                    best_score = valid_score
                    patience = 0
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), save_path + 'valid_best_checkpoint{}.pth'.format(i_fold))

                else:
                    patience += 1


                if patience == Early_stopping_patience:
                    break

        """Test the best model"""
        """load trained model"""
        print('load trained model...')
        model.load_state_dict(torch.load(save_path + 'valid_best_checkpoint{}.pth'.format(i_fold)))

        trainset_test_results, Loss_train, Accuracy_train, Precision_train, Recall_train, F1_score_train, AUC_train, PRC_train = \
            test_model(train_dataset_load, save_path, DATASET, Loss)
        Loss_List_train.append(Loss_train)
        Accuracy_List_train.append(Accuracy_train)
        Precision_List_train.append(Precision_train)
        Recall_List_train.append(Recall_train)
        F1_List_train.append(F1_score_train)
        AUC_List_train.append(AUC_train)
        AUPR_List_train.append(PRC_train)
        with open(save_path + 'results.txt', 'a') as f:
            f.write("The result of train set  on {} fold:".format(i_fold) + trainset_test_results + '\n')

        testset_test_results, Loss_test, Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test = \
            test_model(test_dataset_load, save_path, DATASET, Loss)
        Loss_List_test.append(Loss_test)
        Accuracy_List_test.append(Accuracy_test)
        Precision_List_test.append(Precision_test)
        Recall_List_test.append(Recall_test)
        F1_List_test.append(F1_score_test)
        AUC_List_test.append(AUC_test)
        AUPR_List_test.append(PRC_test)
        with open(save_path + 'results.txt', 'a') as f:
            f.write("best_epoch:{} ".format(best_epoch) + testset_test_results + '\n')

    show_result("Trainset", DATASET, Loss_List_train,
                Accuracy_List_train, Precision_List_train, Recall_List_train, F1_List_train, AUC_List_train,
                AUPR_List_train)

    show_result("Testset", DATASET, Loss_List_test,
                Accuracy_List_test, Precision_List_test, Recall_List_test, F1_List_test, AUC_List_test, AUPR_List_test)





