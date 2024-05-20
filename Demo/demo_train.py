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
from model import ColdstartCPI
from dataset import load_dataset
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score,precision_recall_curve, auc
from sklearn import metrics

def roc_auc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def pr_auc(y, pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc

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

def test_model(dataset_load, LOSS):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test = \
        test_precess(model,test_pbar, LOSS)
    results = 'Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};F1 score:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(loss_test, Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test)
    print(results)
    return results,loss_test, Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test

if __name__ == "__main__":
    """select seed"""
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    """parameter"""
    validation = True
    Epoch = 100
    Batch_size = 16
    Learning_rate = 0.0001
    Early_stopping_patience = 5
    """Load preprocessed data."""
    print("Training")
    save_path = "./Results/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_dataset_load, valid_dataset_load, test_dataset_load = load_dataset(batch_size=Batch_size)

    """ create model"""
    model = ColdstartCPI(unify_num=128,head_num=2)
    model = model.cuda()
    Loss = nn.CrossEntropyLoss(weight=None)
    patience = 0
    best_score = 0
    best_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
    if not os.path.exists(save_path + 'valid_best_checkpoint.pth'):
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
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                optimizer.step()
            train_loss_a_epoch = np.average(train_losses_in_epoch)
            # writer.add_scalar('Train Loss/{}'.format(i_fold), train_loss_a_epoch, epoch)

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
                         f'valid_PRC: {PRC_dev:.5f} '
                         )
            print(print_msg)

            if valid_score > best_score:
                best_score = valid_score
                patience = 0
                best_epoch = epoch + 1
                torch.save(model.state_dict(), save_path + 'valid_best_checkpoint.pth')
            else:
                patience += 1
            if patience == Early_stopping_patience:
                break

    """Test the best model"""
    """load trained model"""
    print('load trained model...')
    model.load_state_dict(torch.load(save_path + 'valid_best_checkpoint.pth'))

    trainset_test_results, Loss_train, Accuracy_train, Precision_train, Recall_train, F1_score_train, AUC_train, PRC_train = \
        test_model(train_dataset_load, Loss)
    with open(save_path + 'results.txt', 'a') as f:
        f.write("The result of train set:"+ trainset_test_results + '\n')

    testset_test_results, Loss_test, Accuracy_test, Precision_test, Recall_test, F1_score_test, AUC_test, PRC_test = \
        test_model(test_dataset_load, Loss)

    with open(save_path + 'results.txt', 'a') as f:
        f.write("The result of test set:" + testset_test_results + '\n')



