# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/8/23 10:10
@author: Qichang Zhao
@Filename: model.py
@Software: PyCharm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.weight_norm import weight_norm

class WOPretrain(nn.Module):
    def __init__(self,unify_num,head_num):
        super(WOPretrain, self).__init__()
        self.drug_embed = nn.Embedding(65, unify_num,padding_idx=0 )
        self.protein_embed = nn.Embedding(26, unify_num,padding_idx=0 )
        self.c_g_unit = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, unify_num),
            nn.PReLU()
        )
        self.c_m_unit = nn.Sequential(
            nn.Linear(unify_num, 300),
            nn.PReLU(),
            nn.Linear(300, unify_num),
            nn.PReLU()
        )

        self.p_g_unit = nn.Sequential(
            nn.Linear(787, 787),
            nn.PReLU(),
            nn.Linear(787, unify_num),
            nn.PReLU()
        )
        self.p_m_unit = nn.Sequential(
            nn.Linear(unify_num, 300),
            nn.PReLU(),
            nn.Linear(300, unify_num),
            nn.PReLU()
        )
        self.Interacting_Layer = nn.TransformerEncoderLayer(unify_num, head_num,batch_first=True)

        self.predict_layer = nn.Sequential(
            nn.Linear(unify_num*2, unify_num*2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(unify_num*2, 2),
        )

    def forward(self, input_tensors):
        c_g_f, c_smiles, p_g_f, p_aas = input_tensors
        bs = c_g_f.shape[0]

        c_m = self.drug_embed(c_smiles)
        p_m = self.protein_embed(p_aas)
        c_size = c_m.shape[1]
        p_size = p_m.shape[1]
        """同一纬度"""
        c_g_f = self.c_g_unit(c_g_f)
        c_m = self.c_m_unit(c_m)
        p_g_f = self.p_g_unit(p_g_f)
        p_m = self.p_m_unit(p_m)

        """特征拼接"""
        unite_m= torch.cat([c_g_f.unsqueeze(1),c_m,p_g_f.unsqueeze(1),p_m],dim = 1)
        # unite_mask = torch.cat([torch.zeros([bs,1]).cuda(), c_mask, torch.zeros([bs,1]).cuda(), p_mask],dim = 1)

        """特征交互"""
        # inter_m = self.Interacting_Layer(unite_m,src_key_padding_mask = unite_mask)
        inter_m = self.Interacting_Layer(unite_m)

        """特征提取"""
        c_g_f = inter_m[:,0]
        p_g_f= inter_m[:,c_size+1]
        inter_f = torch.cat([c_g_f,p_g_f],dim = 1)

        """预测模块"""
        predict = self.predict_layer(inter_f)
        return predict

class WODecouple(nn.Module):
    def __init__(self,unify_num,head_num):
        super(WODecouple, self).__init__()

        self.Interacting_Layer = nn.TransformerEncoderLayer(unify_num, head_num,batch_first=True)

        self.predict_layer = nn.Sequential(
            nn.Linear(unify_num*2, unify_num*2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(unify_num*2, 2),
        )

    def forward(self, input_tensors):
        c_g_f, c_m, p_g_f, p_m = input_tensors
        bs = c_g_f.shape[0]
        c_size = c_m.shape[1]
        p_size = p_m.shape[1]

        """特征拼接"""
        unite_m= torch.cat([c_g_f.unsqueeze(1),c_m,p_g_f.unsqueeze(1),p_m],dim = 1)
        # unite_mask = torch.cat([torch.zeros([bs,1]).cuda(), c_mask, torch.zeros([bs,1]).cuda(), p_mask],dim = 1)

        """特征交互"""
        # inter_m = self.Interacting_Layer(unite_m,src_key_padding_mask = unite_mask)
        inter_m = self.Interacting_Layer(unite_m)

        """特征提取"""
        c_g_f = inter_m[:,0]
        p_g_f= inter_m[:,c_size+1]
        inter_f = torch.cat([c_g_f,p_g_f],dim = 1)

        """预测模块"""
        predict = self.predict_layer(inter_f)
        return predict

class WOTransformer(nn.Module):
    def __init__(self,unify_num):
        super(WOTransformer, self).__init__()
        self.c_g_unit = nn.Sequential(
            nn.Linear(300, 300),
            nn.PReLU(),
            nn.Linear(300, unify_num),
            nn.PReLU()
        )

        self.p_g_unit = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, unify_num),
            nn.PReLU()
        )

        self.predict_layer = nn.Sequential(
            nn.Linear(unify_num*2, unify_num*2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(unify_num*2, 2),
        )

    def forward(self, input_tensors):
        c_g_f, p_g_f = input_tensors

        """同一纬度"""
        c_g_f = self.c_g_unit(c_g_f)
        p_g_f = self.p_g_unit(p_g_f)

        """特征拼接"""
        inter_f = torch.cat([c_g_f,p_g_f],dim = 1)

        """预测模块"""
        predict = self.predict_layer(inter_f)
        return predict

class MolTrans_pretrain(nn.Sequential):
    '''
        Interaction Network with 2D interaction map
    '''

    def __init__(self, ):
        super(MolTrans_pretrain, self).__init__()
        self.max_d = 50
        self.max_p = 545
        self.dropout_rate = 0.1
        self.gpus = torch.cuda.device_count()
        # encoder
        self.flatten_dim = 78192
        # specialized embedding with positional one
        self.icnn = nn.Conv2d(1, 3, 3, padding=0)

        self.drug_decoder = nn.Sequential(
            nn.Linear(300, 384),
            nn.PReLU(),
        )

        self.protein_decoder = nn.Sequential(
            nn.Linear(1024, 384),
            nn.PReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),

            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),

            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),

            # output layer
            nn.Linear(32, 2)
        )

    def forward(self, inputs):
        d, p = inputs
        d = self.drug_decoder(d)
        p = self.protein_decoder(p)
        # # repeat to have the same tensor size for aggregation
        batch_size = d.shape[0]
        d_aug = torch.unsqueeze(d, 2).repeat(1, 1, self.max_p, 1)  # repeat along protein size
        p_aug = torch.unsqueeze(p, 1).repeat(1, self.max_d, 1, 1)  # repeat along drug size

        i = d_aug * p_aug  # interaction
        i_v = i.permute(0,3,1,2)
        # batch_size x embed size x max_drug_seq_len x max_protein_seq_len
        i_v = torch.sum(i_v, dim=1)
        # print(i_v.shape)
        i_v = torch.unsqueeze(i_v, 1)
        # print(i_v.shape)

        i_v = F.dropout(i_v, p=self.dropout_rate)

        # f = self.icnn2(self.icnn1(i_v))
        f = self.icnn(i_v)

        # print(f.shape)

        # f = self.dense_net(f)
        # print(f.shape)

        f = f.view(batch_size, -1)
        # print(f.shape)

        # f_encode = torch.cat((d_encoded_layers[:,-1], p_encoded_layers[:,-1]), dim = 1)

        # score = self.decoder(torch.cat((f, f_encode), dim = 1))
        score = self.decoder(f)
        return score

class DrugBAN_pretrain(nn.Module):
    def __init__(self,drug_dim = 300,protein_dim=1024):
        super(DrugBAN_pretrain, self).__init__()
        num_filters = 128
        mlp_in_dim = 256
        mlp_hidden_dim = 512
        mlp_out_dim = 128
        out_binary = 2
        ban_heads = 2
        self.bcn = weight_norm(
            BANLayer(v_dim=drug_dim, q_dim=protein_dim, h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, inputs):
        v_d, v_p = inputs
        f, att = self.bcn(v_d, v_p)
        score = self.mlp_classifier(f)
        return score

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x



if __name__ == "__main__":
    c_g_f = torch.ones([2,300]).cuda()
    c_m = torch.ones([2,50,300]).cuda()
    p_g_f = torch.ones([2,1024]).cuda()
    p_m = torch.ones([2,545,1024]).cuda()

    model = DrugBAN_pretrain().cuda()
    # or
    # model = MolTrans_pretrain().cuda()
    output = model([c_m, p_m])

    print(output.shape)