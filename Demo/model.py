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

class ColdstartCPI(nn.Module):
    def __init__(self,unify_num,head_num):
        super(ColdstartCPI, self).__init__()
        self.c_g_unit = nn.Sequential(
            nn.Linear(300, 300),
            nn.PReLU(),
            nn.Linear(300, unify_num),
            nn.PReLU()
        )
        self.c_m_unit = nn.Sequential(
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
        self.p_m_unit = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, unify_num),
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
        c_g_f, c_m, p_g_f, p_m, c_mask, p_mask = input_tensors
        bs = c_g_f.shape[0]
        c_size = c_m.shape[1]
        p_size = p_m.shape[1]

        c_g_f = self.c_g_unit(c_g_f)
        c_m = self.c_m_unit(c_m)
        p_g_f = self.p_g_unit(p_g_f)
        p_m = self.p_m_unit(p_m)

        unite_m= torch.cat([c_g_f.unsqueeze(1),c_m,p_g_f.unsqueeze(1),p_m],dim = 1)

        inter_m = self.Interacting_Layer(unite_m)

        c_g_f = inter_m[:,0]
        p_g_f= inter_m[:,c_size+1]
        inter_f = torch.cat([c_g_f,p_g_f],dim = 1)
        predict = self.predict_layer(inter_f)
        return predict


if __name__ == "__main__":
    c_g_f = torch.ones([2,300]).cuda()
    c_m = torch.ones([2,16,300]).cuda()
    p_g_f = torch.ones([2,1024]).cuda()
    p_m = torch.ones([2,21,1024]).cuda()
    c_mask = torch.zeros([2,16]).cuda()
    p_mask = torch.zeros([2,21]).cuda()

    model = ColdstartCPI(100,4).cuda()
    output = model(c_g_f,c_m,p_g_f,p_m, c_mask, p_mask)
    print(output.shape)