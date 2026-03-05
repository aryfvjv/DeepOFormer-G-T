#!/usr/bin/python3
# -*-coding:utf-8-*-
# @file:model
# @author:20402
# @date:2026/3/4
# @version:1.0
# @desc:deepOFormer-G/T模型

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepOFormerGT(nn.Module):
    """
    DeepOFormer-G/T 模型
    分支网络：输入工况特征（连续+类别） → 特征嵌入 → Transformer编码 → 全局平均池化 → MLP → 向量b
    主干网络：输入空间坐标(x,y,z) → MLP → 向量t
    输出：b与t的点积 + 偏置
    """
    def __init__(self,
                 cont_features,              # 连续特征名称列表
                 cat_features,                # 类别特征名称列表
                 cat_cardinalities,           # 每个类别特征的类别数列表
                 d_model=64,                  # Transformer嵌入维度
                 nhead=4,                     # 注意力头数
                 num_layers=2,                 # Transformer编码器层数
                 dim_feedforward=128,          # FFN隐藏层维度
                 branch_output_dim=32,         # 分支输出向量维度
                 trunk_input_dim=3,            # 主干输入维度（坐标）
                 trunk_hidden_dim=64,          # 主干隐藏层维度
                 trunk_num_layers=3,            # 主干网络层数
                 use_bias=True):
        super(DeepOFormerGT, self).__init__()

        self.num_cont = len(cont_features)
        self.num_cat = len(cat_features)
        self.d_model = d_model
        self.branch_output_dim = branch_output_dim
        self.use_bias = use_bias

        # ---------- 分支网络 ----------
        # 连续特征投影层：每个特征独立映射到 d_model
        self.cont_proj = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(self.num_cont)
        ])

        # 类别特征嵌入层
        self.cat_embed = nn.ModuleList([
            nn.Embedding(card, d_model) for card in cat_cardinalities
        ])

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分支MLP：将池化后的 d_model 向量映射到 branch_output_dim
        self.branch_mlp = nn.Sequential(
            nn.Linear(d_model, branch_output_dim),
            nn.ReLU(),
            nn.Linear(branch_output_dim, branch_output_dim)
        )

        # ---------- 主干网络 ----------
        trunk_layers = []
        in_dim = trunk_input_dim
        for i in range(trunk_num_layers - 1):
            trunk_layers.append(nn.Linear(in_dim, trunk_hidden_dim))
            trunk_layers.append(nn.ReLU())
            in_dim = trunk_hidden_dim
        trunk_layers.append(nn.Linear(in_dim, branch_output_dim))
        self.trunk = nn.Sequential(*trunk_layers)

        # 偏置项
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('bias', None)

    def forward(self, cont, cat, trunk):
        """
        cont: [batch, num_cont] 连续特征
        cat:  [batch, num_cat]  类别特征（整数索引）
        trunk:[batch, trunk_input_dim] 空间坐标
        """
        # 生成连续特征token
        cont_tokens = []
        for i in range(self.num_cont):
            # 取出第i个特征，形状 [batch, 1]
            feat = cont[:, i:i+1]
            token = self.cont_proj[i](feat)          # [batch, d_model]
            cont_tokens.append(token)

        # 生成类别特征token
        cat_tokens = []
        for i in range(self.num_cat):
            token = self.cat_embed[i](cat[:, i])     # [batch, d_model]
            cat_tokens.append(token)

        # 拼接所有token，形成序列 [batch, seq_len, d_model]
        tokens = torch.stack(cont_tokens + cat_tokens, dim=1)

        # Transformer编码
        encoded = self.transformer(tokens)           # [batch, seq_len, d_model]

        # 全局平均池化
        pooled = encoded.mean(dim=1)                  # [batch, d_model]

        # 分支输出向量 b
        b = self.branch_mlp(pooled)                   # [batch, branch_output_dim]

        # 主干输出向量 t
        t = self.trunk(trunk)                         # [batch, branch_output_dim]

        # 点积 + 偏置
        out = (b * t).sum(dim=-1, keepdim=True)
        if self.bias is not None:
            out = out + self.bias


        return out
