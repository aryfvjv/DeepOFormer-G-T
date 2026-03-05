#!/usr/bin/python3
# -*-coding:utf-8-*-
# @file:utils
# @author:20402
# @date:2026/3/4
# @version:1.0
# @desc:工具函数
import numpy as np
import torch
from sklearn.metrics import  r2_score,mean_absolute_error,mean_squared_error



def ml2re_loss(pred,target,eps=1e-8):
    """
    均值L2相对误差(mean L2 relative Error)
    loss = sqrt(mean((pred-target)/(target+eps))^2)
    :param pred:
    :param target:
    :param eps:
    :return:
    """
    relative_error = (pred-target)/(torch.abs(target) + eps)
    squared = relative_error ** 2
    return torch.sqrt(torch.mean(squared))

def evaluate_model(model,loader,device,return_pred=False):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            cont = batch['cont'].to(device)
            cat = batch['cat'].to(device)
            trunk = batch['trunk'].to(device)
            target = batch['target'].to(device)

            output = model(cont,cat,trunk)
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    preds = np.concatenate(all_preds,axis=0).flatten()
    targets = np.concatenate(all_targets,axis=0).flatten()

    # 训练时使用了log，因此preds和targets都是log值
    # 计算原始值的相对误差时，需要exp
    preds_orig = np.exp(preds)
    targets_orig = np.exp(targets)

    mae = mean_absolute_error(targets_orig, preds_orig)
    rmse = np.sqrt(mean_squared_error(targets_orig, preds_orig))
    r2 = r2_score(targets_orig, preds_orig)

    # 相对误差
    relative_error = np.abs(preds_orig - targets_orig) / (targets_orig + 1e-8)
    mre = np.mean(relative_error)
    rmre = np.sqrt(np.mean(relative_error ** 2))

    if return_pred:
        return mae, rmse, r2, mre, rmre, preds_orig, targets_orig
    else:

        return mae, rmse, r2, mre, rmre
