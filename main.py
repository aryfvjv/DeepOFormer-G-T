#!/usr/bin/python3
# -*-coding:utf-8-*-
# @file:main
# @author:20402
# @date:2026/2/27
# @version:2.0
# @desc: 包含敏感性分析和绘图

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd

from dataset import get_dataloaders
from utils import ml2re_loss, evaluate_model
from model import DeepOFormerGT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA not available, using CPU (training may be slow)')

def train_one_epoch(model, loader, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        cont = batch['cont'].to(device)
        cat = batch['cat'].to(device)
        trunk = batch['trunk'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()
        output = model(cont, cat, trunk)
        loss = ml2re_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * cont.size(0)

    avg_loss = total_loss / len(loader.dataset)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    return avg_loss

def main(args):
    # 数据加载
    train_loader, test_loader, scaler_cont, cat_encoders = get_dataloaders(
        args.data_file, batch_size=args.batch_size, train_ratio=args.train_ratio,
        use_synthetic=args.use_synthetic
    )

    cat_cols = ['材料', '应力比', '频率(hz)']
    cat_cardinalities = [len(cat_encoders[col]) for col in cat_cols]

    model = DeepOFormerGT(
        cont_features=['转速(rpm)', '温度(k)', '应力幅（Mpa)'],
        cat_features=cat_cols,
        cat_cardinalities=cat_cardinalities,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        branch_output_dim=args.branch_output_dim,
        trunk_input_dim=3,
        trunk_hidden_dim=args.trunk_hidden_dim,
        trunk_num_layers=args.trunk_num_layers,
        use_bias=True
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    writer = SummaryWriter(log_dir=args.log_dir)

    best_rmre = float('inf')
    for epoch in range(1, args.epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer)

        mae, rmse, r2, mre, rmre = evaluate_model(model, test_loader, device)
        writer.add_scalar('Metrics/test_mae', mae, epoch)
        writer.add_scalar('Metrics/test_rmse', rmse, epoch)
        writer.add_scalar('Metrics/test_r2', r2, epoch)
        writer.add_scalar('Metrics/test_mre', mre, epoch)
        writer.add_scalar('Metrics/test_rmre', rmre, epoch)

        scheduler.step(rmre)

        print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test RMRE: {rmre:.4f} | R2: {r2:.4f}')

        if rmre < best_rmre:
            best_rmre = rmre
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f'  -> Best model saved (RMRE={rmre:.4f})')

    writer.close()
    print('Training finished.')

    # 训练完成后进行可视化分析
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    model.eval()

    # 1. 散点图
    plot_predictions(model, test_loader, device, save_path='scatter_test.png')

    # 3. 不确定性量化（MC Dropout）
    plot_uncertainty(model, test_loader, device, save_path='uncertainty.png')

    # 4. 单变量敏感性分析
    sensitivity_analysis(model, device, scaler_cont, cat_encoders, save_dir='./sensitivity')

def plot_predictions(model, loader, device, save_path='scatter.png'):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            cont = batch['cont'].to(device)
            cat = batch['cat'].to(device)
            trunk = batch['trunk'].to(device)
            target = batch['target'].to(device)
            output = model(cont, cat, trunk)
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten()

    preds_orig = np.exp(preds)
    targets_orig = np.exp(targets)

    plt.figure(figsize=(6,6))
    plt.scatter(targets_orig, preds_orig, alpha=0.6, s=20)
    min_val = min(targets_orig.min(), preds_orig.min())
    max_val = max(targets_orig.max(), preds_orig.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True Life (cycles)')
    plt.ylabel('Predicted Life (cycles)')
    plt.title('Prediction on Test Set')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_uncertainty(model, loader, device, T=30, save_path='uncertainty.png'):
    model.train()  # 启用 Dropout
    all_means, all_stds, all_targets = [], [], []
    with torch.no_grad():
        for batch in loader:
            cont = batch['cont'].to(device)
            cat = batch['cat'].to(device)
            trunk = batch['trunk'].to(device)
            target = batch['target'].to(device)
            preds_t = []
            for _ in range(T):
                output = model(cont, cat, trunk)
                preds_t.append(output.cpu().numpy())
            preds_t = np.stack(preds_t, axis=-1)  # [batch, T]
            mean = preds_t.mean(axis=-1)
            std = preds_t.std(axis=-1)
            all_means.append(mean)
            all_stds.append(std)
            all_targets.append(target.cpu().numpy())
    means = np.concatenate(all_means).flatten()
    stds = np.concatenate(all_stds).flatten()
    targets = np.concatenate(all_targets).flatten()

    # 转换回原始尺度
    means_orig = np.exp(means)
    stds_orig = means_orig * stds  # 近似：对数空间标准差对应原始乘性误差
    targets_orig = np.exp(targets)

    plt.figure(figsize=(8,6))
    plt.errorbar(targets_orig, means_orig, yerr=2*stds_orig, fmt='o', alpha=0.6, capsize=2, markersize=3)
    plt.plot([targets_orig.min(), targets_orig.max()], [targets_orig.min(), targets_orig.max()], 'r--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True Life')
    plt.ylabel('Predicted Life with 2σ Interval')
    plt.title('Uncertainty Quantification (MC Dropout)')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.savefig(save_path, dpi=150)
    plt.show()

def sensitivity_analysis(model, device, scaler_cont, cat_encoders, save_dir='./sensitivity'):
    """
    固定其他变量，改变应力幅和温度，观察模型预测变化，并与物理模型对比。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 选择一个基准样本：取测试集中一个典型工况
    # 这里手动构造一个样本（需要逆标准化）
    # 原始特征范围：
    # 转速(rpm) 取 10000, 温度(k) 取 650, 应力幅 取 300, 材料 INC718, 应力比 0.4, 频率 50Hz, 坐标取一个点
    base_cont = np.array([[10000, 650, 300]], dtype=np.float32)   # [转速, 温度, 应力幅]
    base_cat = np.array([[0, 1, 0]], dtype=np.int64)   # 编码
    # 实际中应根据编码映射正确赋值，这里简化为0,1,0
    base_trunk = np.array([[-12.5, 18.9, -6.9]], dtype=np.float32)  # 叶根坐标

    # 标准化连续特征
    base_cont_scaled = scaler_cont.transform(base_cont)

    # 转换为 tensor
    cont_tensor = torch.tensor(base_cont_scaled, dtype=torch.float32).to(device)
    cat_tensor = torch.tensor(base_cat, dtype=torch.long).to(device)
    trunk_tensor = torch.tensor(base_trunk, dtype=torch.float32).to(device)

    # 变化范围
    sa_range = np.linspace(50, 500, 50)
    temp_range = np.linspace(600, 900, 50)

    model.eval()
    preds_sa = []
    with torch.no_grad():
        for sa in sa_range:
            cont = base_cont_scaled.copy()
            cont[0, 2] = (sa - scaler_cont.mean_[2]) / scaler_cont.scale_[2]   # 更新应力幅标准化值
            cont_t = torch.tensor(cont, dtype=torch.float32).to(device)
            output = model(cont_t, cat_tensor, trunk_tensor)
            preds_sa.append(output.item())
    preds_sa = np.exp(preds_sa)  # 还原原始尺度

    plt.figure()
    plt.plot(sa_range, preds_sa, 'b-', label='Model prediction')
    # 这里简单用 Basquin 拟合：N = C * Sa^m，数据拟合
    # 在 dataset 中实现物理模型并导入
    # 从 dataset 导入 predict_physical
    plt.xlabel('Stress Amplitude (MPa)')
    plt.ylabel('Predicted Life (cycles)')
    plt.yscale('log')
    plt.title('Sensitivity to Stress Amplitude')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'sensitivity_sa.png'))
    plt.show()

    # 温度敏感性
    preds_temp = []
    with torch.no_grad():
        for T in temp_range:
            cont = base_cont_scaled.copy()
            cont[0, 1] = (T - scaler_cont.mean_[1]) / scaler_cont.scale_[1]   # 更新温度标准化值
            cont_t = torch.tensor(cont, dtype=torch.float32).to(device)
            output = model(cont_t, cat_tensor, trunk_tensor)
            preds_temp.append(output.item())
    preds_temp = np.exp(preds_temp)

    plt.figure()
    plt.plot(temp_range, preds_temp, 'r-', label='Model prediction')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Predicted Life (cycles)')
    plt.yscale('log')
    plt.title('Sensitivity to Temperature')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'sensitivity_temp.png'))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data_life.csv')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=128)
    parser.add_argument('--branch_output_dim', type=int, default=32)
    parser.add_argument('--trunk_hidden_dim', type=int, default=64)
    parser.add_argument('--trunk_num_layers', type=int, default=3)
    parser.add_argument('--log_dir', type=str, default='runs/deepoformer_gt')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--use_synthetic', action='store_true', default=True,
                        help='Use synthetic data generated by physical model')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
