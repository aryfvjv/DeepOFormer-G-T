#!/usr/bin/python3
# -*-coding:utf-8-*-
# @file:dataset
# @author:20402
# @date:2026/2/27
# @version:2.1
# @desc: 包含基于物理模型的合成数据生成（修正缩进错误）

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

class LifeDataset(Dataset):
    """
    自定义数据集类，用于加载疲劳寿命数据，并进行预处理：
    - 类别特征编码为整数索引
    - 连续特征标准化（Z-score）
    - 空间坐标保留为浮点数
    - 目标值取自然对数
    """
    def __init__(self, data_file, cont_cols, cat_cols, trunk_cols, target_col,
                 log_target=True, scaler_cont=None, fit_scaler=True):
        """
        参数:
            data_file (str): CSV数据文件路径
            cont_cols (list): 连续特征列名列表
            cat_cols (list): 类别特征列名列表
            trunk_cols (list): 主干网络输入（空间坐标）列名列表
            target_col (str): 目标值列名
            log_target (bool): 对目标值取自然对数
            scaler_cont (StandardScaler or None): 预先定义的连续特征标准化器
            fit_scaler (bool): 是否拟合标准化器（训练集为True，测试集为False）
        """
        # 读取CSV文件
        self.df = pd.read_csv(data_file)
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.trunk_cols = trunk_cols
        self.target_col = target_col
        self.log_target = log_target

        # ========== 目标值处理（添加容错） ==========
        target_raw = self.df[target_col].values
        # 若为字符串类型（如含有方括号），则清理
        if target_raw.dtype == object:
            target_clean = np.array([float(str(v).strip('[]')) for v in target_raw], dtype=np.float32)
        else:
            target_clean = target_raw.astype(np.float32)

        if log_target:
            target_clean = np.log(target_clean + 1e-12)
        self.target = target_clean.reshape(-1, 1)

        # ========== 类别特征编码 ==========
        self.cat_encoders = {}
        for col in cat_cols:
            self.df[col] = self.df[col].astype('category')
            self.cat_encoders[col] = dict(enumerate(self.df[col].cat.categories))
            self.df[col] = self.df[col].cat.codes.astype(int)

        # ========== 连续特征标准化 ==========
        if scaler_cont is None and fit_scaler:
            self.scaler_cont = StandardScaler()
            self.cont_data = self.scaler_cont.fit_transform(self.df[cont_cols].values)
        elif scaler_cont is not None:
            self.scaler_cont = scaler_cont
            self.cont_data = self.scaler_cont.transform(self.df[cont_cols].values)
        else:
            self.scaler_cont = None
            self.cont_data = self.df[cont_cols].values.astype(np.float32)

        # ========== 类别数据 ==========
        self.cat_data = self.df[cat_cols].values.astype(np.int64)

        # ========== 主干输入（空间坐标） ==========
        self.trunk_data = self.df[trunk_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """返回一个样本的字典"""
        return {
            'cont': torch.tensor(self.cont_data[idx], dtype=torch.float32),
            'cat': torch.tensor(self.cat_data[idx], dtype=torch.long),
            'trunk': torch.tensor(self.trunk_data[idx], dtype=torch.float32),
            'target': torch.tensor(self.target[idx], dtype=torch.float32)
        }


def fit_physical_model(df):
    """
    从原始数据中拟合一个简化的物理模型：
    log(N) = a + b*log(Sa) + c/T + d*log(Sa)* (1/T)
    使用多项式特征 + 线性回归
    返回一个函数 predict(Sa, T) 预测寿命（原始尺度）
    """
    X = df[['应力幅（Mpa)', '温度(k)']].values
    y = np.log(df['总寿命'].values + 1e-12)

    log_Sa = np.log(X[:, 0])
    inv_T = 1.0 / X[:, 1]
    X_poly = np.column_stack([log_Sa, inv_T, log_Sa * inv_T])

    reg = LinearRegression()
    reg.fit(X_poly, y)

    def predict(Sa, T):
        Sa = np.asarray(Sa)
        T = np.asarray(T)
        log_Sa = np.log(Sa)
        inv_T = 1.0 / T
        X_new = np.column_stack([log_Sa, inv_T, log_Sa * inv_T])
        log_N = reg.predict(X_new)
        return np.exp(log_N)

    return predict

def generate_synthetic_data(df, num_synthetic=200, random_seed=42):
    np.random.seed(random_seed)
    predict_physical = fit_physical_model(df)

    # 获取各特征范围
    rpm_range = (df['转速(rpm)'].min(), df['转速(rpm)'].max())
    temp_range = (df['温度(k)'].min(), df['温度(k)'].max())
    sa_range = (df['应力幅（Mpa)'].min(), df['应力幅（Mpa)'].max())
    x_vals = df['x(mm)'].unique()
    y_vals = df['y(mm)'].unique()
    z_vals = df['z(mm)'].unique()

    synthetic_rows = []
    for _ in range(num_synthetic):
        rpm = np.random.uniform(*rpm_range)
        temp = np.random.uniform(*temp_range)
        sa = np.random.uniform(*sa_range)
        idx_xyz = np.random.randint(len(x_vals))
        x = x_vals[idx_xyz]
        y = y_vals[idx_xyz]
        z = z_vals[idx_xyz]
        stress_ratio = np.random.choice(df['应力比'].unique())
        freq = np.random.choice(df['频率(hz)'].unique())
        material = 'INC718'

        # 物理模型预测寿命
        life = predict_physical(sa, temp)
        if isinstance(life, np.ndarray):
            life = life.item() if life.size == 1 else life[0]
        life = float(life)

        # 蠕变寿命
        creep_life = 1e13 * np.exp(-0.01 * (temp - 600))
        creep_life = float(creep_life)

        total_life = min(life, creep_life)
        total_life = float(total_life)

        new_row = {
            '材料': material,
            '应力比': stress_ratio,
            '频率(hz)': freq,
            '转速(rpm)': rpm,
            '应力(mpa)': sa,
            '温度(k)': temp,
            '应力幅（Mpa)': sa,
            'x(mm)': x,
            'y(mm)': y,
            'z(mm)': z,
            '中值疲劳寿命（单位：repeats)': life,
            '蠕变寿命': creep_life,
            '总寿命': total_life
        }
        synthetic_rows.append(new_row)

    synthetic_df = pd.DataFrame(synthetic_rows)
    combined_df = pd.concat([df, synthetic_df], ignore_index=True)
    return combined_df

def get_dataloaders(data_file, batch_size=16, train_ratio=0.8, random_seed=42, use_synthetic=True):
    """
    创建训练集和测试集的 DataLoader。
    """
    cont_cols = ['转速(rpm)', '温度(k)', '应力幅（Mpa)']
    cat_cols = ['材料', '应力比', '频率(hz)']
    trunk_cols = ['x(mm)', 'y(mm)', 'z(mm)']
    target_col = '总寿命'

    df_original = pd.read_csv(data_file)

    if use_synthetic:
        df = generate_synthetic_data(df_original, num_synthetic=200, random_seed=random_seed)
        print(f"原始数据量: {len(df_original)}，合成后数据量: {len(df)}")
    else:
        df = df_original

    temp_file = data_file.replace('.csv', '_with_synthetic.csv')
    df.to_csv(temp_file, index=False)

    # 创建完整数据集（用于获取标准化器和编码器）
    full_dataset = LifeDataset(
        data_file=temp_file,
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        trunk_cols=trunk_cols,
        target_col=target_col,
        log_target=True,
        fit_scaler=True
    )
    scaler_cont = full_dataset.scaler_cont
    cat_encoders = full_dataset.cat_encoders

    total = len(full_dataset)
    train_len = int(train_ratio * total)
    test_len = total - train_len
    torch.manual_seed(random_seed)
    train_ds, test_ds = random_split(full_dataset, [train_len, test_len])
    train_indices = train_ds.indices
    test_indices = test_ds.indices

    # 重建训练集和测试集（使用相同的标准化器，不重新拟合）
    train_dataset = LifeDataset(
        data_file=temp_file,
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        trunk_cols=trunk_cols,
        target_col=target_col,
        log_target=True,
        scaler_cont=scaler_cont,
        fit_scaler=False
    )
    test_dataset = LifeDataset(
        data_file=temp_file,
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        trunk_cols=trunk_cols,
        target_col=target_col,
        log_target=True,
        scaler_cont=scaler_cont,
        fit_scaler=False
    )

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, test_loader, scaler_cont, cat_encoders
