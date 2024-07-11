import os
import random

import numpy as np
import pandas as pd

import CoolProp.CoolProp as CP
from sklearn.utils import resample
from redo_utils import find_starts_ends_points, load_NIST_prop, calculate_nondimensional_numbers, calculate_Nu_ratio

# ==============================================================================================
# 基本通用函数
# ==============================================================================================
# 1 读取数据
def load_dns_data(path, source_name):
    '''
    读取DNS数据。
    Args:
        path(str): 文件路径
        source_name(list): 数据来源, DNS数据的文件夹名称
    
    Return:
        data_0d(DataFrame): 0维数据, 通常储存入口参数, 临界温度等参数
        data_1d(DataFrame): 1维数据, 通常储存沿程的wall和bulk参数
        data_2d(dict): 2维数据, keys为参数名, values为DataFrame, 其中行为轴向, 列为径向, 列名为r/R
        x_coor(dict): 沿程坐标x/d, key为source_name
        r_coor(dict): 径向坐标x/d, key为source_name
    '''
    data_2d, x_coor, r_coor =  {}, {}, {}
    data_0d, data_1d = pd.DataFrame(), pd.DataFrame()
    for sn in source_name:
        temp_0d = pd.read_csv(os.path.join(path, sn, 'data_0d.csv'))
        temp_1d = pd.read_csv(os.path.join(path, sn, 'data_1d.csv'))
        x_coor[sn] = temp_1d['x/d'].unique().astype(float)  # 沿程坐标
        x_coor[sn].sort()

        data_0d = pd.concat([data_0d, temp_0d], axis=0, ignore_index=True)
        data_1d = pd.concat([data_1d, temp_1d], axis=0, ignore_index=True)
        
        temp_2d = {}
        for root,dirs,files in os.walk(os.path.join(path, sn, 'data_2d')):
            for file in files:
                file_name = file.rstrip('.csv')
                temp_2d[file_name] = pd.read_csv(os.path.join(root, file))
        data_2d[sn] = temp_2d
        r_coor[sn] = data_2d[sn]['CpbyT'].columns.astype(float)  # 径向坐标

    return data_0d, data_1d, data_2d, x_coor, r_coor

# 2 分类变量数字化
def cate_feat_encode(data, feats, label={}):
    '''
    标签编码分类变量。
    Args:
        data(DataFrame): 数据
        feats(list): 分类特征
        label(dict): {特征名: {特征值: label}}
    '''
    if type(feats) is not list:
        feats = [feats]

    for i in range(len(feats)):
        if feats[i] not in label.keys():
            feat_val = data[feats[i]].unique()
            for j in range(len(feat_val)):
                data.loc[data[feats[i]] == feat_val[j], feats[i]] = j
        else:
            feat_val = label[feats[i]]
            for k,v in feat_val.items():
                data.loc[data[feats[i]] == k, feats[i]] = v
        data[feats[i]] = data[feats[i]].astype(int)
    return

# 3 构造输入数据集
def construct_input_data_from_dns(data_0d, data_1d, basic_feats, pin):
    '''
    标签编码分类变量。
    Args:
        data_0d(DataFrame): 0维数据, 通常储存入口参数, 临界温度等参数
        data_1d(DataFrame): 1维数据, 通常储存沿程的wall和bulk参数
        basic_feats(list): 基本特征
        pin(list): 入口压力, 每个source有一个入口压力

    Return:
        data(DataFrame): 基于DNS的输入数据
    '''
    if ('Tbdim' not in basic_feats) or ('Twdim' not in basic_feats):
        print('Please add feature "Tbdim" or "Twdim"!')
        return

    # 提取实验参数
    data = data_1d[basic_feats]  # 基本参数
    data.rename(columns={'Tbdim':'Tb', 'Twdim':'Tw'}, inplace=True)
    data, str_idx, end_idx = find_starts_ends_points(data, GETIDX=True)
    wc_num = len(str_idx)  # 工况数

    # 工况标记, 便于画图
    data['working_condition'] = np.nan
    for i in range(wc_num):
        data.loc[str_idx[i]:end_idx[i], 'working_condition'] = i

    # 常量参数
    feat_name = ['T_in', 'Re_in', 'heat_flux', 'tube_diameter']
    raw_feat = ['T0', 'ReN', 'qw0', 'L0']
    data[feat_name] = data_0d[raw_feat]
    for i in range(len(feat_name)):
        for j in range(wc_num):
            wc_len = end_idx[j] - str_idx[j] + 1
            data.loc[str_idx[j]:end_idx[j], feat_name[i]] = np.ones(wc_len)*data_0d.loc[j, raw_feat[i]]   

    # 创造空列
    data['p_in'] = np.nan

    # 入口压力
    for i in range(len(data['source'].unique())):
        data.loc[data['source'] == i, 'p_in'] = pin[i]

    data['Re_in'] = data['Re_in']*2  # 原始数据中Re以半径定义
    data['tube_diameter'] = data['tube_diameter']*2e3  # mm储存直径
    data.drop(columns=['inlet'], inplace=True)
    return data

    