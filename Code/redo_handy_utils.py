import os
import re
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

def load_data_from_path(path_list, idx_col=0):
    # 路径是一个字符串，则转化为列表
    if type(path_list) == str:
        path_list = [path_list]
    
    # 路径不是列表，返回错误
    if type(path_list) != list:
        print('Path not valid')
        return
    
    # 依次读取path_list内的csv文件，并concat成Dataframe格式
    df = pd.DataFrame()
    for path in path_list:
        temp = pd.read_csv(path, index_col=idx_col)
        print(os.path.basename(path), temp.shape)
        df = pd.concat([df, temp])
    
    df.reset_index(drop=True, inplace=True)
    print('Size of complete dataset:', df.shape)
    
    return df

# LightGBM的默认参数
def load_default_params(seed):
    params = {
        # "boosting": "gbdt",       # 默认的boosting方式就是gbdt
        "objective": "regression",  # loss函数是L2的回归树
        "metric": "rmse",           # 优化参数rmse
        "max_depth": 10,            # 限制树的最大深度
        "min_data_in_leaf": 20,     # 每个叶节点最少的样本数，防止样本太少导致过拟合
        "num_leaves": 64,           # 叶节点最大个数，num_leaves < 2^max_depth
        
        "learning_rate": 0.01,      # 学习率
        "reg_alpha": 1,             # L1正则化系数
        "reg_lambda": 0.1,          # L2正则化系数
        # "save_binary": True,      # 将数据集存储为二进制文件，以便下次快速读取


        "bagging_fraction": 0.8,     # 不放回随机采样比例，加速和防止过拟合，别名subsample
        "bagging_freq": 5,           # 一次迭代一颗树，每隔k次迭代随机bagging一次，用于下轮k次迭代
        "feature_fraction": 0.9,     # 特征筛选比例，别名colsample_bytree
        
        "verbosity": -1,            # -1为显示致命错误
        "seed": seed
    }
    
    return params

# 输入特征的基本信息
def make_clean_look_of_columns(cols):
    bulk_info = []
    wall_info = []
    ratio_info = []
    dr_info = []
    other_info = []
    for col in cols:
        if '_b' in col:
            bulk_info.append(col)
        elif '_w' in col:
            wall_info.append(col)
        elif '_ratio' in col:
            ratio_info.append(col)
        elif '_dr' in col:
            dr_info.append(col)
        else:
            other_info.append(col)
    
    print('-'*80)
    if len(other_info) != 0:
        print('Other info [%d]:' %len(other_info), ', '.join(other_info))
    if len(dr_info) != 0:
        print('Dr info [%d]:' %len(dr_info), ', '.join(dr_info))
    if len(ratio_info) != 0:
        print('Ratio info [%d]:' %len(ratio_info), ', '.join(ratio_info))
    if len(bulk_info) != 0:
        print('Bulk info [%d]:' %len(bulk_info), ', '.join(bulk_info))
    if len(wall_info) != 0:
        print('Wall info [%d]:' %len(wall_info), ', '.join(wall_info))
    print('\nNumber of features:', len(cols))
    print('-'*80)

def error_rate(y_test, y_pred):
    """
    计算预测的平均相对误差
    """
    if (type(y_test) != np.ndarray) or (type(y_pred) != np.ndarray):
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
    return np.mean(np.abs(y_pred - y_test) / y_test)

def percentile90(y_test, y_pred):
    """
    计算大于90%数据集的最小误差，即误差的90分位数
    """
    if (type(y_test) != np.ndarray) or (type(y_pred) != np.ndarray):
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
    error = np.abs(y_pred - y_test) / y_test
    return np.percentile(error, 90)


def print_data_size(x_train=None, x_test=None, x_val=None):
    print('x_train matrix shape:', x_train.shape) if type(x_train) == pd.core.frame.DataFrame else print()
    print('x_test matrix shape:', x_test.shape) if type(x_test) == pd.core.frame.DataFrame else print()
    print('x_val matrix shape:', x_val.shape) if type(x_val) == pd.core.frame.DataFrame else print()
    print()

# 模型评估
def print_metrics(y_test, y_pred):
    print('MSE   =',mean_squared_error(y_test, y_pred))
    print('RMSE  = %.4f' %np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R2    =', r2_score(y_test, y_pred))
    print()
    # 相对误差
    error = np.abs((y_pred - y_test)/y_test)
    print('Error = %.2f %%' %(error.mean()*100))
    print('P90   = %.2f %%' %(np.percentile(error, 90) * 100))

def print_data_info(data):
    "输出数据集大小、各参数的范围"
    # 数据来源
    # print('-'*80)
    # print('source included')
    # print(data['source'].unique())

    # 每种管径包含的点
    if 'tube_diameter' in data.columns:
        print('-'*80)
        print('Number of cases for each diameter:')
        tube_diameters = data['tube_diameter'].unique().tolist()
        tube_diameters.sort()
        for diameter in tube_diameters:
            print('Number of cases for', diameter, 'mm tube :', np.sum(data['tube_diameter'] == diameter))

    # 流动方向
    if 'flow_direction' in data.columns:
        print('-'*80)
        updw_count = data['flow_direction'].value_counts()
        print('Number of UPWARD points:', updw_count[1])
        print('Number of DOWNWARD points:', updw_count[0])

    # 流量
    if 'mass_flow_rate' in data.columns:
        print('-'*80)
        g = data['mass_flow_rate'].unique()
        g.sort()
        print('Mass flow rate range: %.4f ~ %.4f kg/h' %(g[0], g[-1]))

    # 热流
    if 'heat_flux' in data.columns:
        print('-'*80)
        qw = data['heat_flux'].unique()
        qw.sort()
        print('Heat flux range: %.2f ~ %.2f kW/m2' %(qw[0]/1000, qw[-1]/1000))
    
    # 压力
    if 'p_in' in data.columns:
        print('-'*80)
        p_in = data['p_in'].unique()
        p_in.sort()
        print('Pressure inlet range: %.4f ~ %.4f MPa' %(p_in[0], p_in[-1]))

    # 雷诺数
    if 'Re_in' in data.columns:
        print('-'*80)
        Re_in = data['Re_in'].unique()
        Re_in.sort()
        print('Reynolds inlet range: %.4f ~ %.4f' %(Re_in[0], Re_in[-1]))
    
    # 入口温度
    if 'T_in' in data.columns:
        print('-'*80)
        T_in = data['T_in'].unique()
        T_in.sort()
        print('Temperature inlet range: %.2f ~ %.2f K' %(T_in[0], T_in[-1]))

    # 出口温度
    if 'T_out' in data.columns:
        print('-'*80)
        T_out = data['T_out'].unique()
        T_out.sort()
        print('Temperature outlet range: %.2f ~ %.2f K' %(T_out[0], T_out[-1]))

    # 工质
    if 'working_fluid' in data.columns:
        print('-'*80)
        fluid_info = data['working_fluid'].value_counts()
        for f in fluid_info.index:
            print(f'Number of points for {f}: {fluid_info[f]}')
    
     # 出口温度
    if 'Speed' in data.columns:
        print('-'*80)
        speed = data['Speed'].unique()
        speed.sort()
        print('Rotation speed range: %.2f ~ %.2f r/min' %(speed[0], speed[-1]))

def set_default_params():
    # 设置图片默认字体格式
    plt.rcParams['font.size'] = 18  # 设置字体大小 
    plt.rcParams['axes.labelpad'] = 16
    plt.rcParams['axes.titlepad'] = 16
    plt.rcParams['font.sans-serif'] = ['Calibri']  # sans-serif字体设置为Calibri

def set_params_zh():
    # 针对中文图片，额外设置字体格式
    set_default_params()
    plt.rcParams['font.family'] = ['sans-serif']  # 字体样式设置为sans-serif
    plt.rcParams['font.sans-serif'] = ['SimHei']  # sans-serif字体设置为SimHei
    plt.rcParams['font.serif'] = ['SimHei']  # serif字体设置为SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 负号
# =============================================================================

# 预测值与真实值对比图-Nu comparison
def plot_nucomp(y_test, y_pred, fig_name='nucomp.pdf', xlim=[0,100], ylim=[0,100], save_to='output_images', lang='zh', save_fig=True):
    # 将y_test转化为ndarray格式
    y_test = np.array(y_test)
    
    # 设置字体
    if lang == 'zh':
        set_params_zh()
    else: 
        set_default_params()

    fig_name = 'en_' + fig_name if fig_name == 'en' else fig_name

    plt.figure(figsize=(6,6))  # 设置图片窗口大小
    plt.plot(xlim, ylim, color='tab:red')#, label='准确值') # 红色的直线，连接(0,0)和(100,100)两个点
    plt.scatter(y_test, y_pred, marker='o', color='', edgecolor='tab:blue', s=90) #, label='预测值') # 空心圆圈散点

    
    # 设置刻度名和标签
    if lang == 'zh':
        plt.xlabel('真实Nu')
        plt.ylabel('预测Nu')
        plt.legend(['预测值', '准确值'], loc='upper left')
    else:
        plt.xlabel('True Nu')
        plt.ylabel('Predicted Nu')
        plt.legend(['Model prediction', 'Perfect correlation'], loc='upper left')

    # 设置x和y的刻度范围
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

    # 保存图片，bbox_inches='tight'用于去除图片空白
    if save_fig:
        plt.savefig(os.path.join(save_to, fig_name), dpi=600, format='pdf', bbox_inches='tight')

# Nu数沿程变化
def plot_nuxd(x_test, y_test, y_pred, fig_name='nuxd.pdf', xlim=[0,240], ylim=[0,80], save_to='output_images', lang='zh', save_fig=True):
    
    # 设置字体
    if lang == 'zh':
        set_params_zh()
        plt.legend(['测试集真实值', '模型预测值'])
    else: 
        set_default_params()
        plt.legend(['Nu in testset', 'Model prediction'])

    fig_name = 'en_' + fig_name if fig_name == 'en' else fig_name
    
    plt.figure(figsize=(6,6))
    
    plt.plot(x_test, y_test, lw=3)
    plt.scatter(x_test, y_pred, marker='o', color='', edgecolor='tab:red', s=90)

    plt.ylabel('Nu')
    plt.xlabel('x/d')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

    if save_fig:
        plt.savefig(os.path.join(save_to, fig_name), dpi=600, format='pdf', bbox_inches='tight')
    plt.show()

# 特征重要性排序
def plot_feat_importance(model, max_num_features=10, fig_name='feat_importance.pdf', save_to='output_images', lang='zh', save_fig=True, importance_type='split'):
    
    # 设置字体
    if lang == 'zh':
        set_params_zh()
    else: 
        set_default_params()
    
    fig_name = 'en_' + fig_name if fig_name == 'en' else fig_name
    xlabel = 'Feature importance' if lang=='en' else '特征重要性'

    fig = lgb.plot_importance(
        model, 
        figsize=(8,8), 
        max_num_features=max_num_features,   # k-对前k个重要特征进行排序
        importance_type=importance_type,             # 按分裂次数计算重要性，选择gain则按收益计算
        title=None, 
        xlabel=xlabel, 
        ylabel=None
    )

    fig = fig.get_figure()
    if save_fig:
        plt.savefig(os.path.join(save_to, fig_name), dpi=600, format='pdf', bbox_inches='tight')
    plt.show()
  
def plot_heatmap_imshow(Z, xlim=[0, 1], ylim=[0, 1], figsize=(9, 5), cmap=None, feats=['$x$', '$y$', '$z$'], SHOWFIG=True, POINTDRAW=False, **POINT):
    '''
    Args:
    Z        ::  2d array, 数据
    xlim     ::  array-like, 横坐标的上下限
    ylim     ::  array-like, 纵坐标的上下限
    figsize  ::  tuple, 图片大小
    cmap     ::  str, 颜色
    feats    ::  list, 变量名
    SHOWFIG  ::  bool, 是否展示图片
    POINTDRAW::  bool, 是否绘制散点
    '''
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if POINTDRAW:
        # 绘制
        X_eq = POINT['X_eq']
        X_bd = POINT['X_bd']
        if len(X_eq) > 0:
            ax.plot(X_eq[:,0], X_eq[:,1], POINT['shape_eq'], c=POINT['color_eq'], label='Equation Data (%d points)' % (X_eq.shape[0]), markersize=4, clip_on=False)
        if len(X_bd) > 0:
            ax.plot(X_bd[:,0], X_bd[:,1], POINT['shape_bd'], c=POINT['color_bd'], label='Boundary Data (%d points)' % (X_bd.shape[0]), markersize=4, clip_on=False)
        
    z_fig = ax.imshow(Z, interpolation='nearest', cmap=cmap,
                extent=[xlim[0], xlim[1], ylim[0], ylim[1]],  # 云图的边界
                origin='lower', aspect='auto')  #aspect控制轴的纵横比
    divider = make_axes_locatable(ax)  # 使得divider与ax的高度一致
    cax = divider.append_axes("right", size="5%", pad=0.10)  # 在图片右侧添加bar, size控制宽度, pad控制距离
    cbar = fig.colorbar(z_fig, cax=cax)  # 为bar指定对象
    cbar.ax.tick_params(labelsize=18) 
    ax.set_xlabel(feats[0], size=22)
    ax.set_ylabel(feats[1], size=22)

    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.9, -0.05), 
        ncol=5, 
        frameon=False, 
        prop={'size': 15}
    )
    ax.set_title(feats[2], fontsize = 20) # font size doubled
    ax.tick_params(labelsize=15)


    if SHOWFIG:
        plt.show()
# =============================================================================
