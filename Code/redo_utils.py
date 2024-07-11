import os
from pickle import FALSE
import random
from tkinter.tix import Tree

import torch
import numpy as np
import pandas as pd
from itertools import combinations
from lightgbm.basic import Booster
from numpy.core.defchararray import index, lower, replace
from numpy.lib.function_base import copy, extract
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import CoolProp.CoolProp as CP
from sklearn.utils import resample


# ==============================================================================================
# 基本通用函数
# ==============================================================================================
def set_seed(seed):
    '''
    设置随机数种子
    Args:
        seed   :: int, 随机数
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def property_from_NIST(property_name, T, p, fluid='CarbonDioxide'):
    '''
    简化从NIST.CoolProp导入物性参数的代码

    Args:
        property_name :: str, 物性名
        T             :: float or array-like, 温度
        p             :: float or array-like, 压力
        fluid         :: str, 工质种类
    
    Return:
        property_data :: float or array-like, 热物性参数数据
    '''
    
    property_data = CP.PropsSI(property_name, 'T', T, 'P', p, fluid)
    return property_data

# 找到每个工况的起始位置和结束位置
def find_starts_ends_points(data, GETIDX=False, IDXLEN=False):
    """
    找到每个工况的入口点和出口点。
    Args:
        data    :: DataFrame, 数据
        GETIDX  :: bool, 是否返回索引
        IDXLEN  :: bool, 是否返回索引的长度
    
    Return:
        data    :: DataFrame, 数据
        str_idx :: array-like, 入口索引
        end_idx :: array-like, 出口索引
        idx_len :: int, 索引长度
    """
    if IDXLEN and not GETIDX:
        print('When IDXLEN is True, GETIDX should also be True.')
        return

    data['inlet'] = 0  # 为data新建一列inlet,标记初值为0
    i = 0

    while i < data.shape[0] - 1:
        start = i
        end = start + 1
        while data.loc[end, 'x/d'] > data.loc[end - 1, 'x/d']:
            end += 1
            if end == data.shape[0]:
                break
        i = end
        # 入口赋值1，出口-1
        data.loc[start, 'inlet'] = 1
        data.loc[end - 1, 'inlet'] = -1
    if GETIDX:
        str_idx = data[data['inlet'] == 1].index
        end_idx = data[data['inlet'] == -1].index
        idx_len = len(str_idx)
        if IDXLEN:
            return data, str_idx, end_idx, idx_len
        else:
            return data, str_idx, end_idx
    else:
        return data

# 找到每个工况的起始位置和结束位置
def rolling_mean_data(raw_data, feat, win_len):
    """
    对某特征进行滑动平均消除噪音, 忽略前win_len个数据。
    Args:
        raw_data    :: DataFrame, 数据
        feat    :: str, 特征名
        win_len :: int, 窗口长度
    
    Return:
        data    :: DataFrame, 数据
    """
    data, str_idx, end_idx  = find_starts_ends_points(raw_data.copy(), GETIDX=True)
    for i in range(len(str_idx)):
        raw_values = data.loc[str_idx[i]:str_idx[i]+(win_len-2), feat].copy().values
        data.loc[str_idx[i]:end_idx[i], feat] = data.loc[str_idx[i]:end_idx[i], feat].rolling(window=win_len).mean()
        data.loc[str_idx[i]:str_idx[i]+(win_len-2), feat] = raw_values
    data.drop(columns='inlet', inplace=True)
    return data

# 计算参数的轴向梯度
def calculate_feature_axial_gradient(data, Temperature=True, dimensional=False, property=False):
    '''
    计算温度、物性、无量纲数等的轴向梯度
    '''
    data = find_starts_ends_points(data)  # 标记数据的起始位置
    start_idx = data[data['inlet'] == 1].index.tolist() # 提取起始位置，相当于标记工况
    start_idx.append(data.shape[0])
    
    if Temperature:
        data['dTw/dx'] = 0
        data['dTb/dx'] = 0
        for i in range(len(start_idx) - 1):
            for j in range(start_idx[i] + 1, start_idx[i+1]):
                data['dTw/dx'] = (data.loc[j, 'Tw'] - data.loc[j-1, 'Tw']) / (data.loc[j-1, 'x/d'] - data.loc[j-1, 'x/d'])
                data['dTb/dx'] = (data.loc[j, 'Tb'] - data.loc[j-1, 'Tb']) / (data.loc[j-1, 'x/d'] - data.loc[j-1, 'x/d'])
    
    if dimensional:
        dimen_1 = ['Bo', 'Kv', 'Gr']
        dimen_2 = ['Pr', 'Re']
        for d in dimen_1:
            data[d+'_diff_ax'] = 0
            for i in range(len(start_idx) - 1):
                for j in range(start_idx[i] + 1, start_idx[i+1]):
                    data[d+'_diff_ax'] = (data.loc[j, d] - data.loc[j-1, d]) / (data.loc[j-1, 'x/d'] - data.loc[j-1, 'x/d'])

        for d in dimen_2:
            data[d+'_diff_ax_w'] = 0
            data[d+'_diff_ax_b'] = 0
            for i in range(len(start_idx) - 1):
                for j in range(start_idx[i] + 1, start_idx[i+1]):
                    data[d+'_diff_ax_w'] = (data.loc[j, d+'_w'] - data.loc[j-1, d+'_w']) / (data.loc[j-1, 'x/d'] - data.loc[j-1, 'x/d'])
                    data[d+'_diff_ax_b'] = (data.loc[j, d+'_b'] - data.loc[j-1, d+'_b']) / (data.loc[j-1, 'x/d'] - data.loc[j-1, 'x/d'])

    if property:
        prop = ['conductivity', 'Cp', 'density', 'enthalpy', 'Pr', 'viscosity', 'expansivity']
        # 创造空数组
        for p in prop:
            data[p+'_diff_ax_w'] = 0
            data[p+'_diff_ax_b'] = 0
            for i in range(len(start_idx) - 1):
                for j in range(start_idx[i] + 1, start_idx[i+1]):
                    data[p+'_diff_ax_w'] = (data.loc[j, p+'_w'] - data.loc[j-1, p+'_w']) / (data.loc[j-1, 'x/d'] - data.loc[j-1, 'x/d'])
                    data[p+'_diff_ax_b'] = (data.loc[j, p+'_b'] - data.loc[j-1, p+'_b']) / (data.loc[j-1, 'x/d'] - data.loc[j-1, 'x/d'])

    # 由于采用后向差分，初始位置可以删去
    idx_to_drop = data[data['inlet'] == 1].index
    data.drop(index=idx_to_drop, inplace=True)
    data.drop(columns='inlet', inplace=True) # 删去标记初始位置和结束位置的inlet列
    data.reset_index(drop=True, inplace=True)

    return data


# Jackson得到的关联式
def calculate_Nu_ratio(data, fluid='CarbonDioxide'):
    """
    计算 Nu/Nu_f 的值。
    实际努塞尔数Nu是由qw和(Tw-Tb)计算得到
    强迫对流努塞尔数Nuf是由Pbetukhov和Jackson的关联式计算得到
    """

    Tpc = CP.PropsSI(fluid, 'Tcrit')

    data['n'] = 0.4

    # 工况1
    idx1 = ((data['Tb'] <= data['Tw']) & (data['Tw'] <= Tpc)) | ((1.2 * Tpc <= data['Tw']) & (data['Tb'] <= data['Tw']))
    data['n'][idx1] = 0.4

    # 工况2
    idx2 = (data['Tb'] <= Tpc) & (Tpc <= data['Tw'])
    data['n'][idx2] = 0.4 + 0.2 * (data['Tw'][idx2] / Tpc - 1)

    # 工况3
    idx3 = ((Tpc <= data['Tb']) & (data['Tb'] <= 1.2 * Tpc)) | (data['Tb'] <= data['Tw'])
    data['n'][idx3] = 0.4 + 0.2 * (data['Tw'][idx3] / Tpc - 1) * (1 - 5 * (data['Tb'] / Tpc - 1))
    
    data['Nuf'] = 0.0183 * data['Re_b']**0.82 * data['Pr_b']**0.5 * (data['Cp_ratio'])**(data['n']) * (data['density_ratio'])**0.3
    data['Nu_ratio'] = data['Nu'] / data['Nuf']
    data.drop(columns=['n'], inplace=True)
    return data

# # Jackson得到的关联式
# def calculate_Nu_ratio_Bo(data, fluid='CarbonDioxide'):
#     """
#     由Jackson的隐式Bo*关联式计算得到
#     """
    

#     data['n'] = 0.4

#     # 工况1
#     idx1 = ((data['Tb'] <= data['Tw']) & (data['Tw'] <= Tpc)) | ((1.2 * Tpc <= data['Tw']) & (data['Tb'] <= data['Tw']))
#     data['n'][idx1] = 0.4

#     # 工况2
#     idx2 = (data['Tb'] <= Tpc) & (Tpc <= data['Tw'])
#     data['n'][idx2] = 0.4 + 0.2 * (data['Tw'][idx2] / Tpc - 1)

#     # 工况3
#     idx3 = ((Tpc <= data['Tb']) & (data['Tb'] <= 1.2 * Tpc)) | (data['Tb'] <= data['Tw'])
#     data['n'][idx3] = 0.4 + 0.2 * (data['Tw'][idx3] / Tpc - 1) * (1 - 5 * (data['Tb'] / Tpc - 1))
    
#     data['Nuf'] = 0.0183 * data['Re_b']**0.82 * data['Pr_b']**0.5 * (data['Cp_ratio'])**(data['n']) * (data['density_ratio'])**0.3
#     data['Nu_ratio'] = data['Nu'] / data['Nuf']
#     data.drop(columns=['n'], inplace=True)
#     return data

# 改
# 对x/d进行MinMax归一化
def normalize_xd(data):
    """
    将x/d特征按照同工况内的最大最小值进行归一化处理。
    """
    # 找到每个工况的起点和终点位置
    data = find_starts_ends_points(data)
    start_idx = data[data['inlet'] == 1].index.tolist()
    # +1确保start_idx里的每个点都是管道起点，但注意append的点之后没有数据
    start_idx.append(data.shape[0] + 1)  

    for i in range(len(start_idx) - 1):
        min_idx = start_idx[i]
        max_idx = start_idx[i+1] - 1
        min = data.loc[start_idx[i], 'x/d']
        max = data.loc[start_idx[i+1] - 1, 'x/d']
        data.loc[min_idx:max_idx, 'x/d'] = (data.loc[min_idx:max_idx, 'x/d'] - min) / (max - min)
    
    # 处理之后对数据集进行整理
    data.drop(columns=['inlet'], inplace=True)
    data.reset_index(inplace=True, drop=True)

    return data

def extract_by_range(data, start=None, end=None, col='Bo'):
    """
    根据给定的起始start、终止end位置，从中截取相应的数据片段。
    """
    # 确定上下界限
    lower_bound = start if start else (-np.inf)
    upper_bound = end if end else (np.inf)
    
    # 筛选符合提交的index
    mark = (data[col] > lower_bound) & (data[col] < upper_bound)

    # 筛选数据
    res = data.loc[mark].copy()

    return res

def Kv_Bo_split(data, Kv_criteria, Bo_criteria):
    """
    返回字典res，键从1~（Bo间隔数×Kv间隔数）变化
    Kv数分区包含Bo数分区，即Bo数分区相邻
    Kv数从大到小排序，Bo数从小到大排序
    """
    res = {}
    # 区间间隔数
    Kv_num_interval = len(Kv_criteria) - 1
    Bo_num_interval = len(Bo_criteria) - 1
    # 组合数
    combin_num = Kv_num_interval*Bo_num_interval
    # 划分Kv数，Kv子区间再依据Bo数划分
    for i in range(0, combin_num , Bo_num_interval):
        a = int((combin_num - Bo_num_interval - i)/Bo_num_interval)
        tmp = extract_by_range(data, start=Kv_criteria[a], end=Kv_criteria[a+1], col='Kv')
        # 划分Bo数
        for j in range(1, len(Bo_criteria)):
            res[i+j] = extract_by_range(tmp, start=Bo_criteria[j-1], end=Bo_criteria[j], col='Bo')
    return res

# 改
def zone_split(data, version='v1'):
    """
    选择Kv和Bo无量纲数的划分判据，并将数据集切分为不同的区域，对应不同的影响因素。
    """
    zone_data = {}
    # v1：Bo和Kv用相同判据
    if version == 'v1':
        Kv = [-np.inf, 6e-7, np.inf]
        Bo = [-np.inf, 6e-7, np.inf]
    
    # v2：Kv用Murphy判据，Bo用Jackson判据（不考虑传热恢复）
    if version == 'v2':
        Kv = [-np.inf, 9.5e-7, 4e-6, np.inf]
        Bo = [-np.inf, 6e-7, 8e-6, np.inf]

    # Kv数和Bo数均用Jiang判据
    if version == 'Jiang':
        Kv = [-np.inf, 6e-7, np.inf]
        Bo = [-np.inf, 2e-7, np.inf]

    # Bo数用Jackson判据（不考虑传热强化）
    # Kv用Murphy判据（考虑不可忽略时的流动加速）
    if version == 'JacksonMurphy':
        Kv = [-np.inf, 9.5e-7, np.inf]
        Bo = [-np.inf, 6e-7, np.inf]
    
    # 各文献排列组合结果
    if version == 'GUO_v1':
        Kv = [-np.inf, 2e-7, np.inf]
        Bo = [-np.inf, 1.2e-7, 5e-7, np.inf]

    if version == 'GUO_v2':
        Kv = [-np.inf, 2e-7, np.inf]
        Bo = [-np.inf, 1.2e-7, 1e-6, np.inf]

    if version == 'GUO_v3':
        Kv = [-np.inf, 6e-7, np.inf]
        Bo = [-np.inf, 3e-6, 1e-4, np.inf]

    zone_data = Kv_Bo_split(data, Kv, Bo)
    
    return zone_data

# 分层抽样
def stratify(data, Ntotal, Nlayer, weight=None, seed=None):
    '''
    根据给定条件，完成分层抽样。
    即将原始数据raw_set分为Nlayer层，每层的数量由权重weight确定，总共抽取Ntotal个点。
    注: Ntotal太大，容易报错。

    Args:
        raw_set ::  DataFrame, 原始数据
        Ntotal  ::  int, 抽取的数据点总数
        Nlayer  ::  int, 分层抽样的层数
        weight  ::  nd.array, 权重
        seed    ::  int, 随机数种子

    Returns:
        res     ::  DataFrame, 抽样结果
    '''
    # 设置随机数
    random.seed(seed)
    raw_set = data.copy()

    # 确定weight是数组格式，且总和为1，长度与层数相同
    if weight is not None:
        if isinstance(weight, np.ndarray):
            print('Weight must be a np.ndarray type!')
            return
        elif np.sum(weight) < 0.99:
            print('Sum of weight must be 1!')
            return
        elif len(weight) != Nlayer:
            print('Number of weight element should be Nlayer!')
            return
    else:
        weight = np.ones(Nlayer) / Nlayer  # 没有规定weight，即认为各层权重相同
    
    # # 确保idx是整数，以采用Dataframe的iloc函数——现在取消idx
    # if type(idx) is not int:
    #     print('idx for location should be int!')
    #     return
    
    # 确定每层抽取的样本数，最后一层应该覆盖剩余的所有点
    Nsample = np.array([int(n) for n in np.ceil(weight*Ntotal)])
    Nsample[-1] = np.abs(Ntotal - np.sum(Nsample[:-1]))
    
    # 由于不同权重下每个层的间隔不同，需要单独计算每个层的长度以及开始索引
    layer_idx = []
    a = 0
    for n in np.round(weight*raw_set.shape[0]):
        a = int(n) + a
        layer_idx.append(a)
    layer_idx = np.array(layer_idx)
    layer_idx[-1] = raw_set.shape[0]
    
    # 开始分层
    res = pd.DataFrame()
    # 将待操作的某层提取成layer，计算每层各样本间的平均距离Davg
    for i in range(Nlayer):
        if i == 0:
            layer = raw_set.iloc[:layer_idx[i],:]
            Davg = np.round(layer_idx[i] / Nsample[i])
        else:
            layer = raw_set.iloc[layer_idx[i-1]:layer_idx[i],:]
            Davg = np.round((layer_idx[i] - layer_idx[i-1]) / Nsample[i])
        
        REDO = True
        while REDO:
            sample_idx = np.array(random.sample(range(len(layer)), Nsample[i]))  # 抽取样本的索引
            sample_idx.sort()  # 对索引排序
            
            for j in range(1, len(sample_idx)):
                # 若某两个样本间距离小于0.1Davg，则重新抽样
                if sample_idx[j] - sample_idx[j-1] < Davg/10:
                    REDO = True
                    break
            REDO = False
            # 将每层的抽样堆叠起来
            res = pd.concat([res, layer.iloc[sample_idx, :]])
    
    return res

def stratify_Luofeng_IRcases(data, num_points_per_case=20, num_section=10, random_state=0):
    # 对罗峰的红外数据进行分层抽样
    luofeng_cases = [x for x in data['source'].unique() if 'LuoFeng' in x]
    print('Number of Luo Feng cases:', len(luofeng_cases))

    for i,case in enumerate(luofeng_cases):
        subset = data[data['source'] == case]  # 提取某工况的所有数据
        # 如果该工况数据量大于1000，则认为是红外数据，需要标识和抽样
        if subset.shape[0] > 1000:
            temp = stratify(subset, num_points_per_case, num_section, seed=random_state)
            # 删除红外工况的原始数据，将抽样后的数据储存在luofeng_data中
            data.drop(index=data[data['source'] == case].index.tolist(), inplace=True)
            data = pd.concat([data, temp])
    
    data.reset_index(drop=True, inplace=True)

    return data

# ==============================================================================================
# 数据预处理函数
# ==============================================================================================
def set_OneHotEncoder(data, cols):
    '''
    对无序离散特征哑编码

    Args:
        data :: Dataframe, 待处理数据集
        cols :: str or list, 呀编码特征名
    
    Return:
        data :: Dataframe, 编码后的数据集
    '''
    cols = np.array(cols).reshape((-1))
    # col_vals = np.array(list(map(str, data[col].unique())))
    # print(col_vals)
    for col in cols:
        feat_name = []
        for uq in data[col].unique():
            feat_name.append(col + '_' + str(uq))
        onehot_array = OneHotEncoder(sparse=False).fit_transform(data[[col]]).astype(int)
        temp = pd.DataFrame(onehot_array, columns=feat_name)
        data.drop(columns=col, inplace=True)
        data = pd.concat([temp, data], axis=1)
    
    return data

# ==============================================================================================
# 分割数据集-训练集、验证集和测试集
# ==============================================================================================    
def train_val_test_split(x, y, train_size, val_size, test_size, sort_index=True, seed=None, shuffle=False, stratify=None):
    '''
    根据工况条件，访问NIST数据库读取物性参数。适用于数据集中包含多种工况的情况。

    Args:
        train_size :: float or int, 训练集大小
        val_size   :: float or int, 验证集大小
        test_size  :: float or int, 测试集大小
        seed       :: int, 随机状态
        sort_index :: bool, 是否对抽样完的数据集排序, 只针对Dataframe
        shuffle    :: bool, 是否分层抽样, True的时候stratify不能为None
        stratify   :: array-like, 分层抽样的依据

    Returns:
        x_train    :: array-like, 训练集特征
        x_val      :: array-like, 验证集特征
        x_test     :: array-like, 测试集特征
        y_train    :: array-like, 训练集标签
        y_val      :: array-like, 验证集标签
        y_test     :: array-like, 测试集标签
    '''
    # 确保训练/验证/测试集的大小符合规格
    if train_size + val_size + test_size != 1:
        print(f'The sum of train_size, val_size and test_size = {train_size + val_size + test_size}, should be in the 1.')
    
    # 如果验证集缺失，只返回训练集和测试集
    if val_size is None:
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, 
        random_state=seed, shuffle=shuffle, stratify=stratify)
        if sort_index:
            x_train, x_test, y_train, y_test = x_train.sort_index(), x_test.sort_index(), y_train.sort_index(), y_test.sort_index()
        return x_train, x_test, y_train, y_test

    else:
        # 划分测试集
        x_left, x_test, y_left, y_test = train_test_split(x, y, test_size=test_size, 
        random_state=seed, shuffle=shuffle, stratify=stratify)

        # 划分训练集和验证集
        x_train, x_val, y_train, y_val = train_test_split(x_left, y_left, train_size=train_size/(val_size+train_size), 
        random_state=seed, shuffle=shuffle, stratify=stratify)

        if sort_index:
            x_train, x_val, x_test, y_train, y_val, y_test = x_train.sort_index(), x_val.sort_index(), x_test.sort_index(), y_train.sort_index(), y_val.sort_index(), y_test.sort_index()
        
        return x_train, x_val, x_test, y_train, y_val, y_test

# ==============================================================================================
# 多工况物性参数计算函数
# ==============================================================================================    

# def load_NIST_prop(prop, fluid='CarbonDioxide', CALCULATE_DR=True):
#     '''
#     根据工况条件，访问NIST数据库读取物性参数。适用于数据集中包含多种工况的情况。

#     Args:
#         prop  :: 基本工况条件
#         fluid :: 流体名称

#     Returns:
#         prop  :: DataFrame, 输出数据集中增加的物性参数有 
#                     viscosity_in, 
#                     conductivity_b,     Cp_b,     density_b,     enthalpy_b,      Pr_b,     Re_b,     viscosity_b,     expansivity_b,
#                     conductivity_w,     Cp_w,     density_w,     enthalpy_w,      Pr_w,     Re_w,     viscosity_w,     expansivity_w,
#                     conductivity_ratio, Cp_ratio, density_ratio, enthalpy_ratio,  Pr_ratio, Re_ratio, viscosity_ratio, expansivity_ratio

#     '''

#     # 将状态参数由dataframe转化成ndarray格式
#     Tb = np.array(prop['Tb'])
#     Tw = np.array(prop['Tw'])
#     Tin = np.array(prop['T_in'])
#     p = np.array(prop['p_in']) # 忽略沿程压降，认为沿程压力与入口相同
    
#     # 动力粘度
#     prop['viscosity_in'] = property_from_NIST('viscosity', Tin, p, fluid) # Pa s

#     # 基于流体温度的热物性
#     prop['conductivity_b'] = property_from_NIST('conductivity', Tb, p, fluid)
#     prop['Cp_b'] = property_from_NIST('Cpmass', Tb, p, fluid)
#     prop['density_b'] = property_from_NIST('Dmass', Tb, p, fluid)
#     prop['enthalpy_b'] = property_from_NIST('H', Tb, p, fluid)
#     prop['Pr_b'] = property_from_NIST('Prandtl', Tb, p, fluid)
#     prop['viscosity_b'] = property_from_NIST('viscosity', Tb, p, fluid)
#     prop['expansivity_b'] = property_from_NIST('isobaric_expansion_coefficient', Tb, p, fluid)
#     prop['Re_b'] = prop['Re_in'] * prop['viscosity_in'] / prop['viscosity_b']

#     # 基于壁面温度的热物性
#     prop['conductivity_w'] = property_from_NIST('conductivity', Tw, p, fluid)
#     prop['Cp_w'] = property_from_NIST('Cpmass', Tw, p, fluid)
#     prop['density_w'] = property_from_NIST('Dmass', Tw, p, fluid)
#     prop['enthalpy_w'] = property_from_NIST('H', Tw, p, fluid)
#     prop['Pr_w'] = property_from_NIST('Prandtl', Tw, p, fluid)
#     prop['viscosity_w'] = property_from_NIST('viscosity', Tw, p, fluid)
#     prop['expansivity_w'] = property_from_NIST('isobaric_expansion_coefficient', Tw, p, fluid)
#     prop['Re_w'] = prop['Re_in'] * prop['viscosity_in'] / prop['viscosity_w']
#     # prop['T_i/o_ratio'] = prop['T_in']/prop['T_out']

#     # 计算比参数
#     prop['conductivity_ratio'] = prop['conductivity_w'] / prop['conductivity_b']
#     prop['density_ratio'] = prop['density_w'] / prop['density_b']
#     prop['enthalpy_ratio'] = prop['enthalpy_w'] / prop['enthalpy_b']
#     prop['Pr_ratio'] = prop['Pr_w'] / prop['Pr_b']
#     prop['viscosity_ratio'] = prop['viscosity_w'] / prop['viscosity_b']
#     prop['expansivity_ratio'] =  prop['expansivity_w'] / prop['expansivity_b']
#     prop['Re_ratio'] = prop['Re_w'] / prop['Re_b']

#     # 计算差值参数
#     ## 径向梯度
#     if CALCULATE_DR:
#         prop['conductivity_dr'] = prop['conductivity_w'] - prop['conductivity_b']
#         prop['density_dr'] = prop['density_w'] - prop['density_b']
#         prop['enthalpy_dr'] = prop['enthalpy_w'] - prop['enthalpy_b']
#         prop['Pr_dr'] = prop['Pr_w'] - prop['Pr_b'] 
#         prop['viscosity_dr'] = prop['viscosity_w'] - prop['viscosity_b']
#         prop['expansivity_dr'] = prop['expansivity_w'] - prop['expansivity_b']
#         prop['Re_dr'] = prop['Re_w'] - prop['Re_b']


#     # 计算比热容的比值,Cp_bar表示壁面焓变化到流体焓所需的平均热量
#     Cp_bar = (prop['enthalpy_w'] - prop['enthalpy_b']) / (prop['Tw'] - prop['Tb'])
#     prop['Cp_ratio'] = Cp_bar / prop['Cp_b']

#     return prop


# 读取基于Tb的物性参数
def load_NIST_prop_from_Tb(prop, Tb_cal='Tb', fluid='CarbonDioxide'):
    '''
    根据工况条件, 访问NIST数据库读取基于流体温度Tb的物性参数。
    适用于数据集中包含多种工况的情况。

    Args:
        prop   :: DataFrame, 基本工况条件, 特征参数包括T_in, Tb, p_in, Re_in
        Tb_cal :: str, 待计算的流体温度
        fluid  :: str, 流体名称

    Returns:
        prop  :: DataFrame, 输出数据集中增加的物性参数有 
                    viscosity_in, 
                    conductivity_b,     Cp_b,     density_b,     enthalpy_b,      Pr_b,     Re_b,     viscosity_b,     expansivity_b
    '''

    # 将状态参数由dataframe转化成ndarray格式
    Tin = np.array(prop['T_in'])
    Tb = np.array(prop[Tb_cal])
    p = np.array(prop['p_in']) # 忽略沿程压降, 认为沿程压力与入口相同

    # 基于壁面温度的热物性
    Tb_suf = Tb_cal.lstrip('Tb')
    
    # 动力粘度, 计算沿程雷诺数
    if 'viscosity_in' not in prop.columns:
        prop['viscosity_in'] = property_from_NIST('viscosity', Tin, p, fluid) # Pa s

    # 如果所有值为nan，创造空列
    if prop[Tb_cal].isnull().all():
        prop_b_cols = ['conductivity_b'+Tb_suf, 'Cp_b'+Tb_suf, 'density_b'+Tb_suf, 
        'enthalpy_b'+Tb_suf, 'Pr_b'+Tb_suf, 'viscosity_b'+Tb_suf, 'expansivity_b'+Tb_suf, 'Re_b'+Tb_suf]
        prop[prop_b_cols] = np.nan
        return prop


    # 基于流体温度的热物性
    prop['conductivity_b'+Tb_suf] = property_from_NIST('conductivity', Tb, p, fluid)
    prop['Cp_b'+Tb_suf] = property_from_NIST('Cpmass', Tb, p, fluid)
    prop['density_b'+Tb_suf] = property_from_NIST('Dmass', Tb, p, fluid)
    prop['enthalpy_b'+Tb_suf] = property_from_NIST('H', Tb, p, fluid)
    prop['Pr_b'+Tb_suf] = property_from_NIST('Prandtl', Tb, p, fluid)
    prop['viscosity_b'+Tb_suf] = property_from_NIST('viscosity', Tb, p, fluid)
    prop['expansivity_b'+Tb_suf] = property_from_NIST('isobaric_expansion_coefficient', Tb, p, fluid)
    prop['Re_b'+Tb_suf] = prop['Re_in'] * prop['viscosity_in'] / prop['viscosity_b'+Tb_suf]

    return prop

# 读取基于Tw的物性参数
def load_NIST_prop_from_Tw(prop, Tw_cal='Tw', fluid='CarbonDioxide'):
    '''
    根据工况条件, 访问NIST数据库读取基于壁面温度Tw的物性参数。
    适用于数据集中包含多种工况的情况。
    对于旋转实验, Tw应指定。

    Args:
        prop   :: DataFrame, 基本工况条件, 特征参数包括Tw, p_in, Re_in
        Tw_cal :: str, 待计算的壁温
        fluid  :: str, 流体名称

    Returns:
        prop  :: DataFrame, 输出数据集中增加的物性参数有  
                    conductivity_w,     Cp_w,     density_w,     enthalpy_w,      Pr_w,     Re_w,     viscosity_w,     expansivity_w
    '''

    # 将状态参数由dataframe转化成ndarray格式
    Tw = np.array(prop[Tw_cal])
    p = np.array(prop['p_in']) # 忽略沿程压降, 认为沿程压力与入口相同 
    Tw_suf = Tw_cal.lstrip('Tw')

    # 如果所有值为nan，创造空列
    if prop[Tw_cal].isnull().all():
        prop_w_cols = ['conductivity_w'+Tw_suf, 'Cp_w'+Tw_suf, 'density_w'+Tw_suf, 
        'enthalpy_w'+Tw_suf, 'Pr_w'+Tw_suf, 'viscosity_w'+Tw_suf, 'expansivity_w'+Tw_suf, 'Re_w'+Tw_suf]
        prop[prop_w_cols] = np.nan
        return prop
    
    # 基于壁面温度的热物性
    prop['conductivity_w'+Tw_suf] = property_from_NIST('conductivity', Tw, p, fluid)
    prop['Cp_w'+Tw_suf] = property_from_NIST('Cpmass', Tw, p, fluid)
    prop['density_w'+Tw_suf] = property_from_NIST('Dmass', Tw, p, fluid)
    prop['enthalpy_w'+Tw_suf] = property_from_NIST('H', Tw, p, fluid)
    prop['Pr_w'+Tw_suf] = property_from_NIST('Prandtl', Tw, p, fluid)
    prop['viscosity_w'+Tw_suf] = property_from_NIST('viscosity', Tw, p, fluid)
    prop['expansivity_w'+Tw_suf] = property_from_NIST('isobaric_expansion_coefficient', Tw, p, fluid)
    prop['Re_w'+Tw_suf] = prop['Re_in'] * prop['viscosity_in'] / prop['viscosity_w'+Tw_suf]
    # prop['T_i/o_ratio'+Tw_suf] = prop['T_in']/prop['T_out']

    return prop

# 读取基于Tw的物性参数
def load_NIST_prop_ratio(prop, Tb_cal='Tb', Tw_cal='Tw'):
    '''
    根据工况条件, 访问NIST数据库读取基于壁面温度Tw的物性参数。
    适用于数据集中包含多种工况的情况。
    对于旋转实验, Tw应指定。

    Args:
        prop   :: DataFrame, 基本工况条件, 特征参数包括Tw, p_in, Re_in
        Tb_cal :: str, 待计算的流体温度
        Tw_cal :: str, 待计算的壁温

    Returns:
        prop  :: DataFrame, 输出数据集中增加的物性参数有  
                    conductivity_ratio,     Cp_ratio,     density_ratio,     enthalpy_ratio,      Pr_ratio,     Re_ratio,     viscosity_ratio,     expansivity_ratio
    '''
    # 计算比参数
    Tb_suf = Tb_cal.lstrip('Tb')
    Tw_suf = Tw_cal.lstrip('Tw')

    prop['conductivity_ratio'+Tw_suf] = prop['conductivity_w'+Tw_suf] / prop['conductivity_b'+Tb_suf]
    prop['density_ratio'+Tw_suf] = prop['density_w'+Tw_suf] / prop['density_b'+Tb_suf]
    prop['enthalpy_ratio'+Tw_suf] = prop['enthalpy_w'+Tw_suf] / prop['enthalpy_b'+Tb_suf]
    prop['Pr_ratio'+Tw_suf] = prop['Pr_w'+Tw_suf] / prop['Pr_b'+Tb_suf]
    prop['viscosity_ratio'+Tw_suf] = prop['viscosity_w'+Tw_suf] / prop['viscosity_b'+Tb_suf]
    prop['expansivity_ratio'+Tw_suf] =  prop['expansivity_w'+Tw_suf] / prop['expansivity_b'+Tb_suf]
    prop['Re_ratio'+Tw_suf] = prop['Re_w'+Tw_suf] / prop['Re_b'+Tb_suf]

    # 计算比热容的比值,Cp_bar表示壁面焓变化到流体焓所需的平均热量
    Cp_bar = (prop['enthalpy_w'+Tw_suf] - prop['enthalpy_b'+Tb_suf]) / (prop[Tw_cal] - prop[Tb_cal])
    prop['Cp_ratio'+Tw_suf] = Cp_bar / prop['Cp_b'+Tb_suf]

    return prop

# 读取物性参数
def load_NIST_prop(prop, Tb_cal='Tb', Tw_cal='Tw', fluid='CarbonDioxide', CALCULATE_DR=True):
    '''
    根据工况条件, 访问NIST数据库读取物性参数。适用于数据集中包含多种工况的情况。

    Args:
        prop   :: Dataframe, 基本工况条件, 包含的参数
                  T_in, p_in, Tb, Tw_cal
        Tb_cal :: str, 待计算的流体温度
        Tw_cal :: str, 待计算的壁温
        fluid  :: str, 流体名称

    Returns:
        prop   :: DataFrame, 输出数据集中增加的物性参数有 
                    viscosity_in, 
                    conductivity_b,     Cp_b,     density_b,     enthalpy_b,      Pr_b,     Re_b,     viscosity_b,     expansivity_b,
                    conductivity_w,     Cp_w,     density_w,     enthalpy_w,      Pr_w,     Re_w,     viscosity_w,     expansivity_w,
                    conductivity_ratio, Cp_ratio, density_ratio, enthalpy_ratio,  Pr_ratio, Re_ratio, viscosity_ratio, expansivity_ratio   
    '''

    # 基于流体温度的热物性
    prop = load_NIST_prop_from_Tb(prop, Tb_cal, fluid)

    # 基于壁面温度的热物性
    prop = load_NIST_prop_from_Tw(prop, Tw_cal, fluid)

    # 计算比参数
    prop = load_NIST_prop_ratio(prop, Tb_cal, Tw_cal)

    # 计算差值参数
    ## 径向梯度
    if CALCULATE_DR:
        prop['conductivity_dr'] = prop['conductivity_w'] - prop['conductivity_b']
        prop['density_dr'] = prop['density_w'] - prop['density_b']
        prop['enthalpy_dr'] = prop['enthalpy_w'] - prop['enthalpy_b']
        prop['Pr_dr'] = prop['Pr_w'] - prop['Pr_b'] 
        prop['viscosity_dr'] = prop['viscosity_w'] - prop['viscosity_b']
        prop['expansivity_dr'] = prop['expansivity_w'] - prop['expansivity_b']
        prop['Re_dr'] = prop['Re_w'] - prop['Re_b']

    return prop


def calculate_Tpc(T_lb, T_ub, p, num=5000, fluid='CO2'):
    '''
    在T_lb ~ T_ub范围内计算压力p下的流体准临界温度。

    Args:
        T_lb  ::  float, 温度区间上限
        T_ub  ::  float, 温度区间下限
        p     ::  float, 压力
        num   ::  int, 区间点数
        fluid ::  str, 流体种类

    Return:
        Tpc   ::  float, 准临界温度 
    '''

    T = np.linspace(T_lb, T_ub, num)
    Cp_Decane = property_from_NIST('Cpmass', T, p, fluid)
    Tpc = T[np.argmax(Cp_Decane)]
    if Tpc == T_lb:
        print('T_lb is too high to search for Tpc.')
        
    elif Tpc == T_ub:
        print('T_ub is too low to search for Tpc.')
    
    return Tpc



def calculate_nondimensional_numbers(prop, tube_diameter=None, heat_flux=None, fluid='CarbonDioxide', NU_CALCULATE=True):
    '''
    根据热物性参数，计算无量纲参数。

    Args:
        prop          :: load_NIST_prop_from_mixed_cases 返回的物性参数
        tube_diameter :: float, 管径(mm), 当输入数据集prop中不包含该特征时使用
        heat_flux     :: float, 热流密度(W/m2), 当输入数据集prop中不包含该特征时使用

    Returns:
        prop :: DataFrame, 在数据集中增加物性参数 Gr, Bo, Kv, Nu
    '''
    
    # 若数据集prop中不含管径或热流密度，则接收输入参数
    d = prop['tube_diameter'].copy() if 'tube_diameter' in prop.columns else tube_diameter
    qw = prop['heat_flux'].copy() if 'heat_flux' in prop.columns else heat_flux

    # 修正管径单位
    if d.mean() > 0.1:
        d*=1e-3
    
    prop['Gr'] = 9.81 * prop['expansivity_b'] * qw * (d**4) / prop['conductivity_b'] / (prop['viscosity_b'] / prop['density_b'])**2
    prop['Bo'] = prop['Gr'] / prop['Re_b']**3.425 / prop['Pr_b']**0.8
    prop['Kv'] = 4 * qw * d * prop['expansivity_b'] / prop['Re_b']**2 / prop['viscosity_b'] / prop['Cp_b']
    if NU_CALCULATE:
        prop['Nu'] = qw * d / (prop['conductivity_b'] * (prop['Tw'] - prop['Tb']))

    return prop

def calculate_properties_BuAc(prop):
    '''
    根据读取到的物性参数，计算 Bu, Ac

    Args:
        prop :: load_NIST_prop_from_DNS_case 返回的物性参数？？DNS

    Returns:
        prop :: DataFrame, 在数据集中增添 Bu Ac 两列
    '''

    temp = prop['density_ratio']**(-0.5) * prop['viscosity_ratio']
    prop['Bu'] = prop['Gr'] / (prop['Re_b']**2.625 * prop['Pr_b']**0.4) * temp

    q_plus = prop['expansivity_b']*prop['heat_flux']/(prop['Cp_b']*prop['mass_flow_rate'])
    prop['Ac'] = q_plus / prop['Re_b']**0.625 * temp
    
    return prop


def variance_filter(data, criteria=0, method='cv', skipna=True):
    '''
    变异系数/方差接近于0或等于0, 认为数据变化不大, 对目标参数无影响
    方差过滤受异常值影响很大, 另外方差小的特征并非不重要, 为此阈值一般取0
    '''
    if method == 'cv':
        filt = data.std(skipna=skipna)/data.mean(skipna=skipna)  # 计算变异系数
    elif method == 'var':
        filt = data.std(skipna=skipna)  # 计算方差
    
    feats_filt = filt[(filt.isna())|(abs(filt) <= criteria)].index  # 获取过滤的特征
    data.drop(columns=feats_filt, inplace=True)  # 删除特征
    print('Features delete:', list(feats_filt))

    return

def Nu_from_JiangLi_correlation(data, threshold=1e-3, max_iter=1000, fluid='CarbonDioxide'):
    """
    根据Jiang-Li关联式计算Nu。
        Ref. 李志辉, 姜培学. 超临界压力 CO2 在垂直管内对流换热准则关联式[J]. 核动力工程, 2010, 31(05):72-75.
    
    **目前基本没有用到
    """
    # Load Boundray Conditions
    d = data['tube_diameter'] * 1e-3  # mm, tube diameter
    p = data['p_in'].to_numpy()       # Pa, pressure
    Tin = data['T_in'].to_numpy()     # K, inlet Temperature
    Tb = data['Tb'].to_numpy()       # K, bulk Temperature
    qw = data['heat_flux']            # W/m^2, surface heat flux
    Re_in = data['Re_in']             # Re, characteristic length = diameter of the tube
    Tpc = CP.PropsSI(fluid, "Tcrit")  # K, critical temperature
    
    # Load X coordinates
    x_dim = data['x/d']

    # Calculate Bulk Temperature
    viscosity_in = CP.PropsSI('viscosity', 'T', Tin, 'P', p, fluid)

    # Calculate properties
    result = pd.DataFrame()

    # Calculate Nusselt Number from Jiang-Li correlation
    # 向上流动
    res = []
    Tw_temp = 296*np.ones(int(len(x_dim)))
    for count in range(max_iter):
        Tw = Tw_temp

        # Load new wall dataerties
        data['density_w'] = CP.PropsSI('Dmass', 'T', data['Tw'].to_numpy(), 'P', p, fluid)
        data['enthalpy_w'] = CP.PropsSI('H', 'T', data['Tw'].to_numpy(), 'P', p, fluid)
        data['n'] = 0.4

        # Condition 1
        ind1 = ((data['Tb'] <= data['Tw']) & (data['Tw'] <= Tpc)) | ((1.2*Tpc <= data['Tb']) & (data['Tb'] <= data['Tw']))
        data['n'][ind1] = 0.4
        # Condition 2
        ind2 = (data['Tb'] <= Tpc) & (Tpc <= data['Tw'])
        data['n'][ind2] = 0.4 + 0.2*(data['Tw'][ind2]/Tpc - 1)
        # Condition 3
        ind3 = (Tpc <= data['Tb']) & (data['Tb'] <= 1.2*Tpc) & (data['Tb'] <= data['Tw'])
        data['n'][ind3] = 0.4 + 0.2*(data['Tw'][ind3]/Tpc - 1) * (1 - 5*(data['Tb']/Tpc - 1))
        
        h_temp = qw/(data['Tw'] - data['Tb'])
        Nu_temp = h_temp*d/data['conductivity_b']

        el = 1 + 2.35*data['Pr_b']**(-0.4) * result['Re_b']**(-0.15) * (result['x/d'])**(-0.6) * np.exp((-0.39) * result['Re_b']**(-0.1) * result['x/d'])

        Cp_bar = (data['enthalpy_w'] - data['enthalpy_b'])/(data['Tw'] - data['Tb'])

        result['Cp_ratio'] = Cp_bar/data['Cp_b']
        result['density_ratio'] = data['density_w']/data['density_b']

        Nu_f = 0.0183 * result['Re_b']**0.82 * data['Pr_b']**0.5 * (result['density_ratio'])**0.3 * (result['Cp_ratio'])**data['n'] * el

        result['Nu'] = Nu_f * (np.abs(1 - result['Bo']**0.1 * result['Cp_ratio']**(-0.009) * (result['density_ratio'])**0.35 * (Nu_temp/Nu_f)**(-2)))**0.46
        h = result['Nu'] * data['conductivity_b'] / (d)
        
        Tw_temp = qw/h + data['Tb']
        
        residual = np.absolute(Tw_temp - data['Tw'])
        
        res.append(np.max(residual))

        if np.max(residual) < threshold:
            print('maximum residual = ', np.max(residual))
            print('threshold = ', threshold)
            print('residual < threshold, iterations complete')
            break
        
        if count%250 == 0:
            print(count, '/', max_iter)

    if count == max_iter-1:
        print('maximum number of iterations reached')
        print('current maximum residual %.2e' %np.max(residual))
    
    return 


