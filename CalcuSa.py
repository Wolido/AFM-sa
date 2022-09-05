#!/usr/bin/env python
# -*- coding:utf8 -*-

import pandas as pd
import numpy as np
from pandas.io.html import re
from tables import file
from tqdm import main, trange
import math
import warnings
import os
import readline
import time

# 忽略计算圆柱表面坐标返回nan的警告信息
warnings.filterwarnings('ignore')


# 读取数据
def readdata(data_dir):
    data = pd.read_csv('./'+data_dir, header=None)  # 读取数据，此行需修改
    data_len = len(data[1])  # 记录数据长度，正方形数据，长宽一致，后续多次使用此数据

    data_z_mean = []  # 对z值进行按列平均，用于计算圆柱参数
    for i in range(256):
        data_z_mean.append(data[i].mean())
    data_z_mean = np.array(data_z_mean)  # 转为array
    return data, data_len, data_z_mean

# 初始化搜索范围
def initrange(data_len):
    sample_len = 2000  # 样品真实采集尺寸，单位纳米
    data_x = np.arange(data_len)*sample_len/data_len  # 数据x轴

    # 初始搜索参数范围设定，初始步长为100，尽量不要修改初始步长。
    x_cir = np.arange(-5000, 5000, 100)  # 圆柱圆心x偏移，一般为0-2000, 最后一位为参数步长
    z_cir = np.arange(-9000, -2000, 100)  # 圆柱圆心z偏移，一般为负的纤维直径附近
    r_cir = np.arange(2000, 16000, 100)  # 圆柱半径
    return x_cir, z_cir, r_cir, data_x


# 一组数据的Ra计算函数，传入参数为修正后的样品z值和圆柱的z值, 数据类型均为array
def Ra(z_new, cir_z):
    delta_h_sum = abs(z_new - cir_z).sum()
    return delta_h_sum / data_len  # 返回数据为一行的Ra


# 把样品表面的坐标修正到以(0,0)为圆心的圆柱的位置, 数据类型均为array
def regre(x, z, z_data):  # 修正坐标为x、z
    x_reg = data_x - x  # 所有情况下均对x的均匀数据做修正
    z_reg = z_data - z  # 粗算时对data_z_mean做修正，精算时对每一行z值做修正
    return x_reg, z_reg


# 获得圆柱的表面z坐标，x数据类型为array
def yuanzhu(x, r):  # x是经过regre函数修正过的新坐标
    line_yz = (r**2 - x**2)**0.5
    return line_yz


# 粗算，用于计算圆柱的x，z，r坐标
def findxzr(X, Z, R):
    # 读入三个坐标参数的长度，用于确定最后记录数据的array的shape
    x_len = len(X)
    z_len = len(Z)
    r_len = len(R)
    results = np.zeros((x_len, z_len, r_len))

    for xi in trange(len(X)):
        for zi in range(len(Z)):
            for ri in range(len(R)):
                x_hg, z_hg = regre(X[xi], Z[zi], data_z_mean)  # x_hg和z_rg为修正后的样品表面坐标
                cir_z_new = yuanzhu(x_hg, R[ri])  # cir_z_new为圆柱表面的坐标
                # 计算表面粗糙度，但是可能计算结果为nan，所以先暂存，确认后录入results
                ra_temp = Ra(z_hg, cir_z_new)
                if math.isnan(ra_temp):
                    results[xi, zi, ri] = 10000
                else:
                    results[xi, zi, ri] = ra_temp

    xzr = np.unravel_index(np.argmin(results), results.shape)  # 取xyz最小值坐标
    return xzr


# 将搜索范围进行迭代更新
def changecir(xzr, step):
    global x_cir
    global z_cir
    global r_cir

    x_cir = np.arange(x_cir[xzr[0]]-20*step, x_cir[xzr[0]]+20*step, step)
    z_cir = np.arange(z_cir[xzr[1]]-20*step, z_cir[xzr[1]]+20*step, step)
    r_cir = np.arange(r_cir[xzr[2]]-20*step, r_cir[xzr[2]]+20*step, step)


# 精算
def slowcal(xzr):
    x_cal = x_cir[xzr[0]]  # xzr用于计算的坐标值
    z_cal = z_cir[xzr[1]]
    r_cal = r_cir[xzr[2]]
    ra_all = 0
    for i in range(data_len):
        data_row = data.iloc[i, :]  # 按行读取数据，每行数据均计算表面粗糙度
        x_hg, z_hg = regre(x_cal, z_cal, z_data=data_row)
        cir_z_new = yuanzhu(x_hg, r_cal)
        ra_temp = Ra(z_hg, cir_z_new)
        if math.isnan(ra_temp):
            print('计算错误')
            break
        else:
            ra_all += ra_temp
    final_result = ra_all / data_len
    return final_result


# 表面粗糙度计算主参数
def maincal(file_name):
    global data
    global data_len
    global data_z_mean
    global x_cir
    global z_cir
    global r_cir
    global data_x

    data, data_len, data_z_mean = readdata(file_name)
    x_cir, z_cir, r_cir, data_x = initrange(data_len)
    print('参数搜索中，搜索步长100nm')
    xzr = findxzr(x_cir, z_cir, r_cir)
    changecir(xzr, 50)
    print('\n参数搜索中，搜索步长50nm')
    xzr50 = findxzr(x_cir, z_cir, r_cir)
    changecir(xzr50, 20)
    print('\n参数搜索中，搜索步长20nm')
    xzr20 = findxzr(x_cir, z_cir, r_cir)
    changecir(xzr20, 10)
    print('\n参数搜索中，搜索步长10nm')
    xzr10 = findxzr(x_cir, z_cir, r_cir)
    changecir(xzr10, 5)
    print('\n参数搜索中，搜索步长5nm')
    xzr5 = findxzr(x_cir, z_cir, r_cir)
    changecir(xzr5, 1)
    print('\n参数搜索中，搜索步长1nm')
    xzr1 = findxzr(x_cir, z_cir, r_cir)
    ra = round(slowcal(xzr1), 2)
    print('\n表面粗糙度为：{}'.format(ra))
    return ra


# 获取当前文件夹下的csv文件目录
def getlist():
    dirs = os.listdir('./')
    def file_filter(f):
        if f[-4:] == '.csv':
            return True
        else:
            return False
    return list(filter(file_filter,dirs))


# 程序模式选择，单一数据模式或批量数据模式
def modeselect():
    cal_time = time.strftime('%Y%m%d%H%M%S')
    mode = input('是否使用批量处理模式(Y/n):')
    if mode in ('Y', 'y', ''):
        file_list = getlist()
        for file_dir in file_list:
            print('开始进行表面粗糙度计算，文件名{}'.format(file_dir))
            sa = maincal(file_dir)
            with open('./results-{}.txt'.format(cal_time),'a+') as restxt:
                restxt.write('文件名:{}  表面粗糙度:{}'.format(file_dir,sa))
    else:
        file_name = input('请输入样品名(无需输入扩展名):')
        if file_name[-4:] != '.csv':
            file_name = file_name+'.csv'
        if file_name in getlist():
            sa = maincal(file_name)
            with open('./results-{}.txt'.format(cal_time),'a+') as restxt:
                restxt.write('文件名:{}  表面粗糙度:{}'.format(file_name,sa))
        else:
            print('文件名有误，程序结束。')

if __name__ == '__main__':
    modeselect()
