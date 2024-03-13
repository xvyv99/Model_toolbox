import numpy as np
import pandas as pd
import data_process
from EWM import EWM

def TOPSIS(data: np.ndarray,weight: list) -> list[float]:
    assert data.shape[1] == len(weight)
    Data = np.array(data,dtype=float)
    Weight = np.array(weight)
    m,n = Data.shape # 传入的判断矩阵的形状为m*n

    worst_lst = np.min(Data,axis=0)
    best_lst = np.max(Data,axis=0)
    # 计算最优方案与最劣方案
    
    worst_appr = np.sqrt(np.sum(Weight*np.square(Data-worst_lst),axis=1))
    best_appr = np.sqrt(np.sum(Weight*np.square(Data-best_lst),axis=1))
    score_lst = worst_appr/(worst_appr+best_appr) # 评价对象的综合得分指数
    # 计算各评价指标与最劣及最优向量之间的差距,及评价对象与最优方案的接近程度
    
    return worst_lst,best_lst,score_lst

def test_TOPSIS():
    test_data = data_process.load_test_data(1,4)
    weight = [0.1,0.3,0.6]
    data = data_process.attribute_process(test_data,["Neg","Pos","Pos"])
    assert data_process.approximate_equal(TOPSIS(data,weight)[2][:5],np.array([0.61986409,0.48457217,0.52151297,0.64032108,0.66992462])) # 测试综合得分指数前5项是否正确
    assert data_process.approximate_equal(TOPSIS(data,weight)[0],np.array([0.00000105,0.00000103,0.00000109])) # 测试返回的最劣方案
    ewm_weight = EWM(data)
    assert data_process.approximate_equal(TOPSIS(data,ewm_weight)[2][:3],[0.48453862,0.479783,0.54503262]) # 测试与熵权法协同使用产生的结果是否正确
    # 验证数据来源于SPSSPRO 
