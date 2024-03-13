import numpy as np
import pandas as pd
import data_process

def EWM(data: np.ndarray) -> np.ndarray:
    '''熵权法计算
    Params:
    - data : np.ndarray
        需要进行熵权法计算的判断矩阵,行为评价对象,列为评价指标.
    Returns:
    - Out : np.ndarray
        各个指标的熵权组成的向量.
    '''
    def single_entropy(x):
        '''单项事件的信息熵计算'''
        return 0 if x == 0 else x*np.log(x)
    vec_single_entropy = np.vectorize(single_entropy)

    m,n = data.shape # 传入的判断矩阵为形状m*n
    P = data_process.normalization(data) # 计算概率矩阵P
    E = -(1/np.log(m))*np.sum(vec_single_entropy(P),axis=0) # 计算信息熵E
    D = 1 - E # 计算信息效用值D
    W = D / np.sum(D) # 计算每个指标的熵权W

    return W

def test_EWM():
    data = data_process.load_test_data(2,5) 
    data_proc = data_process.attribute_process(data,["Pos","Pos","Neg"],enable_tol=True)
    w = EWM(data_proc)
    assert data_process.approximate_equal(w,[0.27479,0.32626,0.39895])
    # 验证数据来源于SPSSPRO
