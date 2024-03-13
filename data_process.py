import numpy as np
import pandas as pd

Test_data_path = "./test.csv"

def load_test_data(start: int,end: int) -> np.ndarray:
    return pd.read_csv(Test_data_path).iloc[:,start:end].astype(float).to_numpy()

def approximate_equal(left,right,atol=1e-3):
    return np.all(np.isclose(left,right,atol=atol))

def normalization(data: np.ndarray,proc_type="deafult",axis=0) -> np.ndarray:
    '''对判断矩阵进行标准化处理
    Params:
    - data : np.ndarray
        需要进行标准化处理的判断矩阵.
    - proc_type : str, optional
        进行标准化的类型,默认为"deafult".
    - axis : int, optional
        给定需要标准化的轴,默认为0.对于二维的情形,axis=0代表按行进行聚合,处理的是列,axis=1代表按列进行聚合,处理的是行.
    Returns:
    - Out : np.ndarray
        标准化处理后判断矩阵.
    '''
    def deafult_normal(x: np.ndarray):
        '''默认归一化,算法为$\frac{x_{ij}}{\sum_{i=1}^{n}x_{ij}}$'''
        return x/np.sum(x,axis=axis)

    def L2_normal(x: np.ndarray):
        '''L2范数归一化'''
        return x / np.sqrt(np.sum(np.square(x),axis=axis))

    proc_dic = {
        "deafult":deafult_normal,
        "L2":L2_normal
    }
    return proc_dic[proc_type](data)

def attribute_process(data: np.ndarray,attr_lst: list[tuple|str],enable_tol=False) -> np.ndarray:
    '''指标处理,默认指标为列向量,即一列为同一个指标.
    Params:
    - data : np.ndarray
        需要进行指标处理的判断矩阵.
    - attr_lst : list[tuple|str]
        该列表的每一项的第0个元素为每列的指标类型,其余元素为相关指标的参数,且与处理相关指标的参数一一对应.
    - enable_tol : bool, optional
        是否兼容一整列都为相同的值得情况,默认关闭(False).
    Returns:
    - Out : np.ndarray
       进行指标处理后判断矩阵.
    '''

    assert data.shape[1] == len(attr_lst)
    tol = 0 if enable_tol else 0.0001 # 兼容一整列都为相同的值的情况,对整体结果影响不大,可忽略不计,来源于SPSSPRO.
    
    # 正向,负向指标归一化处理,来源于SPSSPRO
    def Pos_concord(col: np.ndarray):
        '''正向指标归一化'''
        max_val = np.max(col)+tol
        min_val = np.min(col)-tol
        return (col-min_val)/(max_val-min_val)

    def Neg_concord(col: np.ndarray):
        '''负向指标归一化'''
        min_val = np.min(col)-tol
        max_val = np.max(col)+tol
        return (max_val-col)/(max_val-min_val)

    # 极小型指标,中间型指标,区间型指标同向化处理
    def minimal_attr(col: np.ndarray,offset: float) -> np.ndarray:
        '''极小型指标'''
        return 1/(col + offset)

    def intermediate_attr(col: np.ndarray,min_val: float,max_val: float) -> np.ndarray:
        '''中间型指标'''
        assert min_val <= max_val
        def concord(val):
            if min_val<=val<=(1/2)*(min_val+max_val):
                return 2*(val-min_val)/(max_val-min_val)
            elif (1/2)*(min_val+max_val)<val<=max_val:
                return 2*(max_val-val)/(max_val-min_val)
            else:
                raise Exception("Value out of range.")
        vector_concord = np.vectorize(concord)
        return vector_concord(col)

    def interval_attr(col: np.ndarray,min_val: float,max_val: float,min_tol: float,max_tol: float) -> np.ndarray:
        '''区间型指标'''
        assert min_tol <= min_val <= max_val <= max_tol
        def concord(val):
            if min_val<=val<=max_val:
                return 1
            elif val<min_val:
                return 1-(min_val-val)/(min_val-min_tol)
            elif max_val<val:
                return 1-(val-max_val)/(max_tol-max_val)
            else:
                raise Exception("Value out of range.")
        vector_concord = np.vectorize(concord)
        return vector_concord(col)

    func_dic = {
        # 同向化处理
        "Minimal":minimal_attr,
        "Intermediate":intermediate_attr,
        "Interval":interval_attr,
        # 归一化处理
        "Pos":Pos_concord,
        "Neg":Neg_concord
    } # 两种处理不可混用!

    res = np.empty_like(data) # 为处理后的判断矩阵分配内存
    if type(attr_lst[0]) == str: 
        for i,x in enumerate(attr_lst):
            res[:,i] = func_dic[x](data[:,i]) 
    elif type(attr_lst[0]) == tuple:
        for i,x in enumerate(attr_lst):
            res[:,i] = func_dic[x](data[:,i],*x[1:]) 
    else:
        raise Exception("Type not support!")
            # 对原判断矩阵每一列按照所给指标对应的函数类型进行处理
    return res
