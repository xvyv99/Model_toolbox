import numpy as np
import pandas as pd

def concord(data: np.ndarray,attr_lst: list[str],enable_tol=False) -> np.ndarray:
    '''同向化指标'''
    tol = 0 if enable_tol else 0.0001

    def max_concord(col: np.ndarray):
        max_val = np.max(col)+tol
        min_val = np.min(col)-tol
        return (col-min_val)/(max_val-min_val)

    def min_concord(col: np.ndarray):
        min_val = np.min(col)-tol
        max_val = np.max(col)+tol
        return (max_val-col)/(max_val-min_val)

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
        "Minimal":minimal_attr,
        "Intermediate":intermediate_attr,
        "Interval":interval_attr,
        "Max":max_concord,
        "Min":min_concord
    }
    res = np.empty_like(data)

    for i,x in enumerate(attr_lst):
        res[:,i] = func_dic[x](data[:,i]) 
    return res

class EWM:
    Data: np.ndarray
    def __init__(self,data: np.ndarray):
        '''传入矩阵形状为n*m'''
        self.Data = data
        self.n,self.m = self.Data.shape

    def calcWeight(self) -> np.ndarray:
        def calc_expr(x):
            return 0 if x == 0 else x*np.log(x)
        vec_calc = np.vectorize(calc_expr)
        P = self.Data/np.sum(self.Data,axis=0)
        E = -(1/np.log(self.n))*np.sum((vec_calc(P)),axis=0)
        G = 1 - E
        W = G / np.sum(G)

        return W

def test_EWM():
    atol = 1e-3
    data = pd.read_csv("./test.csv").iloc[:,2:5].astype(float).to_numpy()
    data_process = concord(data,["Max","Max","Min"],enable_tol=True)
    e = EWM(data_process)
    w = e.calcWeight()
    assert np.all(np.isclose(w,[0.27479,0.32626,0.39895],atol=atol))
