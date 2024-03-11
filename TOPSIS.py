import numpy as np
import pandas as pd
from enum import Enum

class Attr(Enum):
    Minimal = 0
    Intermediate = 1
    Interval = 2

class TOPSIS:
    Data: np.ndarray
    Weight: np.ndarray
    n: int #待评价对象数
    m: int #对象的指标数

    def __init__(self,data: pd.DataFrame,weight: list):
        assert data.shape[1] == len(weight)
        self.Data = np.array(data,dtype=np.float64)
        self.Weight = np.array(weight)
        self.n,self.m = self.Data.shape

    def concordAttr(self,attr_lst: list[str]):
        '''同向化指标'''
        def max_concord(col: np.ndarray):
            tol = 0.0001
            max_val = np.max(col)+tol
            min_val = np.min(col)-tol
            return (col-min_val)/(max_val-min_val)

        def min_concord(col: np.ndarray):
            tol = 0.0001
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
                Attr.Minimal:minimal_attr,
                Attr.Intermediate:intermediate_attr,
                Attr.Interval:interval_attr,
                "Max":max_concord,
                "Min":min_concord
        }
        
        assert len(attr_lst) == self.Weight.shape[0]
        for i,x in enumerate(attr_lst):
            self.Data[:,i] = func_dic[x](self.Data[:,i]) 
    
    def Normalization(self) -> None:
        '''数据归一化'''
        col: np.ndarray
        for c in range(self.m):
            col = self.Data[:,c]
            self.Data[:,c] = col / np.sqrt(np.sum(np.square(col)))

    def calcAssessment(self) -> list[float]:
        '''对对象进行评价'''
        worst_lst,best_lst = [],[]
        col: np.ndarray
        for x in range(self.m):
            col = self.Data[:,x]
            worst_lst.append(np.min(col))
            best_lst.append(np.max(col))
        print("Worst:",worst_lst,"Best:",best_lst)
        #计算最优方案与最劣方案
        score_lst = []
        worst_appr,best_appr = None,None
        for x in range(self.n):
            worst_appr = np.sqrt(np.sum(self.Weight*np.square(self.Data[x]-worst_lst)))
            best_appr = np.sqrt(np.sum(self.Weight*np.square(self.Data[x]-best_lst)))
            score_lst.append(worst_appr/(worst_appr+best_appr))
        return score_lst

def test_TOPSIS():
    test_data = pd.read_csv("./test.csv").iloc[:,1:4]
    weight = [0.1,0.3,0.6]
    t = TOPSIS(test_data,weight)
    print(t.Data)
    t.concordAttr(["Min","Max","Max"])
    #t.Normalization()
    print(t.calcAssessment())
test_TOPSIS()
