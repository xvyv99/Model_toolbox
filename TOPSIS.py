import numpy as np
from enum import Enum

class Attr(Enum):
    Minimal = 0
    Intermediate = 1
    Interval = 2

class TOPSIS:
    Data: np.ndarray
    Weight: np.ndarray

    def __init__(self,data: np.ndarray,weight: np.ndarray):
        assert data.shape[1] == weight.shape[0] 
        self.Data = data
        self.Weight = weight

    def concordAttr(self,attr_lst: list[Attr]):
        '''同向化指标'''
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
        
        assert len(attr_lst) == self.Weight.shape[0]
        res_matrix = np.zeros_like(self.Data)
        for x in range(self.Data.shape[1]):
            pass
    
    def Normalization(self):
        '''数据归一化'''
        pass

    def calcAssessment(self):
        '''对对象进行评价'''
        def situation():
            '''计算最优方案与最劣方案'''
            pass

def test_TOPSIS():
    pass
