import numpy as np
from typing import Tuple

class AHP:
    Judge_matrix: np.ndarray # 判断矩阵
    n: int # 判断矩阵的阶数
    R_I: list[float] = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 0.49, 0.52, 1.54, 1.56, 1.58, 1.59] # 随机一致性指标 R.I.取值表,列表序号对应于矩阵阶数减一

    def __init__(self,judge_matrix: np.ndarray) -> None:
        assert judge_matrix.shape[0] == judge_matrix.shape[1]
        self.Judge_matrix = judge_matrix
        self.n = self.Judge_matrix.shape[0]

    def checkMatrix(self) -> bool:
        '''判断矩阵的数据检查'''
        for x in range(self.n):
            for y in range(x):
                if self.Judge_matrix[x,y]*self.Judge_matrix[y,x] == 1:
                    continue
                else:
                    return False
        return True
        # 检查是否为正互反矩阵
    
    def testConsistency(self) -> Tuple[bool,float]:
        '''一致性检验'''
        eig_val,__ = np.linalg.eig(self.Judge_matrix)
        lambda_max = np.max(eig_val)
        C_I = (lambda_max - self.n) / (self.n - 1)
        C_R = C_I / (self.R_I[self.n-1])
        if C_R < 0.1:
            return True,C_I
        else:
            return False,C_I

    def calcWeight(self,method: str) -> np.ndarray:
        '''计算权重'''
        def root_method():
            '''方根法'''
            weight_ = np.power(np.prod(self.Judge_matrix,axis=1),1/self.n)
            weight = weight_ / np.sum(weight_)
            return weight

        def sum_method():
            '''和积法'''
            standard_matrix_col = self.Judge_matrix / np.sum(self.Judge_matrix)
            weight_ = np.sum(standard_matrix_col,axis=1)
            weight = weight_ / np.sum(weight_)
            return weight

        def eig_method():
            '''特征值法,可能有问题,慎用!'''
            eig_val,eig_vector = np.linalg.eig(self.Judge_matrix)
            weight_ = eig_vector[np.argmax(eig_val)]
            weight = weight_ / np.sum(weight_)
            return weight

        if method == "Root":
            return root_method()
        elif method == "Sum":
            return sum_method()
        elif method == "Eig":
            return eig_method()

def test_AHP():
    '''功能测试,适用于pytest'''
    atol = 1e-3 # 比较精度,保证相差不大时可判定为近似相等
    AHP_1 = AHP(np.array([[1,2,3],[0.5,1,8],[1/3,0.125,1]]))
    AHP_2 = AHP(np.array([[1,0.5,0.25],[2,1,0.5],[4,2,1]]))
    assert AHP_1.testConsistency()[0] == False
    assert AHP_2.testConsistency()[0] == True
    # 一致性检验测试
    assert AHP_1.checkMatrix()
    assert AHP_2.checkMatrix()
    # 矩阵规范性测试
    assert np.all(np.isclose(AHP_1.calcWeight("Root"),np.array([0.48441,0.42317,0.09242]),atol=atol))
    assert np.all(np.isclose(AHP_2.calcWeight("Sum"),np.array([0.14286,0.28571,0.57143]),atol=atol))
    # 权重计算方法测试,验证数据来源于SPSSPRO
