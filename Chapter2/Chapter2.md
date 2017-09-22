# MachineLearning-ZhouZhihua-notes Chapter 1

# 2.1  
------  
    C(150,500) * C(150,500)  

# 2.2  
------  
    10折交叉检验： 每个样本正反例数目一样，错误率期望50%  
    留一法： 留下一个测试样本，其余样本中有50个与测试类别不同，测试集错误率100%  

# 2.3  
------
    BEP: BEP=P=R  
    F1: F1-(2*P*R)/(P+R)  
    F1(A)>F1(B)  
    BEP(A)>BEP(B)  

# 2.4  
------
    真正例率(TPB): 真实正例被预测为正例的比例  
    查全率(R): 真实正例被预测为正例的比例
    假正例率(FPR): 真实反例被预测为反例的比例  
    查准率(P): 预测为正例的实例中真正正例的比例  

# 2.6  
------
    ROC曲线每个点对应了一个TPR与FPR，此时对应了一个错误率  
    Ecost=(m+∗(1−TPR)∗cost01+m−∗FPR∗cost10)/(m++m−)  
    学习器会选择错误率最小的位置作为截断点  