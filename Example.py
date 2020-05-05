import numpy as np
import Algorithm_new as AL
import scipy.sparse as sparse
def Example():
    beta = np.random.normal(1, 0.1, 30)
    freq_1_r = np.random.poisson(0.1, [1000, 100])
    freq_2_r = np.random.poisson(0.1, [1000, 100])
    freq_1_e = np.random.poisson(0.1, [1000, 100])
    freq_2_e = np.random.poisson(0.1, [1000, 100])
    Cov_1 = np.diag(np.random.uniform(0, 1, 100))
    Cov_2 = np.diag(np.random.uniform(0, 1, 100))
    Base = np.random.normal(0, 1, [30, 100])
    w2v_1 = np.matmul(Base, Cov_1)
    w2v_2 = np.matmul(Base, Cov_2)
    beta_01 = np.linspace(-15, 19, 4)
    beta_02 = np.linspace(-10, 10, 4)
    factor = np.array([[0] * 1000 + [1] * 1000]).T
    Y_1_r = [sum((i + beta_01) > 0) for i in (np.matmul(beta, w2v_1) * freq_1_r).sum(axis=1)]
    Y_2_r = [sum((i + beta_02) > 0) for i in (np.matmul(beta, w2v_2) * freq_2_r).sum(axis=1)]
    Y_1_e = [sum((i + beta_01) > 0) for i in (np.matmul(beta, w2v_1) * freq_1_e).sum(axis=1)]
    Y_2_e = [sum((i + beta_02) > 0) for i in (np.matmul(beta, w2v_2) * freq_2_e).sum(axis=1)]
    Y_r = np.array(Y_1_r + Y_2_r)
    Y_e = np.array(Y_1_e + Y_2_e)
    X_r = sparse.csr_matrix(np.concatenate([freq_1_r, freq_2_r]))
    X_e = sparse.csr_matrix(np.concatenate([freq_1_e, freq_2_e]))
    Data_r = [X_r,factor,Y_r]
    Data_e = [X_e,factor,Y_e]
    return(Data_r,Data_e,Base)
Data_train,Data_test,Base = Example()
model = AL.algor(Data_train,Base.T)
model.Initialization()
model.Training(1.)
Error_r = np.mean(np.abs(model.Predict(Data_train[0],Data_train[1]) - Data_train[2]))/4
Error_e = np.mean(np.abs(model.Predict(Data_test[0],Data_test[1]) - Data_test[2]))/4
print('The training Error is',Error_r,'The testing Error is',Error_e)