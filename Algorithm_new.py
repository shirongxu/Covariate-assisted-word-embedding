import numpy as np
import Weighted_SVM as WS
from cvxopt import matrix, solvers
import cvxopt
import scipy.sparse as sparse
def scipy_sparse_to_spmatrix(A):
    coo = A.tocoo()
    SP = cvxopt.spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return(SP)
def Uni_arr(X):
    # designed for getting different arrays
    R,C = X.shape
    Set = [X[0].tolist()]
    for i in np.arange(1,R):
        temp = sum([X[i].tolist()==j for j in Set])
        if temp == 0:
            Set.append(X[i].tolist())
    return(Set)
class algor(object):
    def __init__(self,Data,w2v):
        # Data[0]: number of observations * word frequency
        # Data[1]: covariates
        # Data[2]: label
        self.X = Data[0]
        self.factor = Data[1]
        self.FacNum = np.shape(np.mat(self.factor))[1]
        self.Y = Data[2]
        self.w2v = w2v.T # self.w2v is dimension of word embedding * number of words
        self.wordNum = np.shape(self.w2v)[1] # The number of sentiment words used
        self.LoBeta = np.shape(self.w2v)[0] # The length of Beta
        self.ite = 0.0001
        self.K = int(np.max(self.Y)) # 0,1,...,K
        self.BETA = np.zeros(self.LoBeta)
        self.W = [] # the set for W
        self.BETA_0 = [] # the set for various beta_0
        self.IndexBeta_0 = []
        self.XforInput = []
        self.RH = [] # used in step 1: B(t_ij) w_{x_{ij}}
        self.LH = [] # used in step 2: beta * D
        self.Vec_S = [] # The contant vector in step 3
        self.AindexBeta0 = np.unique(self.IndexBeta_0)
        self.IndexFY = []
        self.Dict_embed = {}
        self.Dict_beta0 = {}
        self.DB = np.matmul(self.w2v,self.X.T.toarray())
        self.SLL = []
        self.Err = []
        self.Err_1 = []
        self.Beta_set = [self.BETA]
    def Initialization(self):
        self.YforInput = []
        self.BETA = np.random.normal(0,0.1,self.LoBeta)
        Cov_set = Uni_arr(self.factor)
        for i in Cov_set:
            Temp = np.ones(self.wordNum)
            BTemp = 0-np.sort(np.random.uniform(-1, 1, int(self.K)))
            self.Dict_embed.update({str(i):Temp})
            self.Dict_beta0.update({str(i):BTemp})
        for i in range(len(self.Y)):
            for j in range(int(self.K)):
                self.YforInput.append(2 * ((self.Y[i]-j)>0)-1)
        self.a = np.zeros(len(self.YforInput))
    def Build_BW(self):
        # calculating B * W of beta * D (B * W)
        # D * B
        #DB = np.matmul(self.w2v,self.X.T.toarray())
        temp = []
        EmbedTemp = []
        for i in range(len(self.Y)):
            Embed_Temp = self.Dict_embed.get(str(self.factor[i].tolist()))
            EmbedTemp.append(Embed_Temp.tolist())
        EmbedTemp = np.array(EmbedTemp)
        BW = np.multiply(self.X.toarray(),EmbedTemp)
        X_out = np.matmul(BW,self.w2v.T)
        return(X_out)
    def Build_BWB(self):
        BWP = np.matmul(self.BETA,self.w2v)
        Out = self.X.toarray() * BWP
        return(Out)
    def Loss(self,Lam_1,Lam_2):
        NB = np.linalg.norm(self.BETA)**2 * Lam_1 *0.5
        NW = sum([np.linalg.norm(i)**2 for i in list(self.Dict_embed.values())]) * Lam_2*0.5
        N_size = np.shape(self.X)[0]
        Part = np.matmul(self.BETA, self.w2v)
        Result = []
        L = 0.
        for i in range(N_size):
            Beta_0 = self.Dict_beta0.get(str(self.factor[i]))
            W = self.Dict_embed.get(str(self.factor[i]))
            Part_2 = np.multiply(self.X[i].toarray()[0], W)
            Y_F = np.dot(Part, Part_2) + Beta_0
            Y_B = 2*(self.Y[i] - np.array([i for i in range(int(self.K))])>0)-1
            Re = 1 - Y_F * Y_B
            Re1 = sum([np.max([0,i]) for i in Re])
            L += Re1
        return(L+NB+NW)
    def Predict(self,X,cov):
        N_size = np.shape(X)[0]
        Part = np.matmul(self.BETA,self.w2v)
        Result = []
        for i in range(N_size):
            Beta_0 = self.Dict_beta0.get(str(cov[i]))
            W = self.Dict_embed.get(str(cov[i]))
            Part_2 = np.multiply(X[i].toarray()[0],W)
            Y_F = np.sign(np.dot(Part,Part_2)+Beta_0)
            Result.append(sum(Y_F==1))
        return(np.array(Result))
    def Upd_Beta(self,Lam_1):
        sample_weight = []
        Data_input = []
        TempMat = self.Build_BW()
        for i in range(len(self.Y)):
            B0temp = self.Dict_beta0.get(str(self.factor[i].tolist()))
            for j in range(int(self.K)):
                temp = 1 - self.YforInput[i * int(self.K) + j] * B0temp[j]
                sample_weight.append(temp)
                temp_Data_input = (TempMat[i] / temp).tolist()
                Data_input.append(temp_Data_input)
        sample_weight = np.array(sample_weight)
        Data_input = np.array(Data_input)
        #model = WS.WeightSVM()
        #self.BETA = model.Ite()
        #model = VS.weightsvm(C=1./Lam_1,max_iter = 5000,print_step=0)
        #model.fit(Data_input, np.array(self.YforInput), np.array(sample_weight))
        model = WS.WeightSVM(Data_input, self.YforInput, np.array(sample_weight), Lam_1)
        self.BETA = model.Ite()
        self.Beta_set.append(self.BETA)
        #self.BETA = model.beta
    def Upd_W(self,Lam_2):
        Data_all = self.Build_BWB()
        ALL_fac = list(self.Dict_embed.keys())
        for k in ALL_fac:
            Temp_data = Data_all[(self.factor==eval(k)).T[0],:]
            Temp_Y = self.Y[(self.factor==eval(k)).T[0]]
            sample_weight = []
            Data_input = []
            Y_for_input = []
            B0fix = self.Dict_beta0.get(k)
            for i in range(len(Temp_Y)):
                B0temp = self.Dict_beta0.get(str(self.factor[i].tolist()))
                for j in range(int(self.K)):
                    YY = (2*(Temp_Y[i]-j>0)-1)
                    temp = 1 - YY * B0fix[j]
                    sample_weight.append(temp)
                    temp_Data_input = (Temp_data[i] / temp).tolist()
                    Data_input.append(temp_Data_input)
                    Y_for_input.append(YY)
            Data_input = np.array(Data_input)
            model = WS.WeightSVM(Data_input, Y_for_input, np.array(sample_weight), Lam_2)
            W = model.Ite()
            #model = VS.weightsvm(C=1. / Lam_2,max_iter = 10000,print_step=0)
            #model.fit(Data_input, np.array(Y_for_input), np.array(sample_weight))
            #W = model.beta
            self.Dict_embed.update({k:W})
    def Upd_Beta0(self):
        Part = np.matmul(self.BETA, self.w2v)
        Xfor3 = []
        for i in range(len(self.Y)):
            Beta_0 = self.Dict_beta0.get(str(self.factor[i]))
            W = self.Dict_embed.get(str(self.factor[i]))
            Part_2 = np.multiply(self.X[i].toarray()[0], W)
            Com = np.dot(Part, Part_2)
            Xfor3.append(Com)
        ALL_beta0 = list(self.Dict_beta0.keys())
        for B0 in ALL_beta0:
            X = np.array(Xfor3)[self.factor.T[0] == eval(B0)]
            Y = self.Y[self.factor.T[0] == eval(B0)]
            Num_obs = int(len(Y) * self.K)
            self.Vec_S = np.zeros(self.K).tolist() + np.ones(Num_obs).tolist()
            A_1 = sparse.lil_matrix((Num_obs, len(self.Vec_S)))
            Output_1 = []
            for i in range(len(Y)):
                Temp_Y = 2 * (self.Y[i] - np.array([j for j in range(self.K)]) > 0) - 1
                for k in range(self.K):
                    A_1[i * self.K + k,k] = -(Temp_Y[k] + 0.0)
                    A_1[i * self.K + k, self.K + i * self.K + k] = -1.
                    Output_1.append(Temp_Y[k] * X[i] - 1)
            A_2 = sparse.lil_matrix((Num_obs, len(self.Vec_S)))
            Output_2 = []
            for i in range(len(Y)):
                for k in range(self.K):
                    A_2[i * self.K + k,self.K + i * self.K + k] = -1.
                    Output_2.append(0.)

            A_3 = sparse.lil_matrix((int(self.K) - 1, len(self.Vec_S)))
            for m in range(int(self.K) - 1):
                A_3[m, m] = -1.
                A_3[m, m + 1] = 1.
            Output_3 = (np.zeros(int(self.K) - 1)).tolist()

            A_4 = sparse.lil_matrix((self.K, len(self.Vec_S)))
            A_5 = sparse.lil_matrix((self.K, len(self.Vec_S)))
            for n in range(int(self.K)):
                A_4[n, n] = 1.
                A_5[n, n] = -1.
            Output_45 = (np.ones(2 * int(self.K)) - 0.02).tolist()
            A = sparse.vstack([A_1, A_2, A_3, A_4, A_5])
            OutPut = Output_1 + Output_2 + Output_3 +Output_45
            solvers.options['show_progress'] = False
            sol = solvers.lp(matrix(self.Vec_S), scipy_sparse_to_spmatrix(A), matrix(OutPut))
            value = np.array(sol['x']).T[0][0:int(self.K)]
            self.Dict_beta0.update({B0: value})

            #for i in range(len(X)):
                #Temp_Y = 2 * (self.Y[i] - np.array([j for j in range(self.K)]) > 0) - 1
                #for j in range(self.K):
                    #AA1 = 1 - Temp_Y[j] * (X[i] + 0.98)
                    #AA2 = np.array(sol['x']).T[0][int(self.K)*(i+1)+j]
                    #print(max(AA1,0),AA2)
    def Uni_Beta0(self):
        Part = np.matmul(self.BETA, self.w2v)
        Xfor3 = []
        for i in range(len(self.Y)):
            Beta_0 = self.Dict_beta0.get(str(self.factor[i]))
            W = self.Dict_embed.get(str(self.factor[i]))
            Part_2 = np.multiply(self.X[i].toarray()[0], W)
            Com = np.dot(Part, Part_2)
            Xfor3.append(Com)

        X = np.array(Xfor3)
        Y = self.Y
        Num_obs = int(len(Y) * self.K)
        self.Vec_S = np.zeros(self.K).tolist() + np.ones(Num_obs).tolist()
        A_1 = sparse.lil_matrix((Num_obs, len(self.Vec_S)))
        Output_1 = []
        for i in range(len(Y)):
            Temp_Y = 2 * (self.Y[i] - np.array([j for j in range(self.K)]) > 0) - 1
            for k in range(self.K):
                A_1[i * self.K + k, k] = -(Temp_Y[k] + 0.0)
                A_1[i * self.K + k, self.K + i * self.K + k] = -1.
                Output_1.append(Temp_Y[k] * X[i] - 1)
        A_2 = sparse.lil_matrix((Num_obs, len(self.Vec_S)))
        Output_2 = []
        for i in range(len(Y)):
            for k in range(self.K):
                A_2[i * self.K + k, self.K + i * self.K + k] = -1.
                Output_2.append(0.)

        A_3 = sparse.lil_matrix((int(self.K) - 1, len(self.Vec_S)))
        for m in range(int(self.K) - 1):
            A_3[m, m] = -1.
            A_3[m, m + 1] = 1.
        Output_3 = (np.zeros(int(self.K) - 1)).tolist()

        A_4 = sparse.lil_matrix((self.K, len(self.Vec_S)))
        A_5 = sparse.lil_matrix((self.K, len(self.Vec_S)))
        for n in range(int(self.K)):
            A_4[n, n] = 1.
            A_5[n, n] = -1.
        Output_45 = (np.ones(2 * int(self.K)) - 0.02).tolist()
        A = sparse.vstack([A_1, A_2, A_3, A_4, A_5])
        OutPut = Output_1 + Output_2 + Output_3 + Output_45
        solvers.options['show_progress'] = False
        sol = solvers.lp(matrix(self.Vec_S), scipy_sparse_to_spmatrix(A), matrix(OutPut))
        value = np.array(sol['x']).T[0][0:int(self.K)]
        ALL_beta0 = list(self.Dict_beta0.keys())
        for B0 in ALL_beta0:
            self.Dict_beta0.update({B0: value})
    def Stop_cri(self):
        Stop_Cond = np.linalg.norm(self.Beta_set[-1]-self.Beta_set[-2])/np.linalg.norm(self.Beta_set[-1])
        return(Stop_Cond<0.1)

    def Training(self,lam):
        Stop = False
        i = 0
        while (Stop==False)*(i<=10)==1:
            self.Upd_Beta(np.sqrt(lam))
            self.Upd_W(np.sqrt(lam))
            self.Upd_Beta0()
            Stop = self.Stop_cri()
            i = i + 1
            print(i)



