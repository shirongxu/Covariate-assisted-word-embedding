import numpy as np
class WeightSVM(object):
    def __init__(self,X,Y,S,lam,tol=10**-3):
        # X: input data: List or Matrix: Observation by features
        # Y: Response: list or array
        # S: Sample weight:list or array
        self.X = X
        self.Y = Y
        self.eps = tol
        self.SamWe = S
        self.N = np.shape(self.X)[0]
        self.a = np.zeros(len(Y))
        self.UpdOrder = range(self.N)
        self.W = np.zeros(self.X.shape[1])
        self.Lam = lam # Lambda for loss not norm
        self.Qii = []
        self.Loss = 10 ** 6
        self.LS = []
    def PG(self,index,G,a): # i g self.a[i]
        Output = 0.
        if a==0:
            Output = min(G,0)
        if a==self.SamWe[index]:
            Output = max(G,0)
        if 0<a<self.SamWe[index]:
            Output = G
        return(Output)
    def Initial(self):
        for i in range(self.N):
            Temp_ins = self.X[i,:] # check the input here
            Temp_Qii = sum(np.multiply(Temp_ins,Temp_ins))
            self.Qii.append(Temp_Qii)
    def Primal_loss(self):
        L = 1 - np.dot(self.X,self.W) * self.Y
        LL = np.array([max(i,0) for i in L]) * self.SamWe
        LLL = sum(LL) + 0.5 * self.Lam * np.dot(self.W,self.W)
        return(LLL)
    def Tol(self):
        Temp_loss = 0.5 * sum(self.W * self.W) * self.Lam+10 ** -10
        Temp_loss = Temp_loss-sum(self.a)
        Norm = np.abs(Temp_loss-self.Loss)
        Norm = (Norm+0.0)/(np.abs(self.Loss)+0.0)
        if Norm <= self.eps:
            self.Loss = Temp_loss
            self.LS.append(self.Loss)
            return(False)
        if Norm > self.eps:
            self.Loss = Temp_loss
            self.LS.append(self.Loss)
            return(True)
    def Ite(self):
        self.Initial()
        SL = self.SamWe
        YX = [self.Y[i] * self.X[i, :] for i in range(self.N)]
        while self.Tol()==True:
            Temp_Order = np.random.permutation(self.UpdOrder)
            #L = []
            for i in Temp_Order:
                #Temp_ins = np.array(self.X[i, :])# check [0]
                g = np.dot(self.W,YX[i])-1.
                PG = self.PG(i,g,self.a[i])
                Temp = self.a[i]
                if np.abs(PG)!=0:
                    self.a[i] = min(max(self.a[i]- self.Lam * g/self.Qii[i],0),SL[i])
                    self.W = self.W + (self.a[i]-Temp)* YX[i]/self.Lam
            #print(self.Loss)
            #self.LS.append(self.Loss)
        return(self.W)
    def Single(self):
        self.Initial()
        SL = self.SamWe
        YX = [self.Y[i] * self.X[i, :] for i in range(self.N)]
        Temp_Order = np.random.permutation(self.UpdOrder)
        L = []
        for i in Temp_Order:
            # Temp_ins = np.array(self.X[i, :])# check [0]
            g = np.dot(self.W, YX[i]) - 1.
            PG = self.PG(i, g, self.a[i])
            Temp = self.a[i]
            if np.abs(PG) != 0:
                self.a[i] = min(max(self.a[i] - self.Lam*g / self.Qii[i], 0), SL[i])
                self.W = self.W + (self.a[i] - Temp) * YX[i]/self.Lam
        return(self.W)
