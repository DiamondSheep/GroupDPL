import torch

class DL():
    '''
    min ||Y - DX||^2_F 
    '''
    def __init__(self, Data, DictSize=30, lambd=0.0001):
        self.DictSize = DictSize
        self.DataMat = Data
        # initialization
        #self.DictMat = self.normcol_equal(torch.rand(self.DataMat.shape[0], self.DictSize))
        self.DictMat = self.normcol_equal(self.DataMat[:, :self.DictSize])
        self.CoefMat = torch.zeros(self.DictSize, self.DataMat.shape[1])
        self.reconstruct = torch.zeros(self.DataMat.shape)
        self.lambd = lambd

        if torch.cuda.is_available():
            self.DataMat = self.DataMat.cuda()
            self.DictMat = self.DictMat.cuda()
            self.CoefMat = self.CoefMat.cuda()
            self.reconstruct = self.reconstruct.cuda()
        #print("Dictionary Initialized.")

    def Update(self, iterations = 20, tolerance = 0.0001, showFlag = True):
        for i in range(iterations):
            self.UpdateCoef()
            if showFlag:
                print("iteration: "+ str(i))
                print("reconstruct loss: {}".format(self.evaluate()))
            if self.evaluate() < tolerance:
                break
            self.UpdateDict()
        self.UpdateCoef()
        self.DictMat = self.DictMat.half()
        self.CoefMat = self.CoefMat.half()
        if showFlag:
            print("Dictionary computed.")


    def UpdateDict(self):
        self.normcol()
        temp1 = torch.matmul(self.DataMat, self.CoefMat.t())
        temp2 = torch.inverse(torch.matmul(self.CoefMat, self.CoefMat.t()))
        self.DictMat = torch.matmul(temp1, temp2)

    def UpdateCoef(self):
        I_Mat = torch.eye(self.DictSize)
        if torch.cuda.is_available():
            I_Mat = I_Mat.cuda()
        print(self.DictMat.size())
        print(I_Mat.size())
        temp1 = torch.matmul(self.DictMat.t(), self.DataMat)
        temp2 = torch.inverse(torch.matmul(self.DictMat.t(), self.DictMat) + self.lambd * I_Mat)
        self.CoefMat = torch.matmul(temp2, temp1)

    def normcol_equal(self, matin):
        eps = 2.0 ** -52
        tempMat = torch.sqrt(torch.sum(matin * matin, axis=0) + eps)
        matout = matin / tempMat.repeat(matin.shape[0], 1)
        assert matout.shape == matin.shape
        return matout
        
    def normcol(self):
        matout = torch.zeros(self.DictMat.shape)
        for i in range(self.DictSize):
            matout[:, i] = (self.DictMat[:, i] - self.DictMat[:, i].min()) / (self.DictMat[:, i].max() - self.DictMat[:, i].min())
        assert self.DictMat.shape == matout.shape
        self.DictMat = matout
        
    def evaluate(self):
        err = torch.norm(self.construct() - self.DataMat) 
        return err
    
    def construct(self):
        self.reconstruct = torch.matmul(self.DictMat, self.CoefMat)
        return self.reconstruct

if __name__ == "__main__":
    X = torch.rand(4, 4)
    W = torch.rand(4, 4)
    dl = DL(Data=W, DictSize=4)
    dl.Update(iterations = 30)
    print("W:\n", W)
    print("MOD reconstruct:\n", dl.construct())
