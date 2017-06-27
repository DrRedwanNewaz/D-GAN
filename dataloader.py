import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class dataset(object):
    def __init__(self,seq_len):
        self.BATCH_SIZE=[]
        self.DIM=[]
        self.scaler=[]
        self.seq_len=seq_len


    def get_sequence(self,alphabet, seq_length, dim=2):
        dataX = []
        dataY = []
        for i in range(0, len(alphabet) - 2*seq_length, 5):
            seq_in = alphabet[i:i + seq_length]
            seq_out = alphabet[i + seq_length:i + seq_length+seq_length]
            dataX.append(seq_in)
            dataY.append(seq_out)
            # print(seq_in, '->', seq_out)
        # x = np.reshape(dataX, (len(dataX) * seq_length, 1, dim))
        return np.array(dataX), np.array(dataY)

    def get_data(self,train_id,name='train'):
        matlab_var = sio.loadmat('route.mat')
        path = matlab_var['route']
        training = path[0, 0][name]
        x_data = training[0, train_id]['data']
        print("%s id "%name,train_id)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(x_data)
        x_data = scaler.transform(x_data)
        self.scaler=scaler
        return np.array(x_data)


    def viz(self,x,y):
        x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
        y = np.reshape(y, (y.shape[0] * y.shape[1], y.shape[2]))

        print ("Ploting")
        plt.subplot(2, 1, 1)
        plt.scatter(x[:,1],x[:,0])
        plt.axis('square')
        plt.subplot(2, 1, 2)
        plt.scatter(y[:,1],y[:,0])
        plt.axis('square')
        plt.show()

    def get_dataset(self,num_train_set,name='train'):
        x_data = []
        y_data = []
        for i in range(num_train_set):
            x = self.get_data(i,name)
            x, y = self.get_sequence(x, self.seq_len)
            x_data.append(x)
            y_data.append(y)

        x_data = np.array(x_data)
        y_data = np.array(y_data)
        x = np.reshape(x_data, (x_data.shape[0],  x_data.shape[1]* x_data.shape[2], x_data.shape[3]))
        y = np.reshape(y_data, (y_data.shape[0],  y_data.shape[1]* y_data.shape[2], y_data.shape[3]))
        DIM = np.shape(x)
        self.BATCH_SIZE = int(DIM[1] / DIM[0])
        self.DIM=DIM
        print ("Dim:",DIM, "batch size:= ",self.BATCH_SIZE)
        return x,y

