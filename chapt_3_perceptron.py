import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F

plt.ioff()  # Turn off interactive mode

def vectorizationinpython():
    # understanding how python vectorization works 
    x0, x1, x2 = 1, 2, 3 
    bias , w1, w2  = 0.1 , 0.3, 0.5 
    x = [ x0 , x1, x2 ]    # create a list here 
    w = [ bias, w1, w2]

    # FIRST  a for loop 
    def forloop (x, w):
        z = 0 
        for i in range(len(w)):
            z += x[i] * w[i] 
        print('for loop', z)

    # SECOND lets use a  list comprehnsiion  ( mostly used by developers) 
    def list_comprehension(x, w):    
        z = sum(x_i * w_i for x_i , w_i in zip(x, w))
        # can be used sum[x_i * w_i for x_i , w_i in zip(x, w)]  # using [] 
        # print('list comprehnsion', z) 

    # THIRD do it by numpy here 
    import numpy as np 
    x_vec, w_vec = np.array(x), np.array(w)
    # convert it into numpy from list  ( x , w are in list )
    def numpy_format(x, w):
        #two ways of doin dot product 
        z = x_vec.transpose().dot(w_vec)
        # print('first method for dot prodcut', z)
        z = np.dot (x_vec, w_vec)
        # print('second method for dot prodcut', z)
        z = x_vec.dot(w_vec)
        # print('third method for dot prodcut', z)

    x, w  = np.random.rand(int(1e+5)), np.random.rand(int(1e+5))
    # now lets time it  ( numpy (fastest) > list comprehension > forloop)
    from timeit import timeit as timeit
    execution_time_list_comprehensio   = timeit(lambda: list_comprehension(x, w), number= 100)
    print('execution_time_list_comprehensio', execution_time_list_comprehensio)
    execution_time_numpy   = timeit(lambda: numpy_format(x, w), number= 100)
    print('execution_time_numpy', execution_time_numpy)
    # conclusion is : avoid for loops since they are slowlest. itc cause they are sequentially 
    # calculated. dot product in numpy or list cmprehension is done in parallel. 


class Perceptron():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = np.zeros((num_features, 1), dtype = np.float32)
        self.bias = np.zeros(1, dtype = np.float32)
        
    def forward(self, x):
        linear = np.dot(x, self.weights) + self.bias
        predictions  = np.where (linear > 0., 1, 0)
        return predictions
    
    def backward(self, x, y):
        predictions = self.forward(x)
        errors  = y  - predictions
        return errors
    
    def train (self, x, y, epochs):
        for e in range(epochs):  # iterate 
            for i in range (y.shape[0]): # iteate over all examples 
                # errors = self.backward(x[i].reshape(1, self.num_features), y[i]).reshape(-1)
                errors = self.backward(x[i], y[i])
                self.weights += (errors  * x[i]).reshape(self.num_features,1)
                self.bias += errors
    
    def evaluate (self, x, y):
        predictions = self.forward(x).reshape(-1)
        accuracy  = np.sum(predictions == y) / y.shape[0]
        return accuracy
    

device = torch.device('cpu')

# HOW CAN I USE KEEPDIMS=TRUE 

def trainperceptron():
    ppn = Perceptron(num_features=2)

    ppn.train(X_train, y_train, epochs=5)

    print('Model parameters:\n\n')
    print('  Weights: %s\n' % ppn.weights)
    print('  Bias: %s\n' % ppn.bias)

    train_acc = ppn.evaluate(X_train, y_train)
    print('Train set accuracy: %.2f%%' % (train_acc*100))
    
    test_acc = ppn.evaluate(X_test, y_test)
    print('Test set accuracy: %.2f%%' % (test_acc*100))
        
        ##########################
    ### 2D Decision Boundary
    ##########################

    w, b = ppn.weights, ppn.bias

    x0_min = -2
    x1_min = ( (-(w[0] * x0_min) - b[0]) 
            / w[1] )

    x0_max = 2
    x1_max = ( (-(w[0] * x0_max) - b[0]) 
            / w[1] )

    # x0*w0 + x1*w1 + b = 0
    # x1  = (-x0*w0 - b) / w1


    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(7, 3))

    ax[0].plot([x0_min, x0_max], [x1_min, x1_max])
    ax[0].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='class 0', marker='o')
    ax[0].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='class 1', marker='s')

    ax[1].plot([x0_min, x0_max], [x1_min, x1_max])
    ax[1].scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], label='class 0', marker='o')
    ax[1].scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], label='class 1', marker='s')

    ax[1].legend(loc='upper left')
    plt.savefig('chap3_perceptron.png')
    plt.close()
        
    print('done')
    

data  = np.genfromtxt('perceptron_toydata.txt')
X, y = data[:,0:2], data[:,2]
# shuffle it now 
shuffle_idx = np.arange(y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, y  = X[shuffle_idx] , y[shuffle_idx]

X_train, X_test = X[shuffle_idx[:70]], X[shuffle_idx[70:]]
y_train, y_test = y[shuffle_idx[:70]], y[shuffle_idx[70:]]

# Normalize (mean zero, unit variance)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma


         

if __name__ == '__main__':
    vectorization  = False
    train_percetron = True
    if vectorization:
        vectorizationinpython()
    if train_percetron:
        trainperceptron()

    
        
    

print('done')