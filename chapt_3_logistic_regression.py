import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F

plt.ioff()  # Turn off interactive mode

device = torch.device('cpu')


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



# HOW CAN I USE KEEPDIMS=TRUE 

class LogisticRegression1():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(1, num_features, dtype = torch.float32, device = device)  # row vector
        self.bias = torch.zeros(1, dtype = torch.float32, device = device)
    
    def forward(self, x):
        linear = torch.mm(x, self.weights.t()) + self.bias
        probas  = self._sigmoid(linear)
        return probas

    def backward(self, x, y, probas):
        grad_loss_wrt_z = probas.view(-1) - y
        grad_loss_wrt_w = torch.mm(x.t(), grad_loss_wrt_z.view(-1, 1)).t()
        grad_loss_wrt_b = torch.sum(grad_loss_wrt_z)
        return grad_loss_wrt_w, grad_loss_wrt_b
    
    def predict_labels(self, x):
        probas = self.forward(x)
        labels = torch.where(probas >= .5, 1, 0) # threshold function
        return labels    
            
    def evaluate(self, x, y):
        labels = self.predict_labels(x).float()
        accuracy = torch.sum(labels.view(-1) == y.float()).item() / y.size(0)
        return accuracy
    
    def _sigmoid(self, z):
        return 1. / (1. + torch.exp(-z))
    
    def _logit_cost(self, y, proba):
        tmp1 = torch.mm(-y.view(1, -1), torch.log(proba.view(-1, 1)))
        tmp2 = torch.mm((1 - y).view(1, -1), torch.log(1 - proba.view(-1, 1)))
        return tmp1 - tmp2
    
    def train(self, x, y, num_epochs, learning_rate=0.01):
        epoch_cost = []
        for e in range(num_epochs):   
            #### Compute outputs ####
            probas = self.forward(x)
            #### Compute gradients ####
            grad_w, grad_b = self.backward(x, y, probas)
            #### Update weights ####
            self.weights -= learning_rate * grad_w
            self.bias -= learning_rate * grad_b
            #### Logging ####
            cost = self._logit_cost(y, self.forward(x)) / x.size(0)
            cost_scalar = cost.item()  # Convert tensor to scalar
            print('Epoch: %03d' % (e+1), end="")
            print(' | Train ACC: %.3f' % self.evaluate(x, y), end="")
            print(' | Cost: %.3f' % cost_scalar)
            epoch_cost.append(cost_scalar)
        return epoch_cost
    

def trainlogisticregression(): 
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

    model1 = LogisticRegression1(num_features=2)
    epoch_cost = model1.train(X_train_tensor, y_train_tensor, num_epochs=30, learning_rate=0.1)

    print('\nModel parameters:')
    print('  Weights: %s' % model1.weights)
    print('  Bias: %s' % model1.bias)
    plt.plot(epoch_cost)
    plt.ylabel('Neg. Log Likelihood Loss')
    plt.xlabel('Epoch')
    plt.savefig('chap3_logistic_regression_1.png')
    plt.close()        
    print('done')
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

    test_acc = model1.evaluate(X_test_tensor, y_test_tensor)
    print('Test set accuracy: %.2f%%' % (test_acc*100))
    
    
def comp_accuracy(label_var, pred_probas):
    pred_labels = torch.where((pred_probas > 0.5), 1, 0).view(-1)
    acc = torch.sum(pred_labels == label_var.view(-1)).float() / label_var.size(0)
    return acc


class LogisticRegression2(torch.nn.Module):
    
    def __init__(self, num_features):
        super(LogisticRegression2, self).__init__()
        self.linear = torch.nn.Linear(num_features, 1)
        # initialize weights to zeros here,
        # since we used zero weights in the
        # manual approach
        
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()
        # Note: the trailing underscore
        # means "in-place operation" in the context
        # of PyTorch
        
    def forward(self, x):
        logits = self.linear(x)
        probas = torch.sigmoid(logits)
        return probas



def trainlogisticregression2():     
    model2 = LogisticRegression2(num_features=2).to(device)
    optimizer = torch.optim.SGD(model2.parameters(), lr=0.1)
    num_epochs = 30

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
    
    for epoch in range(num_epochs):
        
        #### Compute outputs ####
        out = model2(X_train_tensor)
        
        #### Compute gradients ####
        loss = F.binary_cross_entropy(out, y_train_tensor, reduction='sum')  # you can use 'mean' instead of sum for stable training
        optimizer.zero_grad()
        loss.backward()
        
        #### Update weights ####  
        optimizer.step()
        
        #### Logging ####      
        pred_probas = model2(X_train_tensor)
        acc = comp_accuracy(y_train_tensor, pred_probas)
        print('Epoch: %03d' % (epoch + 1), end="")
        print(' | Train ACC: %.3f' % acc, end="")
        print(' | Cost: %.3f' % F.binary_cross_entropy(pred_probas, y_train_tensor))


    
    print('\nModel parameters:')
    print('  Weights: %s' % model2.linear.weight)
    print('  Bias: %s' % model2.linear.bias)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

    pred_probas = model2(X_test_tensor)
    test_acc = comp_accuracy(y_test_tensor, pred_probas)

    print('Test set accuracy: %.2f%%' % (test_acc*100))





    
         
if __name__ == '__main__':
    trainlogisticregression()
    # trainlogisticregression2()
        
        
    

print('done')