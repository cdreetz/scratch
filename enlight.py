import numpy as np
import torch as torch


# X = (hours studying, hours sleeping), y = score on test
x_all = np.array(([2, 9], [1, 5], [3, 6], [5, 10]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float) 

# scale units aka normalization
x_all = x_all/np.max(x_all, axis=0)
y = y/100

# split the data
X = np.split(x_all, [3])[0] #training data
x_predicted = np.split(x_all, [3])[1] #testing data





class neural_network(object):
    def __init__(self):
    #parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)




    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        return s * (1 - s)

    def tanh(self, s):
        return (exp(s) - exp(-s))/(exp(s) + exp(-s))

    def relu(self, s):
        return np.max(0, s)

    def leakyrelu(self, s):
        return np.max(0.01*s, s)





    def forward(self, X):
    #forward propogation through our network
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o

    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error*self.sigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

    def predict(self):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(x_predicted))
        print("Output: \n" + str(self.forward(x_predicted)))






nn = neural_network()
for i in range(1000):
    print("Input: \n" + str(X))
    print("Actual Output: \n" + str(y))
    print("Predicted Output: \n" + str(nn.forward(X)))
    print("Loss: \n" + str(np.mean(y - nn.forward(X))))
    print("\n")
    nn.train(X, y)



#o = nn.forward(X)

#print(f"Predicted Output: \n" + str(o))
#print(f"Actual Output: \n" + str(y))
