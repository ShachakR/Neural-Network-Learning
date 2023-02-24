import random
import numpy as np 
import csv 
from abc import ABC, abstractmethod
import math
from enum import Enum
import json

class Type(Enum):
    BINARY_CLASSIFICATION = 1
    REGRESSION = 2
    CATEGORICAL_CLASSIFICATION = 3

class Loss(ABC): 
    @abstractmethod
    def compute(self, x, t):
        """ Computes loss function
        x: prediction
        t: target
        """
        pass

    @abstractmethod
    def compute_derivative(self, x, t):
        """ Computes derivative loss function
        x: prediction
        t: target
        """
        pass

class SquareLoss(Loss): 
    
    def compute(self, x, t):
        try:
            r = 1/2*pow(x-t, 2) 
        except OverflowError: 
            r = 100

        return r
    
    def compute_derivative(self, x, t):
        return (x-t)

class HingeLoss(Loss):
    def compute(self, x, t):
        return max(0, 1 - (t * x ))
    
    def compute_derivative(self, x, t): 
        r = (t*x)
        if r <= 0:
            return (1/2) - r

        if r > 0 and r < 1:
            return (1/2) * pow((1-r),2)
        
        return 0

class LogisticLoss(Loss):
    def compute(self, x, t):
        return math.log(1 + pow(math.e, -t*x))
    
    def compute_derivative(self, x, t): 
        top = pow(math.e, -t*x)
        bottom = 1 + pow(math.e, -t*x)

        return -(top/bottom)

class BinaryCrossentropy(Loss):
    def compute(self, x, t):
        epsilon = 1e-7  # to avoid taking the log of 0
    
        # Clip y_pred to avoid log(0) and log(1)
        y_pred = np.clip(x, epsilon, 1 - epsilon)
        
        # Compute the binary cross-entropy loss
        loss = -(t * np.log(y_pred) + (1 - t) * np.log(1 - y_pred))
        
        # Take the mean of the loss across all samples
        loss = np.mean(loss)
        
        return loss
    
    def compute_derivative(self, x, t):
        epsilon = 1e-7  # to avoid division by zero
    
        # Clip y_pred to avoid division by zero
        y_pred = np.clip(x, epsilon, 1 - epsilon)
        
        # Compute the derivative of the binary cross-entropy loss with respect to y_pred
        derivative = -(t / y_pred) + ((1 - t) / (1 - y_pred))
        
        return derivative

class Activation(ABC): 
    @abstractmethod
    def compute(self, x):
        pass

    @abstractmethod
    def compute_derivative(self, x):
        pass

class ReLU(Activation): 
    def compute(self, x):
       return max(0, x)

    def compute_derivative(self, x):
        if x > 0:
            return 1.0
        else:
            return 0
    
    def __str__(self):
        return 'relu'

class Sigmoid(Activation): 
    def compute(self, x):
       r = 1.0 / (1.0 + pow(math.e, -x))
       return r

    def compute_derivative(self, x):
        a = self.compute(x)
        r = a*(1-a)
        return r
    
    def __str__(self):
        return 'sigmoid'

# Need to adjust to work for current implimentation
# class Softmax(Activation):
#     def compute(self, x):
#         """Compute the softmax of a list of numbers x."""
#         # Subtract the maximum value from all elements to avoid numerical instability.
#         x = [xi - max(x) for xi in x]

#         # Compute the exponent of each element.
#         exp_x = [math.exp(xi) for xi in x]

#         # Normalize the exponentiated values.
#         return [xi / sum(exp_x) for xi in exp_x]
    
#     def compute_derivative(self, x):
#         """Compute the derivative of the softmax function for a list of numbers x."""
#         # Compute the softmax of x.
#         y = self.compute(x)

#         # Initialize the derivative as an empty list of the same length as x.
#         d = [0] * len(x)

#         # For each element of y, compute its derivative with respect to x.
#         for i in range(len(y)):
#             for j in range(len(y)):
#                 if i == j:
#                     d[i] += y[i] * (1 - y[i])
#                 else:
#                     d[i] -= y[i] * y[j]

#         return d
    
#     def __str__(self):
#         return 'softmax'
    
    

class Layer(): 
    def __init__(self, number_of_nodes, activation_func: Activation):
        self.n = number_of_nodes
        self.activation_func = activation_func
    
    def activation(self, X): 
        if self.activation_func is None: 
            return X
        return self.activation_func.compute(X)
    
    def activation_prime(self, X): 
        if self.activation_func is None: 
            return 1
        return self.activation_func.compute_derivative(X)

class Model():
    def __init__(self, train_x, train_y, type: Type, layers: list, loss=SquareLoss()):

        # required
        self.train_x = train_x 
        self.train_y = train_y
        self.type = type
        self.layers = layers
        self.predictor = None

        #kwags
        self.loss = loss
    
    def validate(self): 
        if self.layers == None or len(self.layers) == 0: 
            print('Error: No Layers provided')
            return False

        if self.layers[0].n != len(self.train_x[0]):
            print('Error: First layer must have number of nodes equal to number of features')
            return False

        if self.layers[len(self.layers) - 1].n < 1:
            print('Error: Last layer must have at least one node')
            return False
        
        return True

    
    # for now has an edge between each node of 2 layers 
    def number_of_edges(self):
        edges = 0

        for t in range(0, len(self.layers) - 1): 
            edges += self.layers[t].n * self.layers[t + 1].n
        
        return edges
    
    # runs SGD 
    def train(self, iterations, learning_rate=0.01, accuracy_threshold=0.05, regularizer=0, metrics=False):
        if not self.validate():
            return
        
        print('Training started...')
        self.learning_rate = learning_rate
        self.accuracy_threshold = accuracy_threshold
        self.regularizer = regularizer
        
        new_w = self.initW()
        for i in range(0, iterations): 
            new_w = self.SGD(new_w, i)
            self.predictor = new_w

            if metrics:
                loss, accuracy = self.test(new_w) 
                print('Training Run {}: loss = {:.4f}, accuracy = {:.4f}'.format(i+1, loss, accuracy))
        
        loss, accuracy = self.test(self.predictor) 
        print('Last Training Run: loss = {:.4f}, accuracy = {:.4f}'.format(loss, accuracy))
        print('Finished!')
    
    def SGD(self, w, iteration): 
        step = self.learning_rate / (iteration + 1)
        j = np.random.randint(0, len(self.train_x))
        gradient = self.backprop([self.train_x[j], self.train_y[j]], w.copy(), self.layers)
        new_w = np.subtract(w, step*(np.add(np.array(gradient), self.regularizer*np.array(w))))
        return new_w
                   
    def test(self, w): 
        loss = 0
        accuracy = 0
        test_amount = round(len(self.train_x) * 0.1) + 1
        for i in range(0, test_amount): 
            X = self.train_x[i]
            target = self.train_y[i]
            prediction = self.predcit(X, w)
            computed_loss = self.loss.compute(prediction, target)
            loss += computed_loss
            if computed_loss < self.accuracy_threshold:
                accuracy += 1
        
        return loss / len(self.train_x), accuracy / test_amount
            
    def backprop(self, data_point, w, G):
        # page 278 in the textbook
        X = data_point[0]
        target = data_point[1]
        W = [] # where W[t] is the weights between layer t and t + 1

        # building weight matrix for each connection between layers l and l+1
        start_index = 0
        for l in range(0, len(G)-1):
            w_matrix, start_index = self._build_weight_matrix(G[l], G[l+1], w, start_index)
            W.append(w_matrix) 

        #forward
        O = [] # outputs for each layer
        A = [] # inputs for each layer
        O.append(X)
        A.append(X)
        for t in range(1, len(G)): # for every layer, besides the input layer (we already know the output and input of that layer)
            at = []
            ot = []
            for i in range(0, G[t].n): # for every node in this layer
                at_sum = 0
                for j in range(0, G[t-1].n): # for every node in the previous layer
                    at_sum += W[t-1][i][j] * O[t-1][j] # the input of the ith node in this layer is the weighted sum of the nodes in the previous layer
                at.append(at_sum)
                ot.append(G[t].activation(at_sum))
            A.append(at)
            O.append(ot)

        # backwards
        deltas = []

        # initalize a deltas, one for each node besides the input layer
        # NOTE: we start from t = 0 here only for being able to access the deltas based on their layer, but we never use the first layer
        # there are no deltas for the first layer!
        for t in range(0, len(G)):
            deltas.append([])
            for i in range(0, G[t].n):
                deltas[t].append(0)
        
        for i in range(0, G[len(G)-1].n):
            deltas[len(G) - 1][i] = self.loss.compute_derivative(O[t][i], target)

        for t in range(len(G)-2, 0, -1): # for every layer but the output layer and input layer
            for i in range(0, G[t].n): # for every node in this layer
                deltaT_sum = 0
                for j in range(0, G[t+1].n): # for every node in the next layer
                    deltaT_sum += W[t][j][i]*deltas[t+1][j]*G[t+1].activation_prime(A[t+1][j])
                deltas[t][i] = deltaT_sum
        
        gradient = []
        # for every edge set partial derivatives 
        for t in range(1, len(G)):
            for j in range(0, G[t].n):
                value = 0
                for i in range(0, G[t-1].n): # for every node in this layer
                    value = deltas[t][j] * G[t].activation_prime(A[t][j]) * O[t-1][i] 
                    gradient.append(value)
        
        zero_gradient =  not np.any(gradient)

        if zero_gradient: 
            return gradient
        else:
            normalize_grad = gradient / np.linalg.norm(gradient)
            return normalize_grad

    def predcit(self, X, w=None):
        if w is None:
            if self.predictor is None:
                print('No predictor, train the model first')
                return 
            w = self.predictor.copy()

        W = [] # where W[t] is the weights between layer t and t + 1
        G = self.layers

        # building weight matrix for each connection between layers l and l-1
        start_index = 0
        for l in range(0, len(G)-1):
            w_matrix, start_index = self._build_weight_matrix(G[l], G[l+1], w, start_index)
            W.append(w_matrix) 

        #forward
        O = [] # outputs for each layer
        A = [] # inputs for each layer
        O.append(X)
        A.append(X)
        for t in range(1, len(G)): # for every layer, besides the input layer (we already know the output and input of that layer)
            at = []
            ot = []
            for i in range(0, G[t].n): # for every node in this layer
                at_sum = 0
                for j in range(0, G[t-1].n): # for every node in the previous layer
                    at_sum += W[t-1][i][j] * O[t-1][j] # the input of the ith node in this layer is the weighted sum of the nodes in the previous layer
                at.append(at_sum)
                ot.append(G[t].activation(at_sum))
            A.append(at)
            O.append(ot)

        prediction = []

        # for every output neuron, minimum 1 
        for i in range(0, len(O[len(G) - 1])):
            prediction.append(O[len(G) - 1][i])

        if self.type == Type.BINARY_CLASSIFICATION:
                prediction = prediction[0] # only has one output neuron in bianry classification
                prediction = 1 if prediction > 0.5 else 0
        elif self.type == Type.CATEGORICAL_CLASSIFICATION:
            prediction = np.argmax(prediction) + 1
        else: # regression, only has one output neuron
            prediction =  prediction[0]
        
        return prediction
            

    def _build_weight_matrix(self, layer1: Layer, layer2: Layer, W: list, start_index): 
        w = np.zeros((layer2.n, layer1.n))
        index = start_index
        for i in range(0, len(w)):
            for j in range(0, len(w[0])): 
                w[i][j] = W[index]
                index += 1
        return np.ndarray.tolist(w), index-1
    

    def print_network(self):
        layer_i = 0
        for layer in self.layers:
            print('Layer {}: nodes = {}, activation = {}'.format(layer_i, layer.n, layer.activation_func))
            layer_i += 1
    
    def initW(self):
        W_vector = []
        for _ in range(0, self.number_of_edges()): 
            W_vector.append(np.random.uniform(-0.2, 0.2))

        return W_vector

def build_data(csv_data_file, add_bias=False): 
    # data is a csv file 
    data = []
    features = []
    labels = []
    with open(csv_data_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(row)

    random.shuffle(data)
    for d in data:
        try: 
            t = float(d.pop(len(d) - 1))
            x = [float(i) for i in d] 
            if add_bias is True:
                x.append(1.0) # for bias 

            features.append(x)
            labels.append(t)
        except: 
            continue

    return features,labels 

def split_train_test(features, labels, train_split=0.8): 
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    train_number = len(features) * train_split
    
    for i in range(0, round(train_number)): 
        train_x.append(features[i])
        train_y.append(labels[i])
    
    for i in range(int(train_number), len(features)): 
        test_x.append(features[i])
        test_y.append(labels[i])
    
    return train_x, train_y, test_x, test_y


