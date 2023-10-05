import random
import numpy as np
import csv
import json
from abc import ABC, abstractmethod
import math
from enum import Enum

class Type(Enum):
    BINARY_CLASSIFICATION = 1
    REGRESSION = 2
    CATEGORICAL_CLASSIFICATION = 3


class ActivationFactory:
    @staticmethod
    def create_activation(name):
        if name == "ReLU":
            return ReLU()
        elif name == "Sigmoid":
            return Sigmoid()
        # Add more activation functions as needed.

class LossFactory:
    @staticmethod
    def create_loss(name):
        if name == "SquareLoss":
            return SquareLoss()
        elif name == "HingeLoss":
            return HingeLoss()
        elif name == "LogisticLoss":
            return LogisticLoss()
        elif name == "BinaryCrossentropy":
            return BinaryCrossentropy()
        # Add more loss functions as needed.

class Loss(ABC):
    @abstractmethod
    def compute(self, x, t):
        pass

    @abstractmethod
    def compute_derivative(self, x, t):
        pass

class SquareLoss(Loss):
    def compute(self, x, t):
        return 1/2 * (x - t) ** 2

    def compute_derivative(self, x, t):
        return x - t

class HingeLoss(Loss):
    def compute(self, x, t):
        return max(0, 1 - (t * x))

    def compute_derivative(self, x, t):
        if t * x <= 1:
            return -t
        else:
            return 0

class LogisticLoss(Loss):
    def compute(self, x, t):
        return math.log(1 + math.exp(-t * x))

    def compute_derivative(self, x, t):
        return -t / (1 + math.exp(t * x))

class BinaryCrossentropy(Loss):
    def compute(self, x, t):
        epsilon = 1e-7
        y_pred = np.clip(x, epsilon, 1 - epsilon)
        loss = -(t * np.log(y_pred) + (1 - t) * np.log(1 - y_pred))
        return np.mean(loss)

    def compute_derivative(self, x, t):
        epsilon = 1e-7
        y_pred = np.clip(x, epsilon, 1 - epsilon)
        return (y_pred - t) / (y_pred * (1 - y_pred))

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
        return 1 if x > 0 else 0

class Sigmoid(Activation):
    def compute(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def compute_derivative(self, x):
        a = self.compute(x)
        return a * (1 - a)

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
        self.train_x = train_x
        self.train_y = train_y
        self.type = type
        self.layers = layers
        self.predictor = None
        self.loss = loss
        self.layers.insert(0, Layer(len(self.train_x[0]), None)) # Create input layer

    def validate(self):
        if not self.layers or len(self.layers) == 0:
            print('Error: No Layers provided')
            return False

        if self.layers[0].n != len(self.train_x[0]):
            print('Error: First layer must have the number of nodes equal to the number of features')
            return False

        if self.layers[-1].n < 1:
            print('Error: Last layer must have at least one node')
            return False

        return True

    def number_of_edges(self):
        edges = 0
        for t in range(len(self.layers) - 1):
            edges += self.layers[t].n * self.layers[t + 1].n
        return edges

    def train(self, epochs, learning_rate=0.01, accuracy_threshold=0.05, regularizer=0, batch_size=32, metrics=False):
        if not self.validate():
            return

        print('Training started...')
        self.learning_rate = learning_rate
        self.accuracy_threshold = accuracy_threshold
        self.regularizer = regularizer
        w = self.initW()
        best_weights = w  # Store the best weights
        min_loss = 999999

        for epoch in range(epochs):
            # Shuffle the training data at the beginning of each epoch
            combined_data = list(zip(self.train_x, self.train_y))
            random.shuffle(combined_data)
            self.train_x, self.train_y = zip(*combined_data)

            num_batches = len(self.train_x) // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx + 1) * batch_size

                batch_x = self.train_x[start_idx:end_idx]
                batch_y = self.train_y[start_idx:end_idx]

                w = self.SGD(w, batch_x, batch_y)

            loss, accuracy = self._metrics(w)
            
            if loss < min_loss:  # Check if this epoch has the lowest loss
                min_loss = loss
                best_weights = w.copy()  # Save the best weights

            if metrics:
                print('Epoch {}: loss = {:.4f}, accuracy = {:.2f}%'.format(epoch + 1, loss, 100 * accuracy))

        print('Finished!')
        loss, accuracy = self._metrics(best_weights)  # Use the best weights for final evaluation
        self.predictor = best_weights  # Update the predictor with the best weights
        print('Final predictor results: loss = {:.4f}, accuracy = {:.2f}%'.format(loss, 100 * accuracy))
    

    def _metrics(self, w):
        loss = 0
        accuracy = 0
        test_amount = round(len(self.train_x) * 0.1) + 1
        for i in range(test_amount):
            X = self.train_x[i]
            target = self.train_y[i]
            prediction = self.predict(X, w)
            computed_loss = self.loss.compute(prediction, target)
            loss += computed_loss
            if self.type == Type.REGRESSION and computed_loss < self.accuracy_threshold:
                accuracy += 1
            else:
                if prediction == target:
                    accuracy += 1

        return loss / len(self.train_x), accuracy / test_amount
    
    def test(self, x_data, y_data):
        loss = 0
        accuracy = 0
        test_amount = round(len(x_data))
        for i in range(test_amount):
            X = x_data[i]
            target = y_data[i]
            prediction = self.predict(X, self.predictor)
            computed_loss = self.loss.compute(prediction, target)
            loss += computed_loss
            if self.type == Type.REGRESSION and computed_loss < self.accuracy_threshold:
                accuracy += 1
            else:
                if prediction == target:
                    accuracy += 1

        return loss / len(x_data), accuracy / test_amount

    def SGD(self, w, batch_x, batch_y):
        step = self.learning_rate
        j = np.random.randint(0, len(batch_x))
        gradient = self.backpropagation([batch_x[j], batch_y[j]], w.copy(), self.layers)
        new_w = np.subtract(w, step * (np.add(np.array(gradient), self.regularizer * np.array(w))))
        return new_w

    def backpropagation(self, data_point, w, G):
        # page 278 in the textbook:
        # Understanding Machine Learning: From Theory to Algorithms by Shai Shalev-Shwartz and Shai Ben-David (2014)
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
        
        t = len(G) - 1
        for i in range(0, G[t].n):
            deltas[t][i] = self.loss.compute_derivative(O[t][i], target)

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

    def predict(self, X, w=None):
        if w is None:
            if self.predictor is None:
                print('No predictor, train the model first')
                return
            w = self.predictor.copy()

        W = []
        G = self.layers

        start_index = 0
        for l in range(len(G) - 1):
            w_matrix, start_index = self._build_weight_matrix(G[l], G[l + 1], w, start_index)
            W.append(w_matrix)

        O = []
        A = []
        O.append(X)
        A.append(X)
        for t in range(1, len(G)):
            at = []
            ot = []
            for i in range(G[t].n):
                at_sum = 0
                for j in range(G[t - 1].n):
                    at_sum += W[t - 1][i][j] * O[t - 1][j]
                at.append(at_sum)
                ot.append(G[t].activation(at_sum))
            A.append(at)
            O.append(ot)

        prediction = []

        for i in range(len(O[-1])):
            prediction.append(O[-1][i])

        if self.type == Type.BINARY_CLASSIFICATION:
            prediction = prediction[0]
            prediction = 1 if prediction > 0.5 else 0
        elif self.type == Type.CATEGORICAL_CLASSIFICATION:
            prediction = np.argmax(prediction) + 1
        else:
            prediction = prediction[0]

        return prediction

    def _build_weight_matrix(self, layer1: Layer, layer2: Layer, W: list, start_index):
        w = np.zeros((layer2.n, layer1.n))
        index = start_index
        for i in range(len(w)):
            for j in range(len(w[0])):
                w[i][j] = W[index]
                index += 1
        return np.ndarray.tolist(w), index - 1

    def print_network(self):
        for layer_i, layer in enumerate(self.layers):
            print('Layer {}: nodes = {}, activation = {}'.format(layer_i, layer.n, layer.activation_func))

    def initW(self):
        W_vector = []
        for _ in range(self.number_of_edges()):
            W_vector.append(np.random.uniform())
        return W_vector
    
    def save_model(self, file_path):
        model_data = {
            "type": self.type.value,
            "layers": [
                {
                    "number_of_nodes": layer.n,
                    "activation_func": layer.activation_func.__class__.__name__
                }
                for layer in self.layers[1:]  # Skip input layer
            ],
            "loss": self.loss.__class__.__name__,
            "predictor": self.predictor.tolist() if self.predictor is not None else None,
            "train_x": self.train_x,
            "train_y": self.train_y
        }

        with open(file_path, 'w') as model_file:
            json.dump(model_data, model_file)
            
    @staticmethod
    def load_model(file_path):
        with open(file_path, 'r') as model_file:
            model_data = json.load(model_file)

        model_type = Type(model_data["type"])
        layers = [
            Layer(layer["number_of_nodes"], ActivationFactory.create_activation(layer["activation_func"]))
            for layer in model_data["layers"]
        ]
        loss = LossFactory.create_loss(model_data["loss"])
        train_x = model_data["train_x"]
        train_y = model_data["train_y"]

        model = Model(train_x, train_y, model_type, layers, loss)
        model.predictor = np.array(model_data["predictor"]) if model_data["predictor"] is not None else None

        return model


def build_data(csv_data_file, add_bias=False):
    data = []
    data_points = []
    labels = []

    with open(csv_data_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(row)

    random.shuffle(data)
    for d in data:
        try:
            t = float(d.pop())
            x = [float(i) for i in d]
            if add_bias:
                x.append(1.0)
            data_points.append(x)
            labels.append(t)
        except ValueError:
            continue

    return data_points, labels

def split_train_test(features, labels, train_split=0.8):
    train_x = features[:int(len(features) * train_split)]
    train_y = labels[:int(len(labels) * train_split)]
    test_x = features[int(len(features) * train_split):]
    test_y = labels[int(len(labels) * train_split):]
    return train_x, train_y, test_x, test_y
