# Neural-Network-Learning
 
# Purpose
My attempt to build a neural network framework from scratch, this mainly for my own learning and to help me improve my understanding of neural networks and meachine learning. 

# Usage 
Refer to any of the test files (i.e test_simple_regression.py) for examples on usage

## Prerequisites

Before using the script, make sure you have the following Python libraries installed:

- `numpy`
- `csv`
- `json`
- `math`

You can install these dependencies using `pip`

## Data Formating
```python
from neural_network import *

data_points, labels = build_data("path/to/csv/dataset.csv")
train_x, train_y, test_x, test_y = train_test_split(data_points, labels, train_split=0.8)
```

## Creating the Model
```python
# Define the layers with activation functions
# Input layer is automatically created.
# Output layer needs to be defined as the last layer in the list
layers = [
    Layer(number_of_nodes=10, activation_func=ActivationFactory.create_activation("ReLU")),
    Layer(number_of_nodes=5, activation_func=ActivationFactory.create_activation("ReLU")),
    Layer(number_of_nodes=1, activation_func=ActivationFactory.create_activation("Sigmoid"))
]

# Specify the type of task (e.g., regression)
task_type = Type.REGRESSION

# Choose a loss function (e.g., Square Loss)
loss_function = LossFactory.create_loss("SquareLoss")

# Create the model
model = Model(train_x, train_y, type=task_type, layers=layers, loss=loss_function)
```

## Train the Model
```python
model.train(epochs=100, metrics=True)
```

## Testing the Model
```python
test_loss, test_accuracy = model.test(model.predictor, test_x, test_y)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
```

## Saving and Loading the Model
```python
# Save model
model.save_model("model.json")

# Load and Predit with Model
model = Model.load_model("model.json")
```

# Resources Used
 * Understanding Machine Learning: From Theory to Algorithms by Shai Shalev-Shwartz and Shai Ben-David (2014)
 * Pattern Recognition and Machine Learning by Christopher M. Bishop
