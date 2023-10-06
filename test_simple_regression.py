from neural_network import *

# Load and preprocess your data
data_points, labels = build_data("test_data_sets/regression/simple.csv")
train_x, train_y, test_x, test_y = train_test_split(data_points, labels, train_split=0.8)

# Define your neural network architecture
layers = [
    Layer(number_of_nodes=1, activation_func=None)
]

# Specify the type of task (e.g., regression)
task_type = Type.REGRESSION

# Choose a loss function (e.g., Square Loss)
loss_function = LossFactory.create_loss("SquareLoss")

# Create a Model instance
model = Model(train_x, train_y, type=task_type, layers=layers, loss=loss_function)
model.print_network()

# Train the model
model.train(epochs=10, batch_size=7, metrics=True)

# Test the model
test_loss, test_accuracy = model.test(model.predictor, test_x, test_y)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print(model.predict([1234])) # expected result should be ~2468

# Save model
model.save_model("model.json")

# # Load and Predit with Model
model = Model.load_model("model.json")
print(model.predict([1234]))