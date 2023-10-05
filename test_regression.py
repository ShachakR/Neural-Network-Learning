import neural_network as NN

# Load and preprocess your data
data_points, labels = NN.build_data("test_data_sets/regression/simple.csv")
train_x, train_y, test_x, test_y = NN.train_test_split(data_points, labels, train_split=0.2)

# Define your neural network architecture
layers = [
    NN.Layer(number_of_nodes=1, activation_func=None)
]

# Create a Model instance
model = NN.Model(train_x, train_y, type=NN.Type.REGRESSION, layers=layers, loss=NN.SquareLoss())
model.print_network()

# Train the model
model.train(epochs=7000, batch_size=7, metrics=True)

# Test the model
test_loss, test_accuracy = model.test(test_x, test_y)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print(model.predict([1234])) # expected result should be ~2468

# Save model
model.save_model("model.json")

# # Load and Predit with Model
model = NN.Model.load_model("model.json")
print(model.predict([1234]))