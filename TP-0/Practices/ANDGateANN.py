import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


np.random.seed(42)
tf.random.set_seed(42)


def build_and_model():
    model = Sequential([
        Dense(1, activation='sigmoid', input_shape=(2,))
    ]) #only one layer with one neuron using sigmoid activation function

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='binary_crossentropy',
        metrics=['accuracy']
    ) # using binary crossentropy as the loss function since it's a binary classification problem, and Adam optimizer with a learning rate of 0.1 for faster convergence
    return model


def build_xor_model():
    model = Sequential([
        Dense(4, activation='sigmoid', input_shape=(2,)),  # hidden layer
        Dense(1, activation='sigmoid')                     # output layer
    ]) # using a hidden layer with 4 neurons to allow the model to learn the non-linear decision boundary of the XOR problem, and an output layer with 1 neuron for binary classification

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='binary_crossentropy',
        metrics=['accuracy']
    ) # same decision for the loss function and optimizer as the AND model
    return model


def train_and_test_gate(model, X, y, gate_name, epochs=100, verbose=0):
    print(f"Training model for {gate_name} gate")

    history = model.fit(X, y, epochs=epochs, verbose=verbose) # training the model on the input data X and labels y

    # evaluating
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"{gate_name} - Final Loss: {loss:.6f}")
    print(f"{gate_name} - Final Accuracy: {acc:.6f}")

    # predicting
    preds = model.predict(X, verbose=0)
    binary_preds = (preds >= 0.5).astype(int) # converting predicted probabilities to binary class labels using a threshold of 0.5

    print(f"\n{gate_name} Predictions:")
    for i in range(len(X)):
        print(
            f"Input: {X[i].tolist()} | "
            f"Predicted (prob): {preds[i][0]:.6f} | "
            f"Predicted (class): {binary_preds[i][0]} | "
            f"Ground truth: {int(y[i][0])}"
        )

    # print weights and biases
    print(f"\n{gate_name} model parameters:")
    for layer_idx, layer in enumerate(model.layers):
        weights, bias = layer.get_weights()
        print(f"Layer {layer_idx + 1} weights:\n{weights}")
        print(f"Layer {layer_idx + 1} bias:\n{bias}")

    return history


def main():
    # input data 
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float32)

    # AND labels
    y_and = np.array([
        [0],
        [0],
        [0],
        [1]
    ], dtype=np.float32)

    # XOR labels
    y_xor = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=np.float32)

    # AND gate
    and_model = build_and_model()
    train_and_test_gate(and_model, X, y_and, gate_name="AND", epochs=100, verbose=0)

    # XOR gate
    xor_model = build_xor_model()
    train_and_test_gate(xor_model, X, y_xor, gate_name="XOR", epochs=100, verbose=0)


if __name__ == "__main__":
    main()