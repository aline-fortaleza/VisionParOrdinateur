import numpy as np
import tensorflow as tf
from tensorflow import layers

np.random.seed(0) # for reproducibility
tf.random.set_seed(0) # for reproducibility

X = np.array ([
    [0,0],
    [0,1],
    [1,0],
    [1,1]], dtype=np.float32
)
Y = np.array ([
    [0],    
    [0],    
    [0],    
    [1]], dtype=np.float32)

# first model, 1 neuron with sigmoid activation function
model = tf.keras.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=(2,)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, Y, epochs=1000, verbose=0)


