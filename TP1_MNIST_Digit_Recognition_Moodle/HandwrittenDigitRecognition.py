import tensorflow as tf
from tensorflow.keras import layers   # I had to change this os it would work with the model used in the baseline_model 



def baseline_model(num_pixels, num_classes):

    #TODO - Application 1 - Step 6a - Initialize the sequential model
    model = tf.keras.models.Sequential() # creates a empty model network

    #TODO - Application 1 - Step 6b - Define a hidden dense layer with 8 neurons -> dense layer means that every neuron from the previous layer is connected to every neuron of the current layer
    model.add(layers.Dense(8, input_dim=num_pixels, kernel_initializer='normal', activation='relu')) # uses relu -> rectified linear unit as activation function so it can deal with non linearity

    #TODO - Application 1 - Step 6c - Define the output dense layer
    model.add(layers.Dense(num_classes, kernel_initializer='normal', activation='softmax')) # softmax converte raw scores into probabilities, with all the outputs summing to 1 and the highest probability being the predicted class

    # TODO - Application 1 - Step 6d - Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # using crossentopy because it's a multiclass classficiation, i'm using softmax in the output layer and i used one hot encoding for the labels 

    return model


def trainAndPredictMLP(x_train, y_train, x_test, y_test):

    #TODO - Application 1 - Step 3 - Reshape the MNIST dataset - Transform the images to 1D vectors of floats (28x28 pixels  to  784 elements)
    num_pixels = x_train.shape[1] * x_train.shape[2]  # 28*28=784
    x_train = x_train.reshape((x_train.shape[0], num_pixels)).astype('float32') # shape become (num_samples, num_pixels)
    x_test = x_test.reshape((x_test.shape[0], num_pixels)).astype('float32')   

    #TODO - Application 1 - Step 4 - Normalize the input values
    x_train = x_train / 255.0 # divided by 255 because of the gray scale, normalized to interval [0,1]
    x_test = x_test / 255.0

    #TODO - Application 1 - Step 5 - Transform the classes labels into a binary matrix
    y_train = tf.keras.utils.to_categorical(y_train) # one hot encoding for the labels
    y_test = tf.keras.utils.to_categorical(y_test)
    num_classes = y_test.shape[1]  # number of classes


    #TODO - Application 1 - Step 6 - Build the model architecture - Call the baseline_model function
    model = baseline_model(num_pixels, num_classes)

    #TODO - Application 1 - Step 7 - Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2) # training the model with 10 epochs and batch size of 200
    model.save_weights("mlp_mnist_model.weights.h5")  # save the model weights after training
    print("Saved model weights to mlp_mnist_model.weights.h5")

    #TODO - Application 1 - Step 8 - System evaluation - compute and display the prediction error
    scores = model.evaluate(x_test, y_test, verbose=0) # evaluate the model on the test data
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100)) # scores[1] is the accuracy, so 100 - accuracy gives the error percentage
    #print("Test MSE:", scores[1])

    return


def CNN_model(input_shape, num_classes): # input shape is (28,28,1) and num_classes is 10 for MNIST

    # TODO - Application 2 - Step 5a - Initialize the sequential model
    model = tf.keras.models.Sequential()

    # #TODO - Application 2 - Step 5b - Create the first hidden layer as a convolutional layer
    # model.add(layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=input_shape)) # 8 filters of size 3x3, relu activation function

    # #TODO - Application 2 - Step 5c - Define the pooling layer
    # model.add(layers.MaxPooling2D(pool_size=(2, 2))) # max pooling with pool size of 2x2

    # #TODO - Application 2 - Step 5d - Define the Dropout layer
    # model.add(layers.Dropout(0.2)) # dropout layer with dropout rate of 0.2 to prevent overfitting 

    # #TODO - Application 2 - Step 5e - Define the flatten layer
    # model.add(layers.Flatten()) # flatten the 2D feature maps to 1D feature vectors

    # #TODO - Application 2 - Step 5f - Define a dense layer of size 128
    # model.add(layers.Dense(128, activation='relu')) # dense layer with 128 neurons and relu activation function

    # #TODO - Application 2 - Step 5g - Define the output layer
    # model.add(layers.Dense(num_classes, activation='softmax')) # output layer with softmax activation for multi-class classification

    # #TODO - Application 2 - Step 5h - Compile the model
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # using categorical crossentropy loss for multi-class classification

    # Question 9
    model.add(layers.Conv2D(30, (5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(15, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def trainAndPredictCNN(x_train, y_train, x_test, y_test):

    #TODO - Application 2 - Step 2 - reshape the data to be of size [samples][width][height][channels]
    x_train = x_train.reshape((x_train.shape[0], 28, 28,1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28,1))
    #TODO - Application 2 - Step 3 - normalize the input values from 0-255 to 0-1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    #TODO - Application 2 - Step 4 - One hot encoding - Transform the classes labels into a binary matrix
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    #TODO - Application 2 - Step 5 - Call the CNN_model function
    model = CNN_model((28, 28, 1), y_test.shape[1]) # input shape is (28,28,1) and number of classes is y_test.shape[1]

    #TODO - Application 2 - Step 6 - Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=200, verbose=2) # training the model with 5 epochs and batch size of 200

    #TODO - Application 2 - Step 7 - Final evaluation of the model - compute and display the prediction error
    scores = model.evaluate(x_test, y_test, verbose=0) # evaluate the model on the test data
    print("CNN Accuracy: %.2f%%" % (scores[1] * 100)) # scores[1] is the accuracy, so 100 - accuracy gives the error percentage

    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():

    #TODO - Application 1 - Step 1 - Load the MNIST dataset in Tensorflow
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    #TODO - Application 1 - Step 2 - Train and predict on a MLP - Call the trainAndPredictMLP function
    #trainAndPredictMLP(x_train, y_train, x_test, y_test)
    

    #TODO - Application 2 - Step 1 - Train and predict on a CNN - Call the trainAndPredictCNN function
    trainAndPredictCNN(x_train, y_train, x_test, y_test)

    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################