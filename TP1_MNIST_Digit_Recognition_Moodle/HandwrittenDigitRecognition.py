import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras import layers
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def baseline_model(num_pixels, num_classes):

    #TODO - Application 1 - Step 6a - Initialize the sequential model
    model = None   # Modify this

    #TODO - Application 1 - Step 6b - Define a hidden dense layer with 8 neurons


    #TODO - Application 1 - Step 6c - Define the output dense layer


    # TODO - Application 1 - Step 6d - Compile the model


    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def trainAndPredictMLP(X_train, Y_train, X_test, Y_test):

    #TODO - Application 1 - Step 3 - Reshape the MNIST dataset - Transform the images to 1D vectors of floats (28x28 pixels  to  784 elements)


    #TODO - Application 1 - Step 4 - Normalize the input values


    #TODO - Application 1 - Step 5 - Transform the classes labels into a binary matrix


    #TODO - Application 1 - Step 6 - Build the model architecture - Call the baseline_model function
    model = None   #Modify this


    #TODO - Application 1 - Step 7 - Train the model


    #TODO - Application 1 - Step 8 - System evaluation - compute and display the prediction error


    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def CNN_model(input_shape, num_classes):

    # TODO - Application 2 - Step 5a - Initialize the sequential model
    model = None   #Modify this


    #TODO - Application 2 - Step 5b - Create the first hidden layer as a convolutional layer


    #TODO - Application 2 - Step 5c - Define the pooling layer


    #TODO - Application 2 - Step 5d - Define the Dropout layer


    #TODO - Application 2 - Step 5e - Define the flatten layer


    #TODO - Application 2 - Step 5f - Define a dense layer of size 128


    #TODO - Application 2 - Step 5g - Define the output layer


    #TODO - Application 2 - Step 5h - Compile the model


    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def trainAndPredictCNN(X_train, Y_train, X_test, Y_test):

    #TODO - Application 2 - Step 2 - reshape the data to be of size [samples][width][height][channels]


    #TODO - Application 2 - Step 3 - normalize the input values from 0-255 to 0-1


    #TODO - Application 2 - Step 4 - One hot encoding - Transform the classes labels into a binary matrix


    #TODO - Application 2 - Step 5 - Call the cnn_model function
    model = None   #Modify this


    #TODO - Application 2 - Step 6 - Train the model


    #TODO - Application 2 - Step 8 - Final evaluation of the model - compute and display the prediction error


    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():

    #TODO - Application 1 - Step 1 - Load the MNIST dataset in Keras


    #TODO - Application 1 - Step 2 - Train and predict on a MLP - Call the trainAndPredictMLP function


    #TODO - Application 2 - Step 1 - Train and predict on a CNN - Call the trainAndPredictCNN function


    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
