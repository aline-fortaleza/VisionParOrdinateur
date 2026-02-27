import matplotlib.pyplot as plt
import numpy as np
import os
import keras
import tensorflow as tf

from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from operator import truediv
from utils import display_activation, plotImage, plotImages, plot_filter, plot_loss, calc_gradcam_heatmap, superimpose_image


def CNN(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, name='conv2d_1'))
    model.add(Conv2D(64, (3, 3), activation='relu',name='conv2d_2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=AdamW(learning_rate=0.001),  
                  metrics=['accuracy'])
    return model



def process_data():
    
    (x_train, y_train), (x_test, y_test)  = mnist.load_data()
    
    #Exercise 4: 
    #(x_train, y_train), (x_test, y_test)  = fashion_mnist.load_data()
    
    #plotImages(x_train,y_train)

    # Lets store the number of rows and columns
    img_rows = x_train[0].shape[0]
    img_cols = x_train[0].shape[1]
    
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    # store the shape of a single image 
    input_shape = (img_rows, img_cols, 1)
    
    # change our image type to float32 data type
    x_train = x_train.astype('float32') #uint8 originally
    x_test = x_test.astype('float32')
    
    # Normalize our data by changing the range from (0 to 255) to (0 to 1)
    x_train /= 255.0
    x_test /= 255.0
    
    # Now we one hot encode outputs
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    num_classes = y_test.shape[1]
    
    return  x_train, y_train, x_test, y_test, input_shape, num_classes


def train():
    
    #Application 1 Step 2, call the process data to get necessary processed data
    x_train,y_train, x_test, y_test,input_shape, num_classes = process_data()

    
    #Application 1 Step 3, define the model
    model = CNN(input_shape,num_classes)

    #Application 1 Step 4 show the summary of the models.
    print(model.summary())
    
    #Application 1 Step 5 fit the model
    history = model.fit(x_train,y_train,batch_size = 32,epochs = 10,verbose = 1,validation_data = (x_test, y_test))

    #Application 1 Step 6 plot the history
    plot_loss(history)
    
    #Application 1 Step 7 Evaluate the model loss and accuracy
    score = model.evaluate(x_test, y_test, verbose=0) 
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    #Appicaiton 1 step 8, save the weights
    model.save_weights('./F-MNISTFilter')# or model.save_weights('./F-MNISTFilter.weight.h5')

def inspect():
    
    #Application 2 Step 2, load the dataset
    x_train,y_train, x_test, y_test,input_shape, num_classes = process_data()

    #Application 2 step 3 load the model and its weights
    model = CNN(input_shape,num_classes)
    model.load_weights('./F-MNISTFilter')# or model.load_weights('./F-MNISTFilter.weight.h5')

    #Application 2 step 4, perform filter explorations by iterating the each of model layer and print the name. Also, print the filter weights and biases for convolution layer
    for layer in model.layers:
        print(layer.name)
        if 'conv' in layer.name:
            filters, biases = layer.get_weights()
            print(filters.shape, biases.shape)

    #Application 2 Step 5, directly access the first layer using indexing (use variable indexing to hold the indexer)
    indexing = 0
    filters, biases = model.layers[indexing].get_weights() 
    print(f"Filter Shape : {filters.shape}")
    print(f"Filters : {filters}")
    print(f"Bias Shape : {biases.shape}")
    print(f"Bias : {biases}")
    
    #Application 2 step 6, normalise the filter between 0 to 1 for visualisation and print the filters
    f_min, f_max = filters.min(), filters.max()
    print(f'Before Normalisation, Min = {f_min} and Max = {f_max}')
    filters = (filters - f_min) / (f_max - f_min)
    print(f'After Normalisation, Min = {filters.min()} and Max = {filters.max()}')

    #Application 2 Step 7 plot the filters and save manually
    plot_filter(filters, 4, int(truediv(filters.shape[-1],4)))
    
    #Application 2 step 8 get the layers to a list for sucessive features extractions. Use indexing on the model.layers
    layer_outputs = [layer.output for layer in model.layers[0:7]] 
    print(f"Layer Outputs: {layer_outputs}")

    #Application 2 step 9 use the defined output layers as parameter to current model, and use model.input as input parameters
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    #Application 2 step 10, get an example of test data from x_test (through indexing) and put them to a variable
    example = 30
    img_tensor = x_test[example]

    #Application 2 step 11, plot the example image using plotImage() function
    plotImage(img_tensor)

    #Application 2 step 12 Expand a first dimension to enable batching 
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    #Application 2 step 13, perform prediciton using the defined activation_model (through predict function)
    activations = activation_model.predict(img_tensor)

    #Application 2 step 14, print the number of extracted activations 
    print("Number of layer activations: " + str(len(activations)))

    #Application 2 step 15 get first activations through array indexer, with using indexing variable of 0, and print the shape
    sel_activation = activations[indexing]
    print(f"First Layer Activation : {sel_activation.shape}")

    #Application 2 step 16 compare your activations with the sub output from model.summary for clarifications
    print(model.summary())

    #Application 2 step 17, plot the last features (32) from the first activation layer
    plt.matshow(sel_activation[0, :, :, -1], cmap='summer')
    plt.legend()

    #Application 2 step 18, use display activation function to plot the first activation
    display_activation(sel_activation, 4, int(truediv(sel_activation.shape[-1],4)))


def gradCam():
    
    #Application 3 step 2, load the dataset, model and its weights (Application 2 Step 2 and step 3).
    x_train,y_train, x_test, y_test,input_shape, num_classes = process_data()
    model = CNN(input_shape,num_classes)
    model.load_weights('./F-MNISTFilter')# or model.save_weights('./F-MNISTFilter.weight.h5')

    #Application 3 Step 3, target the last layer to attach the gradient
    layer_name = 'conv2d_2'
    
    #Application 3 step 4  try to test some image to calculate the associated heatmap.
    for i in range(5):
        experiment = x_test[[i+10*i]]
        make_heatmap = calc_gradcam_heatmap(experiment, model, layer_name)
        plt.imshow(superimpose_image(experiment, make_heatmap))
        plt.show()

    
def main():
    
    #Application 1 Step 1, call function train()
    #train()
    
    #Application 2 Step 1, call function inspect()
    inspect()
    
    #Application 3 Step 1, call function gradCam()
    gradCam()
    
    #Application 4, repeat application 1 - 3 for Fashion MNIST (change the loading data to load fashionmnist instead). Be careful to save the trained model with different name
    pass
    
if __name__ == '__main__':
    main()

