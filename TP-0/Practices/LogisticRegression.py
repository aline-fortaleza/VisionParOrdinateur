import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.metrics import accuracy_score
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
# TODO - Application 3 - Step 5 - Create the ANN model
def modelDefinition():

    # TODO - Application 3 - Step 5a - Define the model as a Sequential model
    model = Sequential()  # Modify this

    # TODO - Application 3 - Step 5b - Add a Dense layer with 8 neurons to the model
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))

    # NEW hidden layer (16 neurons)
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))


    # TODO - Application 3 - Step 5c - Add a Dense layer (output layer) with 1 neuron
    model.add(Dense(1,kernel_initializer='normal'))


    # TODO - Application 3 - Step 5d - Compile the model by choosing the optimizer(adam) ant the loss function (MSE)
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def main():

    # TODO - Application 3 - Step 1 - Read data from "Houses.csv" file
    csvFile = pd.read_csv("./Houses.csv").values

    # TODO - Application 3 - Step 2 - Shuffle the data
    np.random.shuffle(csvFile)


    # TODO - Application 3 - Step 3 - Separate the data from the labels (x_data / y_data)
    #x_data = csvFile[:, :-1]
    #y_data = csvFile[:, -1]

    y_data = csvFile[:, 4]   # noxious
    x_data = np.delete(csvFile, 4, axis=1)  # remove noxious column so we can predict it


    # TODO - Application 3 - Step 4 - Separate the data into training/testing dataset
    split_index = int(0.8 * len(x_data))
    x_train = x_data[:split_index]
    y_train = y_data[:split_index]

    x_test = x_data[split_index:]
    y_test = y_data[split_index:]


    # TODO - Application 3 - Step 5 - Call the function "modelDefinition"
    model = modelDefinition()

    # TODO - Application 3 - Step 6 - Train the model for 100 epochs and a batch of 16 samples
    model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=2)

    # TODO - Application 3 - Step 7 - Predict the house price for all the samples in the testing dataset
    predictions = model.predict(x_test)

    # TODO - Exercise 8 - Compute the MSE for the test data
    mse = np.mean((y_test - predictions.flatten())**2)
    print("Mean Square Error = {}".format(mse))


    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################