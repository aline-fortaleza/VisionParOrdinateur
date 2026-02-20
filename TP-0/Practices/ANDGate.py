#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def activationFunction(n):

    #TODO - Application 1 - Step 4b - Define the binary step function as activation function
    #if n >= 0:
    #    return 1
    #else:
    #    return 0
    # Change activation for sigmoid function:
    e = 2.718281828459045
    act = 1 / (1 + (e ** (-n)))
    if act >= 0.5:
        return 1    
    else:
        return 0
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def forwardPropagation(p, weights, bias):

    a = None # the neuron output

    # TODO - Application 1 - Step 4a - Multiply weights with the input vector (p) and add the bias   =>  n
    n = weights[0]*p[0] + weights[1]*p[1] + bias 


    # TODO - Application 1 - Step 4c - Pass the result to the activation function  =>  a
    a = activationFunction(n)



    return a
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():

    #Application 1 - Train a single neuron perceptron in order to predict the output of an AND gate.
    #The network should receive as input two values (0 or 1) and should predict the target output


    #Input data
    P = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]

    #Labels
    t = [0, 0, 0, 1] # for AND gate
    #t = [0,1,1,1] # for OR gate

    #TODO - Application 1 - Step 2 - Initialize the weights with zero  (weights)
    w = [0,0] # weights for the two inputs
    

    #TODO - Application 1 - Step 2 - Initialize the bias with zero  (bias)
    b = 0 # bias

    #TODO - Application 1 - Step 3 - Set the number of training steps  (epochs)
    epochs = 5

    #TODO - Application 1 - Step 4 - Perform the neuron training for multiple epochs
    for ep in range(epochs):
        for i in range(len(t)):

            #TODO - Application 1 - Step 4 - Call the forwardPropagation method
            a = forwardPropagation(P[i], w, b)

            #TODO - Application 1 - Step 5 - Compute the prediction error (error)
            error = t[i] - a

            #TODO - Application 1 - Step 6 - Update the weights
            w[0] = w[0] + error * P[i][0]
            w[1] = w[1] + error * P[i][1]

            #TODO - Application 1 - Step 7 - Update the bias
            b = b + error # bnew = bold + error

            #looking for early stop 
            #all_correct = True
            #for i in range(len(t)):
            #    pred = forwardPropagation(P[i], w, b)
            #    if pred != t[i]:
            #        all_correct = False
            #        break

            #if all_correct:
            #    print(f"Early stop: converge on epoch {ep+1}")
            #    break 
            



    #TODO - Application 1 - Step 8 - Print weights and bias
    print("Trained weights: ", w)
    print("Trained bias: ", b)
   
    # TODO - Application 1 - Step 9 - Display the results
    for i in range(len(t)):
        a = forwardPropagation(P[i], w, b)
        print("Input: ", P[i], " Predicted values: ", a, " Ground truth label: ", t[i])

   
    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == "__main__":
    main()
#####################################################################################################################
#####################################################################################################################