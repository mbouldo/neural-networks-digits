import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


def init_params_multi(config_dict):
    weights_and_biases = []

    weights_and_biases.append(
        {
            'W': np.random.rand(config_dict['layer_dimensions'][0], config_dict['first_layer_dimensions']) - 0.5,
            'b': np.random.rand(config_dict['layer_dimensions'][0], 1) - 0.5
        }
    )

    for i in range(len(config_dict['layer_dimensions'])):
        W = np.random.rand(config_dict['layer_dimensions'][i], config_dict['layer_dimensions'][i]) - 0.5
        b = np.random.rand(config_dict['layer_dimensions'][i], 1) - 0.5
        weights_and_biases.append(
           {
                'W': W,
                'b': b
           }
        )
    return weights_and_biases
   



print(data[1])

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def tanh(Z):
    return np.tanh(Z)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, W3, b3, X, activation_function_1, activation_function_2):
    Z1 = W1.dot(X) + b1
    A1 = activation_function_1(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = activation_function_2(Z2)

    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3

def ReLU_deriv(Z):
    return Z > 0

def leaky_ReLU(Z):
    return np.maximum(0.01 * Z, Z)

def leaky_ReLU_deriv(Z):
    return np.where(Z > 0, 1, 0.01)


def sigmoid_deriv(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def tanh_deriv(Z):
    return 1 - np.power(tanh(Z), 2)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2,Z3, A3, W1, W2,W3,  X, Y, deriv_activation_function, deriv_relu):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y # this difference is the error
    #A1 is the activation of the previous layer
    # m is the number of training examples

    dW2 = 1 / m * dZ2.dot(A1.T) 
    db2 = 1 / m * np.sum(dZ2)   
    
    
    dZ1 = W2.T.dot(dZ2) * deriv_activation_function(Z1)     
    dW1 = 1 / m * dZ1.dot(X.T) 
    db1 = 1 / m * np.sum(dZ1)

    dZ3 = W3.T.dot(dZ2) * deriv_relu(Z3)
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)

    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2,W3, b3, dW1, db1, dW2, db2,dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2,W3, b3



def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations, activation_function, deriv_activation_function):


    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    m_w = 0
    v_w = 0
    m_b = 0
    v_b = 0
    t = 0
    learning_rate = 0.01

    #W1, b1, W2, b2 = init_params()
    params = data = init_params_multi({
        'first_layer_dimensions': 784,
        'layer_dimensions': [10,10],
    })

    W1 = params[0]['W']
    b1 = params[0]['b']
    W2 = params[1]['W']
    b2 = params[1]['b']
    W3 = params[2]['W']
    b3 = params[2]['b']

    

    accuracy_array = [];
    alpha_array = [];
    
    prev_W1 = 0
    W1_array = []
    prev_b1 = 0
    b1_array = []
    prev_W2 = 0
    W2_array = []
    prev_b2 = 0
    b2_array = []

    prev_W3 = 0
    W3_array = []
    prev_b3 = 0
    b3_array = []




    outcomes = []

    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X, activation_function, ReLU) 
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y, deriv_activation_function, ReLU_deriv)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2,W3,b3, dW1, db1, dW2, db2,dW3, db3, alpha)

        # implementation of Adam

        m_w = beta1 * m_w + (1 - beta1) * dW1
        v_w = beta2 * v_w + (1 - beta2) * np.square(dW1)
        m_b = beta1 * m_b + (1 - beta1) * db1
        v_b = beta2 * v_b + (1 - beta2) * np.square(db1)

        t += 1
        
        m_w_hat = m_w / (1 - beta1 ** t)
        v_w_hat = v_w / (1 - beta2 ** t)
        m_b_hat = m_b / (1 - beta1 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)

        # update parameters
        W1 -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        b1 -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

        current_accuracy = get_accuracy(get_predictions(A2), Y)
        accuracy_array.append(current_accuracy)

        if i > 2:
            if(accuracy_array[i] > accuracy_array[i-1] and accuracy_array[i] > accuracy_array[i-2]):
                alpha = alpha * 1.2
                outcomes.append(1)
            else:
                if(accuracy_array[i]<accuracy_array[i-1]):
                    print('Accuracy is decreasing... and reverting to previous weights...' )

                    alpha = alpha * (1-(accuracy_array[i]/accuracy_array[i-1]))
                    
                    print(1/np.exp(accuracy_array[i]))
                    if(alpha < 0.0001):
                        # we are stuck in a local minima
                        alpha = 10
                    W1 = W1_array[-2]
                    b1 = b1_array[-2]
                    W2 = W2_array[-2]
                    b2 = b2_array[-2]
                    W3 = W3_array[-2]
                    b3 = b3_array[-2]

                    outcomes.append(-1)
                else:
                    outcomes.append(0)

        W1_array.append(W1)
        b1_array.append(b1)
        W2_array.append(W2)
        b2_array.append(b2)
        W3_array.append(W3)
        b3_array.append(b3)


        alpha_array.append(alpha)

        print("Iteration: ", i, "Accuracy: ", current_accuracy, "Alpha: ", alpha)

        if(current_accuracy>0.65):
            print("Iteration COMPLETED...", i)
            break
    return W1, b1, W2, b2, accuracy_array, alpha_array


W1, b1, W2, b2, accuracy_array, alpha_array = gradient_descent(X_train, Y_train, 0.10, 500, sigmoid, sigmoid_deriv)

fig, axs = plt.subplots(2)

axs[0].plot(accuracy_array)
axs[1].plot(alpha_array)
plt.show()
