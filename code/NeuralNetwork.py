import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.io import loadmat

def sigmoid(z):
    """
    Compute the sigmoid function of z
    :param z: array 1D vecotr or 2D matrix
    :return: array or matrix of sigmoid func for every element
    """
    z = np.array(z)
    g = np.zeros(z.shape)
    g = (1/(1+np.exp(-z)))
    return g
def sigmoidGradient(z):
    g = np.zeros(z.shape)
    g = sigmoid(z)*(1 - sigmoid(z))
    return g

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_=0.0):

    """

    :param nn_params: size of Theta1+Theta2
    :param input_layer_size: No of predictors (without bias)
    :param hidden_layer_size: No of neurons in hidden layer - 25
    :param num_labels: No of output classes
    :param X: INPUT DATASET - matrix of shape (m x input_layer_size)
    :param y: TARGET VARIABLE - vector of shape (m,)
    :param lambda_: regularization parameter
    :return: cost function and unrolled vector of partial derivates of the concatenation of Theta1 and Theta2!
    """

    """
    Imamo listu od Theta1+Theta2 parametara i treba da je pretvorimo u matricu gde u prvu matricu treba ubaciti 25x401 elemenata tj. to su vi elementi od 0:25*(20*20+1) elementa upakovani u shape (25,401)
    """
    Theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size,input_layer_size+1))

    Theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,(hidden_layer_size+1)))

    m = y.size # No of observations of training set

    #What do we need to calculate:
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    a1 = np.concatenate([np.ones((m,1)),X],axis=1) # na matricu X dodati jednu kolonu jedinica koji su biases ne bismo li dobili a1 layer.
    a2 = sigmoid(a1.dot(Theta1.T)) # kada pomnozimo transp. Theta1 matricu i matricu a1 i svaki element novog proizvoda ubacimo u sigmoidnu fju, dobijamo a2

    a2 = np.concatenate([np.ones((a2.shape[0],1)),a2],axis=1)
    a3 = sigmoid(a2.dot(Theta2.T)) # PREDICTED VALUES!!!

    y_matrix = y.reshape(-1)
    y_matrix = np.eye(num_labels)[y_matrix]

    temp1 = Theta1
    temp2 = Theta2

    #Add regularization term!
    reg_term = (lambda_ / (2 * m)) * (np.sum(np.square(temp1[:, 1:])) + np.sum(np.square(temp2[:, 1:])))

    J = (-1/m) * np.sum((np.log(a3)*y_matrix)+(np.log(1-a3)*(1-y_matrix))) + reg_term

    #Backpropagation:
    delta_3 = a3-y_matrix # differences between predicted and real values
    delta_2 = delta_3.dot(Theta2)[:,1:]*sigmoidGradient(a1.dot(Theta1.T)) # sigGrad(z(2))

    Delta1 = delta_2.T.dot(a1)
    Delta2 = delta_3.T.dot(a2)

    #Adding regularization

    Theta1_grad = (1/m)*Delta1
    Theta2_grad = (1/m)*Delta2
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]

    grad = np.concatenate([Theta1_grad.ravel(),Theta2_grad.ravel()])
    return J, grad

def rand_initialize_weights(L_in, L_out,epsilon_init=0.12):
    """
    Random initialization of weights of a layer in neural network

    :param L_in:  No of incoming connections
    :param L_out: No of outgoing connections
    :param epsilon_init:
    Range of values which the weight can take from a uniform distribution

    :return:
    W as weights matrix of randomly initialised values in (L_out,1+L_in) shape.
    """
    W = np.zeros((L_out,1+L_in))
    W = np.random.rand(L_out,1+L_in)*2*epsilon_init - epsilon_init
    return W

def debug_initialize_weights(fan_out, fan_in):
    """
    Initialize the weights of a layer with fan_in incoming connections and fan_out outgoings
    connections using a fixed strategy. This will help you later in debugging.

    Note that W should be set a matrix of size (1+fan_in, fan_out) as the first row of W handles
    the "bias" terms.

    Parameters
    ----------
    fan_out : int
        The number of outgoing connections.

    fan_in : int
        The number of incoming connections.

    Returns
    -------
    W : array_like (1+fan_in, fan_out)
        The initialized weights array given the dimensions.
    """
    # Initialize W using "sin". This ensures that W is always of the same values and will be
    # useful for debugging
    W = np.sin(np.arange(1, 1 + (1+fan_in)*fan_out))/10.0
    W = W.reshape(fan_out, 1+fan_in, order='F')
    return W

def compute_numerical_gradient(J, theta, e=1e-4):
    """
    Computes the gradient using "finite differences" and gives us a numerical estimate of the
    gradient.

    Parameters
    ----------
    J : func
        The cost function which will be used to estimate its numerical gradient.

    theta : array_like
        The one dimensional unrolled network parameters. The numerical gradient is computed at
         those given parameters.

    e : float (optional)
        The value to use for epsilon for computing the finite difference.

    Notes
    -----
    The following code implements numerical gradient checking, and
    returns the numerical gradient. It sets `numgrad[i]` to (a numerical
    approximation of) the partial derivative of J with respect to the
    i-th input argument, evaluated at theta. (i.e., `numgrad[i]` should
    be the (approximately) the partial derivative of J with respect
    to theta[i].)
    """
    numgrad = np.zeros(theta.shape)
    perturb = np.diag(e * np.ones(theta.shape))
    for i in range(theta.size):
        loss1, _ = J(theta - perturb[:, i])
        loss2, _ = J(theta + perturb[:, i])
        numgrad[i] = (loss2 - loss1)/(2*e)
    return numgrad

def predict(Theta1, Theta2, X):
    """
    Predict the label of an input given a trained neural network
    Outputs the predicted label of X given the trained weights of a neural
    network(Theta1, Theta2)
    """
    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros(m)
    h1 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), Theta1.T))
    h2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), Theta2.T))
    p = np.argmax(h2, axis=1)
    return p

def display_data(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data stored in X in a nice grid.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        # Display Image
        h = ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                      cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')





