"""
This is a kind of a student project to build as modular as possible neural net used for classification problems.
The major source of inspiration is drawn form an excellent course of Andrew Ng Stanford CS231n.
Another portion of knowledge is retrieved from a master piece of Ian Goodfellow et al. - Deep Learning book.
Author: Daniel Stancl
"""
##### Import packages
import numpy as np
import sys

a = np.random.normal(0, 2, (100, 10))
y = np.random.randint(0, 5, (100,))

##### Single components of forward pass of neural networks.
def affine_forward(X, W, b):
    """
    Computes the forward pass for an affine transformtion.
    The input X should jave a shape [X, d_1,...,d_k].
    N determines the number of obsevations, other dimensions specifies the shape of each example.
    
    :param X: A numpy array conaining input data of the shape [N, d_1,...,d_k]
    :param W: A numpy array of weights with shape [D, M]
    :param b: A numpy array brining a bias term with shape [M, ]
      
    :return out: A numpy array of output of shape [N, M]. Output is given as W^T*X + b
    """
    try:
        out = np.array( [np.dot(W.transpose(), x.reshape(-1)) + b for x in X] )
        return out
    except Exception as e:
        print("Problem with affine forward:", e)
        sys.exit()

def relu_forward(X):
    """
    Compute the forward pass for an operation ReLu
    
    :param X: Inputs of any shape. Numpy array
    :return out: Output of the same shape as X. Numpy array
    """
    try:
        out = np.array( [x if x > 0 else 0 for x in X.reshape(-1)] ).reshape(X.shape)
        return out
    except Exception as e:
        print("Problem with ReLu forward:", e)
        sys.exit()

def softmax(X):
    """
    Computes the forward pass for the softmax function.
    
    :param X: Inputs of shape [N,M]. Numpy array
    :return out: Output of the same shape as X. Numpy array
    """
    try:
        out = np.array( [np.exp(x) / np.exp(x).sum() for x in X] )
        return out
    except Exception as e:
        print("Problem with softmax function:", e)
        sys.exit()

##### Sandwich layer of forward pass of a MLP. Generally, use from a convenction.
def affine_relu_forward(X, W, b):
    """
    Sandwich layer performing forward pass of an affine transformation (W^T*X + b) followed by ReLu.
    
    :param X: A numpy array conaining input data of the shape [N, d_1,...,d_k]
    :param W: A numpy array of weights with shape [D, M]
    :param b: A numpy array brining a bias term with shape [M, ]
      
    :return out: A numpy array of output of shape [N, M]. Output is given as ReLu(W^T*X + b)
    """
    try:
        a = affine_forward(X, W, b)
        out = relu_forward(a)
        return out
    except Exception as e:
        print("Problem with affine relu forward:", e)
        sys.exit()

def affine_softmax_forward(X, W, b):
    """
    Sandwich layer performing forward pass of an affine transformaion (W^T*X + b) followed by softmax function.
    
    :param X: A numpy array conaining input data of the shape [N, d_1,...,d_k]
    :param W: A numpy array of weights with shape [D, M]
    :param b: A numpy array brining a bias term with shape [M, ]
      
    :return out: A numpy array of output of shape [N, M]. Output is given as softmax(W^T*X + b)
    """
    try:
        a = affine_forward(X, W, b)
        out = softmax(a)
        return out
    except Exception as e:
        print("Problem with affine softmax forward:", e)
        sys.exit()
   

##### Single components of backward pass of neural networks - derivative of non-linear functions.


##### initializing weights and bias
def initialize_weights(input_shape, output_shape, mean, std):
    """
    A function initializing a matrix of weights with a predetermined shape
    
    :par input_shape: A number determining a number of rows of the matrix. Integer
    :par output_shape: A number determining a number of columns of the matrix. Integer
    :par mean: A mean of normal distribution elements of a matrix are drawn from. Float
    :par std: A standard deviation of normal distribution elements of a matrix are drawn from. Float

    :return W: Initialized matrix of weights with a shape [input_shape, output_shape]    
    """
    try:
        W = np.random.normal(mean, std, (input_shape, output_shape))
        return W
    except Exception as e:
        print("Problem with weight matrix initialization:", e)
        sys.exit()

def initialize_bias(output_shape, par):
    """
    A function initializin a vectof of a bias term.
    
    :par output_shape: A number determining a shape of the vector. Integer
    :par par: A parameter determining an initial value of a bias term. Float.
    
    :return b: An initialized vector of bias terms with a shape [output_shape,]. 
    """
    try:
        b = par * np.ones(output_shape)
        return b
    except Exception as e:
        print("Problem with bias vector initialization:", e)
        sys.exit()
    
def initialize_weights_bias(X, y, n_nodes, mean = 0, std = 1, bias_par = 0.01):
    """
    A procedure obtaining a required shape of matrix of weights and bias terms followed by initialization of theirs.
    
    :param X: A numpy array containing input data of the shape [N, d_1,...,d_k]
    :param y: A numpy array containing labels for input data.
    :param n_nodes: A list of of integers determining the number of units in hidden layers. List
    
    :return weights: Initialized matrix of weights with a shape [input_shape, output_shape]    
    :return b: An initialized vector of bias terms with a shape [output_shape,]. 
    """
    try:
        output_shape = np.unique(y).shape[0]
        if len(n_nodes) > 1:
            shapes = [ [ X[0].reshape(-1).shape[0] ], n_nodes ]
            shapes = sum(shapes, [])
            shapes.append(output_shape)
        else:
            shapes = [ X[0].reshape(-1).shape[0], n_nodes, output_shape ]
        weights_values = [ initialize_weights(input_shape = shapes[i], output_shape = shapes[i+1], mean = mean, std = std) for i in range(len(shapes) - 1) ]
        weights_keys = [ f'W{i}' for i in range(1, len(weights_values) + 1) ]
        weights = dict( zip( weights_keys, weights_values) )
        bias_values = [ initialize_bias(output_shape = shapes[i], par = bias_par) for i in range(1, len(shapes)) ]
        bias_keys = [ f'b{i}' for i in range(1, len(bias_values) + 1) ]
        bias = dict( zip( bias_keys, bias_values) )
        return weights, bias
    except Exception as e:
        print('Problem with initialization of a matrix of weights and a vector of bias term:', e)
    
def initialize_layers(X, y, n_nodes):
    """
    Instance initializing input layer, hidden layer/s and output layer.
    
    :param X: Dataframe of inputs. Numpy array of shape [N, d_1,...,d_k]
    :param n_nodes: List containing the number of nodes in individual hidden layers. List of integers
    :param y: Vector of label for the corresponding training data set. Numpy array of shape [N.]
    
    :return layers: Dictionary containing reshaped array of inputs, hidden layers and output.
    """
    N = X.shape[0] # number of examples
    output_shape = np.unique(y).shape[0]
    layers = dict()
    layers['input'] = X.reshape(N, -1)   
    for i in range( len(n_nodes) ):
        layers['h{}'.format(i+1)] = np.zeros( (N, n_nodes[i]) )
    layers['output'] = np.zeros( (N, output_shape) )
    return layers
        
##### Feedforwards pass
class feedforward_pass:
    """
    Class which executes a feedforward pass for a neural network.
    Currently supported feedforward pass are ['relu', 'softmax']
    """
    def __init__(self):
        """
        Class initialization.
        """
    def relu(self, X, W, b):
        """
        Affine-relu transformaton.
        :param X: A numpy array conaining input data of the shape [N, d_1,...,d_k]
        :param W: A numpy array of weights with shape [D, M]
        :param b: A numpy array brining a bias term with shape [M, ]
      
        :return out: A numpy array of output of shape [N, M]. Output is given as relu(W^T*X + b)     
        """
        self.out = affine_relu_forward(X, W, b)
        return self.out
    def softmax(self, X, W, b):
        """
        Affine-softmax transformation
        :param X: A numpy array conaining input data of the shape [N, d_1,...,d_k]
        :param W: A numpy array of weights with shape [D, M]
        :param b: A numpy array brining a bias term with shape [M, ]
      
        :return out: A numpy array of output of shape [N, M]. Output is given as relu(W^T*X + b)     
        """
        self.out = affine_softmax_forward(X, W, b)
        return self.out
         
##### Class for loss functions
class loss:
    """
    Class computing the value of predefined loss function for classification problem.
    
    :param loss_function: Type of loss function a user desires to use. String
    :param y: A vector of true labels. Numpy array of integers
    :param y_pred: A vector of probabilites. Numpy array of floats
    
    :return loss: Value of loss funciton. Float
    """
    def __init__(self):
        """
        anc
        """
    def cross_entropy(self, y, y_pred, reg = 0.0):
        self.y_pred = -np.log(y_pred[range(y.shape[0]), y])
        self.loss = np.mean(self.y_pred)
        return self.loss
       
##### Multi-layer perceptron
class MultiLayerPerceptron:
    """
    Fully connected neural networks with any number of hidden layers.
    
    Components: fit, predict
    """
    def __init__(self):
        """
        Initialize the function.
        """
    def fit(self, X_train, y_train, X_val, y_val, n_nodes, activation_functions, loss_function):
        """
        A function fitting the neural network.
        
        :param X_train: Training dataset. Numpy array of any shape [N, d_1,...,d_k]
        :param X_val: Validation dataset. Numpy array of shape [K, d_1,...,d_k]
        :param y_train: Labels for trainind dataset. Numpy array of shape [N,]
        :param y_val: Lales for validation dataset. Numpy array of shape [K,]
        :param n_nodes: List of numbers determining the number of nodes in each hidden layer. List of any length.
        :param activation_functions: List containing information about which activation functions should be used in hidden layers. List with the same length as n_nodes.
            List of currently supported activation functions: ['relu', 'softomax']
        :param loss_function: A string determining the type of loss function used for computation. String
            List of currently supported loss functions ['cross_entropy']
        """
        ### Define desired loss function
        self.loss_function = getattr(loss(), loss_function)
        ### Check if the number of hidden layers given by a length of the list n_nodes corresponds to the length of the list activation_function
        if (len(n_nodes) + 1) != len(activation_functions):
            print("The number of hidden layers does not correspond to the number of filled activation functions.\n It must hold (len(n_nodes) + 1) = len(activation_functions).")
            sys.exit()
        ### Initialize weight matrices and vectors with bias terms
        self.weights, self.bias = initialize_weights_bias(X = X_train, y = y_train, n_nodes = n_nodes)
        ### Initialize a dictionary with input, hidden and ouput layers
        self.layers_train = initialize_layers(X = X_train, y = y_train, n_nodes = n_nodes)
        self.layers_val = initialize_layers(X = X_val, y = y_train, n_nodes = n_nodes)
        ### Feedforward pass for training data
        i = 0
        for func, W, b in zip(activation_functions, self.weights, self.bias):
            self.feedforward_pass = getattr(feedforward_pass(), func)
            self.layers_train[ [*self.layers_train][i+1] ] = (
                    self.feedforward_pass(X = self.layers_train[ [*self.layers_train][i] ],
                                          W = self.weights[W],
                                          b = self.bias[b]
                                          )
                    )
            i += 1
        
        self.loss = self.loss_function(y = y_train, y_pred = self.layers_train['output'])
        return self.loss
        
    
model = MultiLayerPerceptron()
model.fit(X_train = a, y_train = y, X_val = a, y_val = y, n_nodes = [20, 29], activation_functions = ['relu', 'relu', 'softmax'], loss_function = 'cross_entropy')
