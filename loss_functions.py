import numpy as np
import tensorflow as tf

def softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector.
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    # m = y.shape[0]
    m = y.shape[1]
    grad = softmax(X)
    grad[range(m),y] -= 1
    grad = grad/m
    return grad


class SoftmaxModel():
    """An implementation of a softmax classifier"""

    def __init__(self, state_size, out_size, learning_rate):
        """Initializes the model

        Args:
            output_dim: An int, this is the size of the output layer, on which classification/regression is computed

        """
        self.reservoir_states = tf.placeholder(name="reservoir_states_placeholder", dtype=tf.float32, shape=[1, state_size])
        self.weights_placeholder = tf.placeholder(tf.float32, shape=[state_size, out_size], name="weights_placeholder")
        self.weights = tf.Variable(self.weights_placeholder, name="weights")
        self.set_weigths = tf.assign(self.weights, self.weights_placeholder, validate_shape=False)
        self.biases_placeholder = tf.placeholder(tf.float32, shape=[1, out_size], name="biases_placeholder")
        self.biases = tf.Variable(self.biases_placeholder, name="biases")
        self.set_biases = tf.assign(self.biases, self.biases_placeholder, validate_shape=False)
        self.labels = tf.placeholder(name="labels", shape=[1, out_size], dtype=tf.int32)
        self.learning_rate = learning_rate
        self.run_model()

    def run_model(self):
        self.cost, self.logits, self.losses = self.softmax()
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

    def softmax(self):
        logits = tf.nn.relu(tf.matmul(self.reservoir_states, self.weights) + self.biases)
        # logits = tf.matmul(self.reservoir_states, self.weights) + self.biases
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
        cost = tf.reduce_mean(losses)
        return cost, logits, losses
