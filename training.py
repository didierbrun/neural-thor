import random
import math
import pickle
import numpy as np

from neural.utils import mse, mse_prime, tanh, tanh_prime
from neural.activation_layer import ActivationLayer
from neural.fc_layer import FCLayer
from neural.network import Network

#
# Generate training map 
#
WIDTH = 40.0
HEIGHT = 18.0
COUNT = 1000

OUTS = {
    "N": 0,
    "NE": 1,
    "E": 2,
    "SE": 3,
    "S": 4,
    "SW": 5,
    "W": 6,
    "NW": 7
}

def train_set():
    xt = []
    yt = []
    #
    # Generate COUNT training map
    #
    for i in range(COUNT):
        #
        # Normalize input datas
        #
        sx = float(random.randrange(WIDTH) / WIDTH)
        ix = float(random.randrange(WIDTH) / WIDTH)
        sy = float(random.randrange(HEIGHT) / WIDTH)
        iy = float(random.randrange(HEIGHT) / WIDTH)

        #
        # Force change to get sx=ix or sy=iy scenarios
        #
        if random.randrange(3) == 1:
            sx = ix
        if random.randrange(3) == 1:
            sy = iy

        #
        # Be sure that the player is not already on the exit
        #
        while sx == ix and sy == iy:
            sx = float(random.randrange(WIDTH) / WIDTH)
            sy = float(random.randrange(HEIGHT) / WIDTH)

        #
        # Compute the good move - old fashioned algorithmic way
        #
        o = ''
        if sy < iy: o = o + 'N'
        if sy > iy: o = o + 'S'
        if sx < ix: o = o + 'W'
        if sx > ix: o = o + 'E'
      
        #
        # The output layer is composed on the 8 possible ways
        #
        b = [0.0] * 8
        b[OUTS[o]] = 1.0

        xt += [sx, ix, sy, iy]
        yt += b
    return(np.array(xt).reshape(COUNT, 1, 4), np.array(yt).reshape(COUNT, 1, 8))



#
# Neural network
#

#
# Get a single train set
#
xt, yt = train_set()
x_train = np.array(xt).reshape(COUNT, 1, 4)
y_train = np.array(yt).reshape(COUNT, 1, 8)

#
# Contruct the brain model network
#
net = Network()
net.add(FCLayer(4, 8))                          # Input layer : sx, sy, ix, iy
net.add(ActivationLayer(tanh, tanh_prime))      
net.add(FCLayer(8, 16))                         # Hidden layer
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(16, 8))                         # Output layer : N, NE, E, SE, etc...
net.add(ActivationLayer(tanh, tanh_prime))
net.use(mse, mse_prime)

#
# Train the neural network
#
net.fit(x_train, y_train, epochs = 500, learning_rate = 0.1)

#
# Export the trained brain as a long string contaning all biases & weights
#
brain = pickle.dumps([net.layers[0].weights,net.layers[0].bias,net.layers[2].weights,net.layers[2].bias,net.layers[4].weights,net.layers[4].bias])

# Displays it on the terminal so it can be copy/pasted 
print(brain)

