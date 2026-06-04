import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        


        for i in range(len(weights)):
            
            #if it is the final layer dont apply activation but just spit out the raw numbers
            if i+1 == len(weights):
                out = x @ weights[i] + biases[i]
                break

            #for the current hidden layer
            h = np.maximum(x @ weights[i] + biases[i], 0)
            #x gets reassigned to be the previous layers hidden state, so in the next iter x will be the input to the next layer
            x = h
            
        return np.round(out, 5)