import numpy as np
from typing import List


class Solution:

    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        

        #(2,)
        x = np.array(x, dtype=float)
        #(2, 2)
        W1 = np.array(W1, dtype=float)
        #(2,)
        b1 = np.array(b1, dtype=float)
        #(1, 2)
        W2 = np.array(W2, dtype=float)
        #(1,)
        b2 = np.array(b2, dtype=float)
        #(1,)
        y_true = np.array(y_true, dtype=float)

        
        #(2,)
        z1 = (W1 @ x) + b1
        #(2,)
        a1 = np.maximum(z1, 0)
        #(1,)
        z2 = (W2 @ a1) + b2

        loss = np.mean(np.power(z2-y_true, 2))

        #the shape of the derivative must be == to the shape of the thing you are differentiating wrt
        
        #(1,)
        db2 = 2*(z2-y_true)/len(y_true)
        #the shape of dW2 == the shape of W2
        #the None adds 1 new dimension, so db2 in this case would become (1, 1), and because a1 contains the hidden values, the result of multiplcation will be [[db2[0]*a1[0], db2[0]*a1[1]]], so its 1 row with 2 columns
        dW2 = db2[:, None] * a1  

        #(2,)
        #the transpose is needed to transform it from (1,2) to (2,1), and the squeeze removes the 1 so only (2,) is left
        db1 = np.squeeze(np.transpose((W2*2*(z2-y_true) * (z1>0)) / len(y_true)))
        #(2, 2)
    
        dW1 = db1[:, None] * x

        #floats are stored with a sign bit even for 0, but for Python/Numpy -0.0==0 but in the output it will be 0 so the checker is saying my solution is wrong
        def clean(arr):
            
            arr = np.round(arr, 4)
            #wherever there is a 0, so this will scan for both -0.0 and 0.0 because to python they are the same thing, but we are explictily setting it to 0
            arr[arr==0] = 0.0
            return arr


        return {"loss": float(np.round(loss, 4)), "dW1": clean(dW1).tolist(), "db1": clean(db1).tolist(), "dW2": clean(dW2).tolist(), "db2": clean(db2).tolist()}