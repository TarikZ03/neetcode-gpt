import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list

        #xavier init tries to keep the variance of activations roughly the same from layer to layer, so if input has std=1 then output should also have std=1 and the next layer also has std=1 etc.
        
        torch.manual_seed(0)
        
        #2 is there just because of simplifiying avereging fan_in and fan_out
        std = torch.sqrt(torch.tensor(2/(fan_in+fan_out)))
        W = torch.randn(fan_out, fan_in) * std

        return torch.round(W, decimals=4).tolist()


    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        
        torch.manual_seed(0)

        #torch.sqrt only works on torch tensor so we wrap the result in a torch tensor and this just becomes tensor(whatever the result of 2/fan_in is)
        #the 2 here actually doubles the variance because ReLu kills half of neurons so it doubles the variance, the 2 does not come from averaging but trying to combat relu killing half the distribution, i.e. everything below 0
        std = torch.sqrt(torch.tensor(2/fan_in))

        W = torch.randn(fan_out, fan_in) * std

        return torch.round(W, decimals=4).tolist()
        


    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.

        torch.manual_seed(0)
            
        stds = []
        #store weights per layer
        weights = []
        #initialize layers
        for i in range(num_layers):

            if init_type == "xavier":
                #for the first layer the fan_in=input_dim but for any other layer, the fan_in is the hidden_dim
                std = torch.sqrt(torch.tensor(2/(input_dim+hidden_dim if i==0 else hidden_dim+hidden_dim)))
                W = torch.randn(hidden_dim, input_dim if i==0 else hidden_dim) * std
                weights.append(W)

            if init_type == "kaiming":
                std = torch.sqrt(torch.tensor(2/(input_dim if i==0 else hidden_dim)))
                W = torch.randn(hidden_dim, input_dim if i==0 else hidden_dim) * std
                weights.append(W)

            if init_type == "random":
                W = torch.randn(hidden_dim, input_dim if i==0 else hidden_dim)
                weights.append(W)
        
        #create a random input vector
        #the checker wants x to be created here, so the first batch to random numbers is consumed by the weight init and then x takes the last one
        x = torch.randn(input_dim)

        #forward pass
        for i in range(len(weights)):
            h = torch.relu(weights[i] @ x)
            x = h
            stds.append(torch.std(h))

        #each s in stds is a tensor so we need item converts it back to python float and we round each number
        return [round(s.item(), 2) for s in stds]