# import os
import torch
from torch import nn
import numpy as np
import random

from globals import *
from neural import NeuralNetwork

class Organism:
    def __init__(self, id, input_dim, hidden_dim, output_dim, width, height, device="cpu"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        self.brain = NeuralNetwork(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.brain.net.apply(self.init_weights)

        self.id = id
        self.width = width
        self.height = height
        self.x = random.randint(0, self.width - 1)
        self.y = random.randint(0, self.height - 1)
    
    def get_action(self, input):
        with torch.no_grad():
            logits = self.brain(input)
        
        return np.argmax(logits).detach().numpy()
    
    def action_to_coords(self, action):
        x = action % 3
        y = action // 3

        x -= 1
        y -= 1

        return (x, y)
    
    def set_coords(self, x, y):
        self.x = x
        self.y = y

        self.x = max(min(self.x, self.width-1), 0)
        self.y = max(min(self.y, self.height-1), 0)
    
    def coords_allowed(self, coords):
        return (coords[0] >= 0 and coords[0] < self.width) and (coords[1] >= 0 and coords[1] < self.height)
    
    def update_coords(self, action):
        self.x += action[0]
        self.y += action[1]

        self.x = max(min(self.x, self.width-1), 0)
        self.y = max(min(self.y, self.height-1), 0)

    def get_id(self):
        return self.id
    
    def get_x(self):
        return self.x

    def get_y(self):
        return self.y
    
    def get_color(self):
        pass

    def print_weights(self):
        print(self.brain)
        print(self.brain.net[0].weight)
        print(self.brain.net[2].weight)

    def set_weights(self, weights):
        """
        weights is tuple of numpy matrices
        """
        with torch.no_grad():
            self.brain.net[0].weight = nn.Parameter(torch.from_numpy(weights[0]))
            self.brain.net[2].weight = nn.Parameter(torch.from_numpy(weights[1]))
        
        return True
    
    def get_weights(self):
        return (self.brain.net[0].weight.detach().numpy(), self.brain.net[2].weight.detach().numpy())

    def __str__(self):
        return f"id: {self.id}\nx: {self.x}\ny: {self.y}"
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)