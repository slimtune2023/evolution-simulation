import random
import numpy as np
import torch

from globals import *
from organism import Organism

class Simulation:
    def __init__(self, width=100, height=100, num_organisms=1, dims=(9, 12, 9), device="cpu"):
        self.width = width
        self.height = height
        self.num_organisms = num_organisms
        self.dims = dims
        self.device = device

        self.ids = list(range(self.num_organisms))
        self.organisms = []
        self.world = []

        self.epoch = 0
        self.time = 0

        self.mutation_prob = 0.05

        self.initialize()

    def initialize(self):
        for id in self.ids:
            organism = Organism(id, self.dims[0], self.dims[1], self.dims[2], self.width, self.height, self.device)
            self.organisms.append(organism)
        
        self.set_coords()

    def set_coords(self):
        coordinates = list(range(self.width * self.height))
        random.shuffle(coordinates)
        
        self.reset_world()

        for i in range(self.num_organisms):
            coord = coordinates[i]
            x = coord % self.width
            y = coord // self.width

            self.organisms[i].set_coords(x, y)
            self.world[x][y] = self.organisms[i].get_id()

    def reset_world(self):
        self.world = [[-1] * self.height for i in range(self.width)]

    def step(self):
        self.ids = list(range(self.num_organisms))
        random.shuffle(self.ids)

        for id in self.ids:
            organism = self.organisms[id]
            x = organism.get_x()
            y = organism.get_y()
            
            input = []
            
            for i in range(x-1, x+2):
                for j in range(y-1, y+2):
                    if (i < 0 or i >= self.width):
                        input.append(1)
                    elif (j < 0 or j >= self.height):
                        input.append(1)
                    elif (self.world[i][j] >= 0):
                        input.append(1)
                    else:
                        input.append(0)
            
            input = torch.tensor([input], dtype=torch.float32)
            action = organism.get_action(input)
            coords = organism.action_to_coords(action)

            prev_x = x
            prev_y = y

            self.world[prev_x][prev_y] = -1

            x += coords[0]
            y += coords[1]

            if (organism.coords_allowed((x, y))):
                if (self.world[x][y] < 0):
                    self.world[x][y] = id
                    organism.set_coords(x, y)
                else:
                    self.world[prev_x][prev_y] = id
                    organism.set_coords(prev_x, prev_y)
            else:
                self.world[prev_x][prev_y] = id
                organism.set_coords(prev_x, prev_y)
        
        self.time += 1
    
    def evolve(self):
        survived = []

        for organism in self.organisms:
            org_survive = self.natural_selection(organism)
            if (org_survive):
                survived.append(organism.get_id())
        
        survival_rate = 1.0 * len(survived) / self.num_organisms
        print(f"epoch {self.epoch}: survival rate " + "{:.3f}".format(survival_rate))
        
        gene_pool = [self.organisms[id].get_weights() for id in survived]

        for id in self.ids:
            organism = self.organisms[id]
            weights = self.mutate(gene_pool[random.randint(0, len(gene_pool)-1)])

            organism.set_weights(weights)
        
        self.set_coords()
        self.epoch += 1
        self.time = 0
        
    def natural_selection(self, organism, mode=0):
        """
        returns true if the organism survives and false otherwise
        """
        if (mode == 0):
            x = organism.get_x()
            y = organism.get_y()

            if (x < self.width // 2):
                return True
            else:
                return False
        elif (mode == 1):
            x = organism.get_x()
            y = organism.get_y()
            count = 0

            for i in range(x-1, x+2):
                for j in range(y-1, y+2):
                    if (i < 0 or i >= self.width):
                        continue
                    elif (j < 0 or j >= self.height):
                        continue
                    elif (self.world[i][j] >= 0):
                        count += 1
                    else:
                        continue
            
            if (count > 1):
                return False
            else:
                return True
        elif (mode == 2):
            x = organism.get_x()
            y = organism.get_y()
            count = 0

            for i in range(x-1, x+2):
                for j in range(y-1, y+2):
                    if (i < 0 or i >= self.width):
                        continue
                    elif (j < 0 or j >= self.height):
                        continue
                    elif (self.world[i][j] >= 0):
                        count += 1
                    else:
                        continue
            
            if (count < 6):
                return False
            else:
                return True
        else:
            return True
    
    def mutate(self, weights):
        if (random.random() < self.mutation_prob):
            if (random.random() < 0.5):
                dim = weights[0].shape
                mut = random.randint(0, dim[0]*dim[1]-1)
                weights[0][mut % dim[0]][mut // dim[0]] *= -1.0
            else:
                dim = weights[1].shape
                mut = random.randint(0, dim[0]*dim[1]-1)
                weights[1][mut % dim[0]][mut // dim[0]] *= -1.0
        
        return weights

    def get_world(self):
        return self.world

    def print_world(self):
        print(" " + "-" * self.width + " ")
        for row in self.world:
            s = "|"
            for x in row:
                if (x < 0):
                    s += " "
                else:
                    s += "X"
            s+= "|"
            print(s)
        print(" " + "-" * self.width + " ")