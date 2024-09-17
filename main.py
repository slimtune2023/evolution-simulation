import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
from torch import nn
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from organism import Organism
from simulation import Simulation

def main():
    world_size = 60
    num_orgs = 200

    sim = Simulation(width=world_size, height=world_size, num_organisms=num_orgs)
    
    while True:
        if (sim.epoch % 5 == 0):
            fig, ax = plt.subplots()
            img = ax.matshow(np.asarray(sim.get_world()))
            cb = plt.colorbar(img, ax=[ax], location='right')

            def update(frame):
                start = time.time()
                sim.step()
                end = time.time()
                ax.set_xlabel(f"epoch/time: {sim.epoch}/{sim.time}, {end - start}")
                img.set_data(np.asarray(sim.get_world()))
                return img    
            
            ani = animation.FuncAnimation(fig=fig, func=update, frames=99, interval=40, repeat=False)
            plt.show()
        else:
            for i in range(100):
                sim.step()
        
        sim.evolve()

if __name__ == "__main__":
    main()