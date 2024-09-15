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
    world_size = 30
    num_orgs = 100

    sim = Simulation(width=world_size, height=world_size, num_organisms=num_orgs)
    t = 0
    
    # os.system('cls')
    # print(f"time: {t}")
    # sim.print_world()
    # time.sleep(1)
    
    # while True:
    #     sim.step()
    #     t += 1

    #     _ = input()
    #     os.system('cls')
    #     print(f"time: {t}")
    #     sim.print_world()
    
    fig, ax = plt.subplots()
    img = ax.matshow(np.asarray(sim.get_world()))
    cb = plt.colorbar(img, ax=[ax], location='right')

    def update(frame):
        start = time.time()
        sim.step()
        end = time.time()
        ax.set_xlabel(f"time: {frame}, {end - start}")
        img.set_data(np.asarray(sim.get_world()))
        return img    
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=400, interval=300, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()