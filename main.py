import torch
from torch import nn
import time

from organism import Organism

def main():
    id = 0
    input_dim = 8
    hidden_dim = 12
    output_dim = 9
    device = "cpu"
    world_size = 30

    org = Organism(id, input_dim, hidden_dim, output_dim, device, world_size, world_size)
    m1, m2 = org.get_weights()

    print(m1.shape)
    print(m2.shape)

    for i in range(8):
        input = [0] * 8
        input[i] = 1

        input = torch.tensor([input], dtype=torch.float32)

        start = time.time()

        action = org.get_action(input)
        end = time.time()
        print(end - start)
        print(f"{i}: {input}, {action}")

if __name__ == "__main__":
    main()