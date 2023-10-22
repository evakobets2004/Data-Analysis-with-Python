import numpy as np
import torch
from torch import nn

def create_model():
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 16),
        nn.ReLU(),
        nn.Linear(16, 10)
    )

    return model
model = create_model()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

small_model = nn.Linear(128, 256)
print(count_parameters(small_model))

medium_model = nn.Sequential(*[nn.Linear(128, 32, bias=False), nn.ReLU(), nn.Linear(32, 10, bias=False)])
print(count_parameters(medium_model))

    
    
