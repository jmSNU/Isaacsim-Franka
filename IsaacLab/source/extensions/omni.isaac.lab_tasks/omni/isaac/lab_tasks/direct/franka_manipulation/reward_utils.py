import torch
import numpy as  np

def tolerance(x, bounds, margin, value_at_margin=0.1):
    lower, upper = bounds
    assert lower < upper and torch.all(margin > 0)
    
    # Check if x is within bounds
    in_bounds = torch.logical_and(lower <= x, x <= upper)
    
    if torch.all(margin == 0):
        value = torch.where(in_bounds, 1.0, 0.0)
    else:
        # Compute distance d and apply sigmoid function
        d = torch.where(x < lower, lower - x, x - upper) / margin
        sigmoid = lambda x, value_at_1: 1 / ((x * np.sqrt(1 / value_at_1 - 1))**2 + 1)
        value = torch.where(in_bounds, 1.0, sigmoid(d, value_at_margin))

    return value

def hamacher_product(a,b):
    denom = a+b-(a*b)
    h_prod = ((a*b)/denom) if denom>0 else 0.0
    return h_prod
