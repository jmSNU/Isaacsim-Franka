import torch
import numpy as np

def _sigmoid(x, value_at_1, sigmoid):
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                f"`value_at_1` must be nonnegative and smaller than 1, got {value_at_1}."
            )
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(
                f"`value_at_1` must be strictly between 0 and 1, got {value_at_1}."
            )

    if sigmoid == "gaussian":
        scale = torch.sqrt(-2 * torch.log(torch.tensor(value_at_1, dtype=x.dtype)))
        return torch.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == "hyperbolic":
        scale = torch.arccosh(1 / torch.tensor(value_at_1, dtype=x.dtype))
        return 1 / torch.cosh(x * scale)

    elif sigmoid == "long_tail":
        scale = torch.sqrt(1 / torch.tensor(value_at_1, dtype=x.dtype) - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == "reciprocal":
        scale = 1 / torch.tensor(value_at_1, dtype=x.dtype) - 1
        return 1 / (torch.abs(x) * scale + 1)

    elif sigmoid == "cosine":
        scale = torch.arccos(2 * torch.tensor(value_at_1, dtype=x.dtype) - 1) / np.pi
        scaled_x = x * scale
        ret = torch.where(
            torch.abs(scaled_x) < 1, (1 + torch.cos(np.pi * scaled_x)) / 2, 0.0
        )
        return ret.item() if isinstance(x, torch.Tensor) and x.numel() == 1 else ret

    elif sigmoid == "linear":
        scale = 1 - torch.tensor(value_at_1, dtype=x.dtype)
        scaled_x = x * scale
        ret = torch.where(torch.abs(scaled_x) < 1, 1 - scaled_x, 0.0)
        return ret.item() if isinstance(x, torch.Tensor) and x.numel() == 1 else ret

    elif sigmoid == "quadratic":
        scale = torch.sqrt(1 - torch.tensor(value_at_1, dtype=x.dtype))
        scaled_x = x * scale
        ret = torch.where(torch.abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)
        return ret.item() if isinstance(x, torch.Tensor) and x.numel() == 1 else ret

    elif sigmoid == "tanh_squared":
        scale = torch.arctanh(torch.sqrt(1 - torch.tensor(value_at_1, dtype=x.dtype)))
        return 1 - torch.tanh(x * scale) ** 2

    else:
        raise ValueError(f"Unknown sigmoid type {sigmoid!r}.")


def tolerance(x, bounds, margin, sigmoid, value_at_margin=0.1):
    lower, upper = bounds
    if lower > upper and torch.all(margin < 0) :
        message = f"""lower value{lower} must be smaller than {upper}""" if lower>upper else """margin should be larger than zero"""
        raise Exception(message)
    
    # Check if x is within bounds
    in_bounds = torch.logical_and(lower <= x, x <= upper)
    
    if torch.all(margin == 0):
        value = torch.where(in_bounds, 1.0, 0.0)
    else:
        # Compute distance d and apply sigmoid function
        d = torch.where(x < lower, lower - x, x - upper) / margin
        value = torch.where(in_bounds, 1.0, _sigmoid(d, value_at_margin, sigmoid))

    return value

def hamacher_product(a,b):
    denom = a+b-(a*b)
    h_prod = torch.where(denom>0, (a*b)/denom, 0.0)
    return h_prod
