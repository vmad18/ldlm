
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from tqdm.auto import tqdm
from functools import partial

from transformers.trainer_pt_utils import AcceleratorConfig

from diffusion.neural_diffusion import DiTModel


def exists(x):
    """Checks if a value is not None."""
    return x is not None

def default(val, d):
    """Returns the value if it exists, otherwise returns a default."""
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """Extracts values from 'a' at indices 't' and reshapes them to match 'x_shape'."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def right_pad_dims_to(x, t):
    """Pads the dimensions of 't' to match the number of dimensions of 'x'."""
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    """Generates a cosine noise schedule."""
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min=clip_min)






