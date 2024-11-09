import random
from .pyg_mol_utils import *
from .batch import *

__all__ = ["removeHs", "generate_random_color", "generate_random_color_list"]


def generate_random_color():
    r = random.randint(0, 255) / 255
    g = random.randint(0, 255) / 255
    b = random.randint(0, 255) / 255
    return r, g, b


def generate_random_color_list(num_colors):
    return [generate_random_color() for _ in range(num_colors)]
