import inspect
from collections import deque
from typing import Any, Callable, Dict, List, Tuple

from dataclass_flex import create_dataclass_wrapper
from networkx import Graph
from torch_geometric.data import Data

from .converters import (
    convert_Data_to_NetworkXGraph,
    convert_Data_to_Smiles,
    convert_Smiles_to_Data,
)

# Define your conversion functions here
conversion_functions: Dict[Tuple[type, type], Callable[[Any], Any]] = {
    (Data, str): convert_Data_to_Smiles,
    (str, Data): convert_Smiles_to_Data,
    (Data, Graph): convert_Data_to_NetworkXGraph,
    # Do not define convert_A_to_C and convert_C_to_A, we will find a path through B
}

# Infer the dataclasses from the conversion_functions

graph_dynamic = create_dataclass_wrapper(conversion_functions)
