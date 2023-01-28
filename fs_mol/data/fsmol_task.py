from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from more_itertools import partition
from dpu_utils.utils import RichPath
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from torch_geometric.data import Data
from ogb.utils.mol import smiles2graph
from fs_mol.custom.graphormer_utils import preprocess_item
import torch
from torch_geometric.data import Data

from fs_mol.custom.utils import convert_to_pyg_graph
from .s_attn import add_subgraph_info

def get_task_name_from_path(path: RichPath) -> str:
    # Use filename as task name:
    name = path.basename()
    if name.endswith(".jsonl.gz"):
        name = name[: -len(".jsonl.gz")]
    return name


@dataclass
class GraphData:
    """Data structure holding information about a graph with typed edges.

    Args:
        node_features: Initial node features as ndarray of shape [V, ...]
        adjacency_lists: Adjacency information by edge type as list of ndarrays of shape [E, 2]
        edge_features: Edge features by edge type as list of ndarrays of shape [E, edge_feat_dim].
            If not present, all edge_feat_dim=0.
    """

    node_features: np.ndarray
    adjacency_lists: List[np.ndarray]
    edge_features: List[np.ndarray]

@dataclass(frozen=True)
class PyG_MoleculeDatapoint:
    task_name: str
    graph: Data # This includes the label as well in the `y` property
    
# @dataclass(frozen=True)
# class GraphormerMoleculeDatapoint:
#     task_name: str
#     smiles: str
#     graph: GraphormerGraph
#     numeric_label: float
#     bool_label: bool
#     fingerprint: Optional[np.ndarray]
#     descriptors: Optional[np.ndarray]

@dataclass(frozen=True)
class MoleculeDatapoint:
    """Data structure holding information for a single molecule.

    Args:
        task_name: String describing the task this datapoint is taken from.
        smiles: SMILES string describing the molecule this datapoint corresponds to.
        graph: GraphData object containing information about the molecule in graph representation
            form, according to featurization chosen in preprocessing.
        numeric_label: numerical label (e.g., activity), usually measured in the lab
        bool_label: bool classification label, usually derived from the numeric label using a
            threshold.
        fingerprint: optional ECFP (Extended-Connectivity Fingerprint) for the molecule.
        descriptors: optional phys-chem descriptors for the molecule.
    """

    task_name: str
    smiles: str
    graph: GraphData
    numeric_label: float
    bool_label: bool
    fingerprint: Optional[np.ndarray]
    descriptors: Optional[np.ndarray]

    def get_fingerprint(self) -> np.ndarray:
        if self.fingerprint is not None:
            return self.fingerprint
        else:
            # TODO(mabrocks): It would be much faster if these would be computed in preprocessing and just passed through
            mol = Chem.MolFromSmiles(self.smiles)
            fingerprints_vect = rdFingerprintGenerator.GetCountFPs(
                [mol], fpType=rdFingerprintGenerator.MorganFP
            )[0]
            fingerprint = np.zeros((0,), np.float32)  # Generate target pointer to fill
            DataStructs.ConvertToNumpyArray(fingerprints_vect, fingerprint)
            return fingerprint

    def get_descriptors(self) -> np.ndarray:
        if self.descriptors is not None:
            return self.descriptors
        else:
            # TODO(mabrocks): It would be much faster if these would be computed in preprocessing and just passed through
            mol = Chem.MolFromSmiles(self.smiles)
            descriptors = []
            for _, descr_calc_fn in Descriptors._descList:
                descriptors.append(descr_calc_fn(mol))
            return np.array(descriptors)

# @dataclass(frozen=True)
# class GraphormerTask:
#     name: str
#     samples: List[GraphormerMoleculeDatapoint]
    
#     def get_pos_neg_separated(self) -> Tuple[List[GraphormerMoleculeDatapoint], List[GraphormerMoleculeDatapoint]]:
#         pos_samples, neg_samples = partition(pred=lambda s: s.bool_label, iterable=self.samples)
#         return list(pos_samples), list(neg_samples)

#     @staticmethod
#     def load_from_file(path: RichPath) -> "GraphormerTask":
#         samples = []
#         for raw_sample in path.read_by_file_suffix():
#             graph_data = raw_sample.get("graph")
#             smiles = raw_sample["SMILES"]

#             fingerprint_raw = raw_sample.get("fingerprints")
#             if fingerprint_raw is not None:
#                 fingerprint: Optional[np.ndarray] = np.array(fingerprint_raw, dtype=np.int32)
#             else:
#                 fingerprint = None

#             descriptors_raw = raw_sample.get("descriptors")
#             if descriptors_raw is not None:
#                 descriptors: Optional[np.ndarray] = np.array(descriptors_raw, dtype=np.float32)
#             else:
#                 descriptors = None
                
#             raw_graph = smiles2graph(smiles)
            
#             graphormer_graph = preprocess_item(raw_graph)

#             samples.append(
#                 MoleculeDatapoint(
#                     task_name=get_task_name_from_path(path),
#                     smiles=raw_sample["SMILES"],
#                     bool_label=bool(float(raw_sample["Property"])),
#                     numeric_label=float(raw_sample.get("RegressionProperty") or "nan"),
#                     fingerprint=fingerprint,
#                     descriptors=descriptors,
#                     graph=graphormer_graph,
#                 )
#             )

def convert_adjacency_list_to_edge_feats(adjacency_lists):
    single_bonds = adjacency_lists[0]
    double_bonds = adjacency_lists[1]
    triple_bonds =  adjacency_lists[2]

    return [0 for bond in single_bonds] + [1 for bond in double_bonds] + [2 for bond in triple_bonds]

def generate_adjacency_lists(graph_data_adjLists):
    adjacency_lists = []
    for adj_list in graph_data_adjLists:
        if len(adj_list) > 0:
            adjacency_lists.append(np.array(adj_list, dtype=np.int64))
        else:
            adjacency_lists.append(np.zeros(shape=(0, 2), dtype=np.int64))
    return adjacency_lists

def legacy_graph_parser(graph_data):
    adjacency_lists = generate_adjacency_lists(graph_data["adjacency_lists"])
            
    return GraphData(
                    node_features=np.array(graph_data["node_features"], dtype=np.float32),
                    adjacency_lists=adjacency_lists,
                    edge_features=[
                        np.array(edge_feats, dtype=np.float32)
                        for edge_feats in graph_data.get("edge_features") or []
                    ],
                )
    
    
def pyg_graph_parser(graph_data):
    # adjacency_lists = generate_adjacency_lists(graph_data["adjacency_lists"])
    
    # x = np.array(graph_data["node_features"], dtype=np.float32)
    
    # edge_attr = convert_adjacency_list_to_edge_feats(adjacency_lists)
    
    # edge_index = torch.cat(list(map(torch.tensor, adjacency_lists)), dim=0).t().contiguous()
    
    # return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return convert_to_pyg_graph(legacy_graph_parser(graph_data))


def parse_graph(graph_data, output_type = 'legacy'):
    if output_type == 'legacy':
        return legacy_graph_parser(graph_data)
    elif output_type == 'pyg':
        return pyg_graph_parser(graph_data)
    elif output_type == 's_attn':
        pyg_graph = pyg_graph_parser(graph_data)
        return add_subgraph_info(pyg_graph, 4)
    

@dataclass(frozen=True)
class FSMolTask:
    """Data structure holding information for a single task.

    Args:
        name: String describing the task's name eg. "CHEMBL1000114".
        samples: List of MoleculeDatapoint samples associated with this task.
    """

    name: str
    samples: List[MoleculeDatapoint]

    def get_pos_neg_separated(self) -> Tuple[List[MoleculeDatapoint], List[MoleculeDatapoint]]:
        pos_samples, neg_samples = partition(pred=lambda s: s.bool_label, iterable=self.samples)
        return list(pos_samples), list(neg_samples)

    @staticmethod
    def load_from_file(path: RichPath, output_type = 'legacy') -> "FSMolTask":
        samples = []
        for raw_sample in path.read_by_file_suffix():
            fingerprint_raw = raw_sample.get("fingerprints")
            if fingerprint_raw is not None:
                fingerprint: Optional[np.ndarray] = np.array(fingerprint_raw, dtype=np.int32)
            else:
                fingerprint = None

            descriptors_raw = raw_sample.get("descriptors")
            if descriptors_raw is not None:
                descriptors: Optional[np.ndarray] = np.array(descriptors_raw, dtype=np.float32)
            else:
                descriptors = None
            
            graph_data = raw_sample.get("graph")
            samples.append(
                MoleculeDatapoint(
                    task_name=get_task_name_from_path(path),
                    smiles=raw_sample["SMILES"],
                    bool_label=bool(float(raw_sample["Property"])),
                    numeric_label=float(raw_sample.get("RegressionProperty") or "nan"),
                    fingerprint=fingerprint,
                    descriptors=descriptors,
                    graph=parse_graph(graph_data, output_type),
                )
            )

        return FSMolTask(get_task_name_from_path(path), samples)


@dataclass(frozen=True)
class FSMolTaskSample:
    """Data structure output of a Task Sampler.

    Args:
        name: String describing the task's name eg. "CHEMBL1000114".
        train_samples: List of MoleculeDatapoint samples drawn as the support set.
        valid_samples: List of MoleculeDatapoint samples drawn as the validation set.
            This may be empty, dependent on the nature of the Task Sampler.
        test_samples: List of MoleculeDatapoint samples drawn as the query set.
    """

    name: str
    train_samples: List[MoleculeDatapoint]
    valid_samples: List[MoleculeDatapoint]
    test_samples: List[MoleculeDatapoint]

    @staticmethod
    def __compute_positive_fraction(samples: List[MoleculeDatapoint]) -> float:
        num_pos_samples = sum(s.bool_label for s in samples)
        return num_pos_samples / len(samples)

    @property
    def train_pos_label_ratio(self) -> float:
        return self.__compute_positive_fraction(self.train_samples)

    @property
    def test_pos_label_ratio(self) -> float:
        return self.__compute_positive_fraction(self.test_samples)
