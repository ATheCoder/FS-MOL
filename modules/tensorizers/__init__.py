import torch
from fs_mol.data.fsmol_task import MergedFSMOLSample

###############################################################################
# Representation-specific tensorizers
###############################################################################
def threed_graph_tensorizer(sample: MergedFSMOLSample):
    # If 3D means PyG Data objects, just return sample.graph
    return sample.graph

def fingerprint_tensorizer(sample: MergedFSMOLSample):
    # Return sample.fingerprints as a torch tensor
    return torch.tensor(sample.fingerprints, dtype=torch.float32)

# Example of how you could add a new representation
def descriptors_tensorizer(sample: MergedFSMOLSample):
    # Suppose you compute a custom embedding here
    # e.g. sample.custom_repr -> a list of floats
    return torch.tensor(sample.descriptors, dtype=torch.float32)

def fingerprint_descriptors_tensorizer(sample: MergedFSMOLSample):
    fingerprints = torch.tensor(sample.fingerprints, dtype=torch.float32)

    descriptors = torch.tensor(sample.descriptors, dtype=torch.float32)


    return torch.cat([fingerprints, descriptors], dim=0)

REPR_TO_TENSORIZER_MAP = {
    '2d': threed_graph_tensorizer,
    'fingerprint': fingerprint_tensorizer,
    'descriptors': descriptors_tensorizer,
    'fingerprint+descriptors': fingerprint_descriptors_tensorizer,
    '2d_gated': threed_graph_tensorizer
}
