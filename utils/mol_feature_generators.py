import numpy as np
from rdkit import DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator


def generate_mol_fingerprint(mol):
    fingerprint_vec = rdFingerprintGenerator.GetCountFPs(
        [mol], fpType=rdFingerprintGenerator.MorganFP
    )[0]

    fingerprint_vec
    fp_numpy = np.zeros((0,), np.int8)  # Generate target pointer to fill
    DataStructs.ConvertToNumpyArray(fingerprint_vec, fp_numpy)
    return fp_numpy


def generate_mol_descriptors(mol):
    result = []
    for descr in Descriptors._descList:
        _, descr_calc_fn = descr
        result.append(descr_calc_fn(mol))

    return result
