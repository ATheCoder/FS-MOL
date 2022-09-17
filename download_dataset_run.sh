#! /bin/bash

# Download the dataset tar file
!curl -L https://figshare.com/ndownloader/files/31345321 --output datasets/fs-mol.tar

# Extract the dataset
!tar -xvf datasets/fs-mol.tar -C datasets

# Run Training
python fs_mol/protonet_train.py ./datasets/fs-mol/