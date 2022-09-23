#! /bin/bash

# Make sure that `./saved_weights/fully_trained.pt` and  `./datasets/fs-mol/valid` also exists.
python fs_mol/protonet_test.py ./saved_weights/fully_trained.pt ./datasets/fs-mol/

