## Install the dependencies

`pip install torch_geometric lightning dpu_utils more_itertools rdkit py-repo-root omegaconf torch_sparse torch_scatter wandb torch_cluster nglview dataclass_flex dill`

`apt update && apt install libxrender1 libxext6`

## Setup Environment PATH
Make a `.env` file inside the root directory of this repo and add absolute path to this repo's folder:

```
PYTHONPATH=/FS-MOL # Absolute path to this repo's folder
```



## Notebooks

There are several notebooks in the `notebooks` directory:

1. The [Molecular Features Notebook](notebooks/molecular_features.ipynb) can be used to view different molecular features and how they are generated.