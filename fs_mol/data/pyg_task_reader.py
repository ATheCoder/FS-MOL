import sys

from pyprojroot import here as project_root
sys.path.insert(0, str(project_root()))

from typing import List
from dpu_utils.utils.richpath import RichPath

from fs_mol.data.pygmol_task import PyGMolTask

def pyg_task_reader_fn(paths: List[RichPath], idx: int):
# We need to pass a task_reader_fn to eval_model as well.
    # Here we need to convert a path into a Task and then separate the graphs out.
    if len(paths) > 1:
        raise ValueError()
    
    task = PyGMolTask.load_from_file(paths[0])
    
    return [task]