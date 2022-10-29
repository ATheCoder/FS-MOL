import torch
from torch_geometric.data import InMemoryDataset
import os.path as osp
from fs_mol.custom.utils import convert_to_pyg_graph
import torch
from urllib.request import urlretrieve
from gradient import DatasetVersionsClient

paperspace_api_key = '4f836db19dee5ff7e7c33b0f7b7343'

def get_dataset_url():
    dataset_version_id = 'dstmnmr5qs5jv1j:exdcgoy'
    
    dataset_client = DatasetVersionsClient(api_key=paperspace_api_key)
    
    return dataset_client.generate_pre_signed_s3_url(dataset_version_id=dataset_version_id, method='getObject', params={'Key': 'data.pt'}).url

class FSMolSelfSupervisedInMemory(InMemoryDataset):
    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None, device=None):
        self.root = root
        super().__init__(root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.device = device
        self.data, self.slices = torch.load(self.processed_paths[0], map_location=self.device)
    
    @property
    def raw_paths(self):
        return [osp.join(self.root, 'data.pt')]
    
    @property
    def processed_paths(self):
        return [osp.join(self.root, 'data.pt')]

    def download(self):
        print('Downloading ...')
        def report(blocknr, blocksize, size):
            current = blocknr*blocksize
            print("\r{0:.2f}%".format(100.0*current/size))
        file_name = get_dataset_url()
        urlretrieve(file_name, self.processed_paths[0], report)
    

# class FSMolSelfSupervisedDataset(Dataset):
#     def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, fs_mol_dir: str='./datasets/fs-mol/train', processed_dir: str = './preprocessed_dir'):
#         self.fs_mol_dir = fs_mol_dir
#         self.Aprocessed_dir = processed_dir
        
#         super().__init__(root, transform, pre_transform, pre_filter)
    
#     @property
#     def raw_paths(self) -> List[str]:
#         return os.listdir(self.fs_mol_dir)
    
#     @property
#     def processed_paths(self) -> List[str]:
#         return [*(osp.join(self.Aprocessed_dir, f'data_{k * 100}.pt') for k in range(50)), osp.join(self.Aprocessed_dir, 'data_4938.pt')]
    
#     def process(self):
#         task_list_path = './datasets/fsmol-0.1.json'


#         train_folder = RichPath.create('./datasets/fs-mol/train/')

#         task_list = RichPath.create(task_list_path).read_by_file_suffix()

#         raw_task_path = [
#             file_name
#             for file_name in train_folder.get_filtered_files_in_dir("*.jsonl.gz")
#             if any(
#                 file_name.basename() == f"{task_name}.jsonl.gz"
#                 for task_name in task_list['train']
#         )
#         ]
        
#         current_batch = []
        
#         counter = 0
        
#         for raw_path in raw_task_path:
#             task = FSMolTask.load_from_file(raw_path)
            
#             pyg_graphs = list(map(convert_to_pyg_graph, task.samples))
            
#             for graph in pyg_graphs:
#                 current_batch.append(graph)
#                 if counter % 100000 == 0:
#                     print(len(current_batch))
#                     torch.save(current_batch, osp.join('./preprocessed_dir', f'data_{counter}.pt'))
#                     current_batch = []
#                 counter = counter + 1
            
#             print(f'Finished processing "{task.name}"')
        
#         if len(current_batch) > 0:
#             torch.save(current_batch, osp.join('./preprocessed_dir', f'data_{counter}.pt'))
    
#     def get(self, idx: int) -> Data:
#         torch.load(current_batch, osp.join('./preprocessed_dir', f'data_{counter}.pt'))