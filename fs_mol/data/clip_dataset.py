import torch
from torch_geometric.data import InMemoryDataset

class CLIPDataset(InMemoryDataset):
    def __init__(self, root, raw_file_path, dest_file_name, transform=None, pre_transform=None, pre_filter=None):
        self.dest = dest_file_name
        self.raw_file_path = raw_file_path
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0], map_location=torch.device('cuda'))
        
    @property
    def raw_file_names(self):
        return [self.raw_file_path]
    
    @property
    def processed_file_names(self):
        return [self.dest]
    
    def process(self):
        data_list = torch.load(self.raw_file_names[0])
        
        print(data_list)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])