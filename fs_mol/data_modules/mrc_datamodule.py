from lightning import LightningDataModule

from fs_mol.data.torch_dl import MRCDataset, MRCDataLoader


class MRCDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage):
        self.valid = MRCDataset("valid", "pyg", self.config['val_n_repeats'], self.config['preload_dataset'], not self.config['isProd'])
        if stage == "fit":
            self.train = MRCDataset(
            "train", "pyg", self.config['train_n_repeats'], self.config['preload_dataset'], not self.config['isProd']
        )


    def train_dataloader(self):
        return MRCDataLoader(
            self.config['representation'],
            self.train,
            batch_size=self.config['batch_size'],
            datatype="pyg",
            num_workers=self.config['dataloader_workers'],
            shuffle=self.config['train_shuffle'],
            support_count=self.config['train_support_count'],
            query_count=self.config['train_query_count'],
        )

    def val_dataloader(self):
        return MRCDataLoader(
            self.config['representation'],
            self.valid,
            batch_size=self.config['valid_batch_size'],
            datatype="pyg",
            num_workers=self.config['dataloader_workers'],
            support_count=16,
            query_count=16,
        )
