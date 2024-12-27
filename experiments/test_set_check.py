import sys,os,inspect
currentdir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir=os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from modules.graph_encoders.GINEncoder import GINEncoder,GINEncoderConfig
from modules.similarity_modules import CNAPSProtoNetSimilarityModule
from mrc_src.encoders.fingerprint_encoder import FingerprintSimpleFeedForward
from fs_mol.utils.metrics import compute_binary_task_metrics
from fs_mol.data.torch_dl import MRCDataset,MRCDataLoader
import torch
import lightning as L
import torchmetrics as tm
from torchmetrics.utilities import dim_zero_cat
from lightning.pytorch.loggers import WandbLogger
from dataclasses import dataclass

@dataclass(frozen=True)
class TrainConfig:
    representation:str="3d"
    batch_size:int=32
    train_support_count:int=4
    train_query_count:int=16
    train_shuffle:bool=True
    beta:float=0.28107
    valid_support_count:int=64
    valid_batch_size:int=1
    envelope_exponent:int=6
    num_spherical:int=7
    num_radial:int=5
    dim:int=256
    cutoff:float=5.0
    layer:int=7
    accumulate_grad_batches:int=1
    learning_rate:float=0.00003
    weight_decay:float=0.0
    dropout:float=0.0
    encoder_dims=[128,128,256,256,512,512]
    train_n_repeats=5
    val_n_repeats=20
    dataloader_workers:int=4
    preload_dataset=True
    isProd=True

config=TrainConfig()

REPR_TO_ENCODER_MAP={"3d":GINEncoder,"fingerprint":FingerprintSimpleFeedForward}
REPR_ENCODER_CONFIG_MAP={
    "3d":GINEncoderConfig(config.dim,config.layer,config.cutoff,dropout=config.dropout),
    "fingerprint":config
}

class StandardDeviationMetric(tm.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("values",default=[],dist_reduce_fx="cat")
    def update(self,value:torch.Tensor):
        self.values.append(value)
    def compute(self):
        vals=dim_zero_cat(self.values)
        return torch.std(vals,correction=0)

class MRCLightningModule(L.LightningModule):
    def __init__(self,config,*args,**kwargs):
        self.config=kwargs.pop("config",config)
        super().__init__(*args,**kwargs)
        self.encoder=REPR_TO_ENCODER_MAP[self.config.representation](
            REPR_ENCODER_CONFIG_MAP[self.config.representation]
        )
        self.similarity_module=CNAPSProtoNetSimilarityModule(self.config.beta,True)
        self.std_dev_metric=StandardDeviationMetric()
        self.save_hyperparameters(ignore=["config"])  # optional

    def calc_loss(self,input_batch):
        batch,labels,is_query,batch_index,task_names=input_batch
        feats=self.encoder(batch)
        logits,batch_labels=self.similarity_module(feats,labels,is_query,batch_index)
        loss=self.similarity_module.calc_loss_from_logits(logits,batch_labels)
        return loss,logits,labels[is_query==1],task_names

    def training_step(self,batch):
        loss,_,_,_=self.calc_loss(batch)
        self.log("train_loss",loss,prog_bar=True,batch_size=self.config.batch_size)
        return loss

    def validation_step(self,batch,batch_idx):
        val_loss,logits,query_labels,task_names=self.calc_loss(batch)
        self.log("valid_loss",val_loss,prog_bar=True,batch_size=1)
        preds=self.similarity_module.get_probabilities_from_logits(logits).cpu()
        metrics=compute_binary_task_metrics(preds,query_labels.detach().cpu().numpy())
        self.std_dev_metric(torch.tensor(metrics.delta_auc_pr))
        for k,v in metrics.__dict__.items():
            self.log(f"valid_{k}",v,batch_size=1)

    # test_step: do the same as validation
    def test_step(self,batch,batch_idx):
        test_loss,logits,query_labels,task_names=self.calc_loss(batch)
        self.log("test_loss",test_loss,prog_bar=True,batch_size=1)
        preds=self.similarity_module.get_probabilities_from_logits(logits).cpu()
        metrics=compute_binary_task_metrics(preds,query_labels.detach().cpu().numpy())
        for k,v in metrics.__dict__.items():
            self.log(f"test_{k}",v,batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,fused=True
        )


# Create the test dataset and loader:
test_dataset=MRCDataset(
    "test",
    mol_type="pyg",
    n_repeats=config.val_n_repeats,
    should_preload=config.preload_dataset,
    debug=not config.isProd
)
test_dl=MRCDataLoader(
    config.representation,
    test_dataset,
    batch_size=config.valid_batch_size,
    datatype="pyg",
    num_workers=config.dataloader_workers,
    support_count=16,
    query_count=16
)

# Load the checkpointed model:
checkpoint_path="/FS-MOL/MRC_Runner/best-checkpoint-v30.ckpt"
model=MRCLightningModule.load_from_checkpoint(checkpoint_path,config=config)

# Run test:
trainer=L.Trainer(logger=WandbLogger() if config.isProd else None)
trainer.test(model,dataloaders=test_dl)
