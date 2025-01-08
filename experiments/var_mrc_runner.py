import sys,os,inspect
currentdir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir=os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from modules.graph_encoders.GINEncoder import GINEncoder,GINEncoderConfig
from modules.similarity_modules import CNAPSProtoNetSimilarityModule
from mrc_src.encoders.fingerprint_encoder import FingerprintSimpleFeedForward
from fs_mol.utils.metrics import compute_binary_task_metrics
from dataclasses import dataclass
import torch
from fs_mol.data.torch_dl import MRCDataset,MRCDataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torchmetrics as tm
from torchmetrics.utilities import dim_zero_cat
from lightning.pytorch.loggers import WandbLogger
import wandb

@dataclass(frozen=True)
class TrainConfig:
    representation:str="fingerprint"
    batch_size:int=32
    train_support_count:int=8
    train_query_count:int=16
    train_shuffle:bool=True
    beta:float=0.124701
    valid_support_count:int=16
    valid_batch_size:int=1
    envelope_exponent:int=6
    num_spherical:int=7
    num_radial:int=5
    dim:int=128
    cutoff:int=5.0
    layer:int=7
    accumulate_grad_batches:int=1
    learning_rate:float=0.00003
    weight_decay:float=0.0
    dropout:float=0.0
    encoder_dims=[128,128,256,256,512,512]
    train_n_repeats=5
    val_n_repeats=5
    dataloader_workers:int=8
    preload_dataset=True
    isProd=True

config=TrainConfig()
REPR_TO_ENCODER_MAP={"2d":GINEncoder,"fingerprint":FingerprintSimpleFeedForward}
REPR_ENCODER_CONFIG_MAP={"2d":GINEncoderConfig(config.dim,config.layer,config.cutoff,dropout=config.dropout),"fingerprint":config}
train_dataset=MRCDataset("train","pyg",config.train_n_repeats,config.preload_dataset,not config.isProd)
valid_dataset=MRCDataset("valid","pyg",config.val_n_repeats,config.preload_dataset,not config.isProd)
train_dl=MRCDataLoader(config.representation,train_dataset,batch_size=config.batch_size,datatype="pyg",num_workers=config.dataloader_workers,shuffle=config.train_shuffle,support_count=config.train_support_count,query_count=config.train_query_count)
valid_dls=MRCDataLoader(config.representation,valid_dataset,batch_size=config.valid_batch_size,datatype="pyg",num_workers=config.dataloader_workers,support_count=16,query_count=16)

class StandardDeviationMetric(tm.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("values",default=[],dist_reduce_fx="cat")
    def update(self,value:torch.Tensor):
        self.values.append(value)
    def compute(self):
        vals=dim_zero_cat(self.values)
        return torch.std(vals,correction=0)

metrics_table_cols=['task_name','size','acc','balanced_acc','f1','prec','recall','roc_auc','avg_precision','kappa','delta_auc_pr','optimistic_auc_pr','optimistic_delta_auc_pr']

class MRCLightningModule(L.LightningModule):
    def __init__(self,config,*args,**kwargs):
        self.config=kwargs.pop("config",config)
        super().__init__(*args,**kwargs)
        self.encoder=REPR_TO_ENCODER_MAP[self.config.representation](REPR_ENCODER_CONFIG_MAP[self.config.representation])
        self.std_dev_metric=StandardDeviationMetric()
        self.similarity_module=CNAPSProtoNetSimilarityModule(self.config.beta,True)
        self.wandb_table=wandb.Table(columns=metrics_table_cols)
    def calculate_feats(self,batch):
        encoded_graphs=self.encoder(batch)
        return encoded_graphs
    def calc_loss(self,input):
        batch,labels,is_query,batch_index,task_names=input
        feats=self.encoder(batch)
        logits,batch_labels=self.similarity_module(feats,labels,is_query,batch_index)
        loss=self.similarity_module.calc_loss_from_logits(logits,batch_labels)
        return loss,logits,labels[is_query==1],task_names
    def training_step(self,batches):
        loss,_,_,_=self.calc_loss(batches)
        self.log('train_step',loss,on_step=True,on_epoch=False,batch_size=config.batch_size)
        self.log('train_loss',loss,on_step=False,on_epoch=True,batch_size=config.batch_size)
        return loss
    def on_validation_epoch_start(self):
        self.wandb_table=wandb.Table(columns=metrics_table_cols)
    def on_validation_epoch_end(self):
        epoch_std_dev=self.std_dev_metric.compute()
        self.log('delta_auc_pr_std',epoch_std_dev,on_epoch=True)
        if wandb.run is not None:
            wandb.log({'metrics_table':self.wandb_table})
    def validation_step(self,batches):
        valid_loss,logits,query_labels,task_names=self.calc_loss(batches)
        self.log("valid_loss",valid_loss,on_step=False,on_epoch=True,batch_size=1)
        batch_preds=self.similarity_module.get_probabilities_from_logits(logits)
        metrics=compute_binary_task_metrics(predictions=batch_preds.cpu(),labels=query_labels.detach().cpu().numpy())
        self.wandb_table.add_data(task_names[0].split('_')[0],*metrics.__dict__.values())
        self.std_dev_metric(torch.tensor(metrics.delta_auc_pr.item()))
        for k,v in metrics.__dict__.items():
            self.log(f"valid_{k}",v,on_epoch=True,on_step=False,batch_size=1)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.config.learning_rate,weight_decay=config.weight_decay,fused=True)

# Loop of 10 training runs:
for i in range(10):
    checkpoint_callback=ModelCheckpoint(dirpath=f'MRC_Runner_{i}',filename='best-checkpoint',save_top_k=2,verbose=True,monitor='valid_optimistic_delta_auc_pr',mode='max',save_last=True)
    model=MRCLightningModule(config=config)
    if config.isProd:
        wandb.init(project=f"{config.representation}_molecular_representation_comparison_final",config=config.__dict__,name=f"run_{i}")
        wandb.watch(model,log='all')
    trainer=L.Trainer(callbacks=[checkpoint_callback],check_val_every_n_epoch=1,logger=WandbLogger() if config.isProd else None,max_epochs=100,accumulate_grad_batches=config.accumulate_grad_batches)
    trainer.fit(model=model,train_dataloaders=train_dl,val_dataloaders=valid_dls)
    if config.isProd:
        wandb.finish()



# Test Set statistics: 0.218, 0.224, 0.225, 0.222, 0.233

# Mean and STD: 0.224 0.005