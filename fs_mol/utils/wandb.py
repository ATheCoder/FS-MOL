import argparse
import wandb
from pathlib import Path
from fs_mol.modules.graph_feature_extractor import make_graph_feature_extractor_config_from_args

from fs_mol.utils.protonet_utils import PrototypicalNetworkTrainerConfig
    
def make_trainer_config(args: argparse.Namespace) -> PrototypicalNetworkTrainerConfig:
    return PrototypicalNetworkTrainerConfig(
        graph_feature_extractor_config=make_graph_feature_extractor_config_from_args(args),
        used_features=args.features,
        distance_metric=args.distance_metric,
        batch_size=args.batch_size,
        tasks_per_batch=args.tasks_per_batch,
        support_set_size=args.support_set_size,
        query_set_size=args.query_set_size,
        validate_every_num_steps=args.validate_every,
        validation_support_set_sizes=tuple(args.validation_support_set_sizes),
        validation_query_set_size=args.validation_query_set_size,
        validation_num_samples=args.validation_num_samples,
        num_train_steps=args.num_train_steps,
        learning_rate=args.lr,
        clip_value=args.clip_value,
    )
    
def start_wandb_run(args, model):
    run = wandb.init(project="FS-MOL", config=args)