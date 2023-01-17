import argparse
import logging
import sys
import json

import torch
from pyprojroot import here as project_root
import wandb
from fs_mol.utils.protonet.encoder_model_selector import make_proto_encoder_model

from fs_mol.utils.wandb import make_trainer_config

sys.path.insert(0, str(project_root()))

from fs_mol.modules.graph_feature_extractor import (
    add_graph_feature_extractor_arguments,
    make_graph_feature_extractor_config_from_args,
)
from fs_mol.utils.cli_utils import add_train_cli_args, set_up_train_run
from fs_mol.utils.protonet_utils import (
    PrototypicalNetworkTrainerConfig,
    PrototypicalNetworkTrainer,
)
from fs_mol.models.contrastive_protonet import PrototypicalNetworkConfig
from fs_mol.models.protonet import AttentionBasedEncoder, AttentionBasedEncoderConfig, VanillaFSMolEncoder
# from fs_mol.models.dumb_graph_attention import DumbGraphAttention

from pathlib import Path


logger = logging.getLogger(__name__)


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Train a Prototypical Network model on molecules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_train_cli_args(parser)

    parser.add_argument(
        "--features",
        type=str,
        choices=[
            "gnn",
            "ecfp",
            "pc-descs",
            "ecfp+fc",
            "pc-descs+fc",
            "gnn+ecfp+fc",
            "gnn+ecfp+pc-descs+fc",
        ],
        default="gnn+ecfp+fc",
        help="Choice of features to use",
    )
    parser.add_argument(
        "--distance_metric",
        type=str,
        choices=["mahalanobis", "euclidean"],
        default="mahalanobis",
        help="Choice of distance to use.",
    )
    add_graph_feature_extractor_arguments(parser)
    
    parser.add_argument(
        "--encoder-type",
        type=str,
        choices=['vanilla', 'bidirectional-attention'],
        default='vanilla',
        help='Choose one of several models as the encoder'
    )

    parser.add_argument("--support_set_size", type=int, default=64, help="Size of support set")
    parser.add_argument(
        "--query_set_size",
        type=int,
        default=256,
        help="Size of target set. If -1, use everything but train examples.",
    )
    parser.add_argument(
        "--tasks_per_batch",
        type=int,
        default=16,
        help="Number of tasks to accumulate gradients for.",
    )

    parser.add_argument("--batch_size", type=int, default=256, help="Number of examples per batch.")
    parser.add_argument(
        "--num_train_steps", type=int, default=10000, help="Number of training steps."
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=50,
        help="Number of training steps between model validations.",
    )
    parser.add_argument(
        "--validation-support-set-sizes",
        type=json.loads,
        default=[16, 128],
        help="JSON list selecting the number of datapoints sampled as support set data during evaluation through finetuning on the validation tasks.",
    )

    parser.add_argument(
        "--validation-query-set-size",
        type=int,
        default=512,
        help="Maximum number of datapoints sampled as query data during evaluation through finetuning on the validation tasks.",
    )

    parser.add_argument(
        "--validation-num-samples",
        type=int,
        default=5,
        help="Number of samples considered for each train set size for each validation task during evaluation through finetuning.",
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument(
        "--clip_value", type=float, default=1.0, help="Gradient norm clipping value"
    )
    parser.add_argument(
        "--pretrained_gnn",
        type=str,
        default=None,
        help="Path to a pretrained GNN model to use as a starting point.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a check point to continue from"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_command_line()
    
    config = make_trainer_config(args)
    
    run = wandb.init(config=config, resume=False)

    out_dir, dataset, aml_run = set_up_train_run(
        f"ProtoNet_{config.used_features}", args, torch=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder_model = make_proto_encoder_model(args)
    model_trainer = PrototypicalNetworkTrainer(config=config, encoder_model=encoder_model).to(device)
    
    if args.checkpoint:
        checkpoint_folder = Path(args.checkpoint)
        
        model_trainer.load_state_dict(torch.load(checkpoint_folder / 'latest_model.pt'))
        model_trainer.optimizer.load_state_dict(torch.load(checkpoint_folder / 'optimizer_state.pt'))
    

    logger.info(f"\tDevice: {device}")
    logger.info(f"\tNum parameters {sum(p.numel() for p in model_trainer.parameters())}")
    logger.info(f"\tModel:\n{model_trainer}")

    if args.pretrained_gnn is not None:
        logger.info(f"Loading pretrained GNN weights from {args.pretrained_gnn}.")
        model_trainer.load_model_gnn_weights(path=args.pretrained_gnn, device=device)

    model_trainer.train_loop(out_dir, dataset, device, aml_run)
    run.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        import pdb

        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
