import argparse
from fs_mol.models.protonet import AttentionBasedEncoder, AttentionBasedEncoderConfig, PrototypicalNetworkConfig, VanillaFSMolEncoder
from fs_mol.modules.graph_feature_extractor import make_graph_feature_extractor_config_from_args


model_name_to_implementation_map = {
    'vanilla': VanillaFSMolEncoder,
    'bidirectional-attention': AttentionBasedEncoder
}

def vanilla_config_generator(args: argparse.Namespace):
    return PrototypicalNetworkConfig(
        graph_feature_extractor_config=make_graph_feature_extractor_config_from_args(args),
        used_features=args.features,
        distance_metric=args.distance_metric,
    )
    
def bidirectional_encoder_config_generator(args: argparse.Namespace):
    return AttentionBasedEncoderConfig()

model_name_to_config_map = {
    'vanilla': vanilla_config_generator,
    'bidirectional-attention': bidirectional_encoder_config_generator
}

def make_proto_encoder_model(args: argparse.Namespace):
    model_name = args.encoder_type
    model_config_generator = model_name_to_config_map[model_name]
    
    return model_name_to_implementation_map[model_name](model_config_generator(args))
