from dataclasses import dataclass, field


@dataclass(frozen=True)
class MXMNetTrainingConfig:
    project_name: str = "MXMNet_New"
    # Training Settings:
    batch_size: int = 64
    train_support_count: int = 16
    train_query_count: int = 1
    train_shuffle: bool = True

    mol_aggr: str = "last"
    # mol_aggr_config = dict(dim=128)
    mol_aggr_config: dict = field(default_factory=dict)
    # mol_aggr_config = dict(in_dim=128, out_dim=1024, n_layers=8)

    temprature: float = 0.07

    # Validation Settings:
    valid_support_count: int = 64
    valid_batch_size: int = 256

    # Model Settings:
    envelope_exponent: int = 6
    num_spherical: int = 7
    num_radial: int = 5
    dim: int = 64
    cutoff: int = 5.0
    layer: int = 6

    accumulate_grad_batches: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 1e-3
    padding_size = 12
    prediction_scaling = 0.325

    dropout: float = 0.0

    encoder_dims = [128, 128, 256, 256, 512, 512]
