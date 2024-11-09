class PretrainGatedGraphConfig:
    project_name = "Prtrain usin 5 Million"
    dim: int = 256
    n_layers: int = 4
    aggregator_heads: int = 4
    aggregator_encoder_blocks: int = 1
    aggregator_decoder_blocks: int = 1
    accumulate_grad_batches: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0
    similarity_module: str = 'batch_protonet'
    prediction_scaling = 1.0
    batch_size: int = 128
    
    attn_dim: int = 256