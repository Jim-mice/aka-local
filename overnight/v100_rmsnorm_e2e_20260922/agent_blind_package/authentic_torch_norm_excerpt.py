# Exact relevant excerpt from pinned Megatron commit
# 5be9626709af2722333bf54797c954c09edeada3.
# Source: megatron/core/transformer/torch_norm.py

class WrappedTorchNorm:
    def __new__(
        cls,
        config,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",
    ):
        assert not config.layernorm_zero_centered_gamma
        assert not config.persist_layer_norm
        assert not config.sequence_parallel
        assert not config.memory_efficient_layer_norm
        if config.normalization == "LayerNorm":
            norm_cls = torch.nn.LayerNorm
        elif config.normalization == "RMSNorm":
            norm_cls = torch.nn.RMSNorm
        elif config.normalization == "L2Norm":
            norm_cls = torch.nn.L2Norm
        else:
            raise Exception("Only LayerNorm, RMSNorm and L2Norm are currently supported")
        return norm_cls(normalized_shape=hidden_size, eps=eps)
