try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    HAS_TE = True
except ImportError:
    HAS_TE = False

class FP8VideoTransformer:
    """
    FP8 quantization for 50% memory reduction
    Compatible with Ada Lovelace (SM89) and Hopper (SM90)
    """
    def __init__(self, layer, fp8_format="E4M3"):
        if not HAS_TE:
            raise ImportError("transformer-engine required for FP8")
            
        self.fp8_recipe = recipe.DelayedScaling(
            fp8_format=recipe.Format.E4M3 if fp8_format == "E4M3" else recipe.Format.E5M2,
            amax_history_len=1024,
            amax_compute_algo="max"
        )
        self.layer = te.pytorch.Linear(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None
        )
        # Copy weights
        with torch.no_grad():
            self.layer.weight.copy_(layer.weight)
            if layer.bias is not None:
                self.layer.bias.copy_(layer.bias)
                
    def forward(self, x):
        with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
            return self.layer(x)
