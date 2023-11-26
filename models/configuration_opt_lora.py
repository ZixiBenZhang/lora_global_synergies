from transformers import PretrainedConfig
from transformers.utils import logging
import toml
from lora.lora_config_parser import parse_lora_config


logger = logging.get_logger(__name__)

OPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/opt-125m": "https://huggingface.co/facebook/opt-125m/blob/main/config.json",
    "facebook/opt-350m": "https://huggingface.co/facebook/opt-350m/blob/main/config.json",
    "facebook/opt-1.3b": "https://huggingface.co/facebook/opt-1.3b/blob/main/config.json",
    "facebook/opt-2.7b": "https://huggingface.co/facebook/opt-2.7b/blob/main/config.json",
    "facebook/opt-6.7b": "https://huggingface.co/facebook/opt-6.7b/blob/main/config.json",
    "facebook/opt-13b": "https://huggingface.co/facebook/opt-13b/blob/main/config.json",
}


class OPTLoraConfig(PretrainedConfig):
    r"""
        This is the configuration class to store the configuration of a [`OPTModel`]. It is used to instantiate a OPT model
        according to the specified arguments, defining the model architecture. Instantiating a configuration with the
        defaults will yield a similar configuration to that of the OPT
        [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) architecture.

        Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
        documentation from [`PretrainedConfig`] for more information.


        Args:
            vocab_size (`int`, *optional*, defaults to 50272):
                Vocabulary size of the OPT model. Defines the number of different tokens that can be represented by the
                `inputs_ids` passed when calling [`OPTModel`]
            hidden_size (`int`, *optional*, defaults to 768):
                Dimensionality of the layers and the pooler layer.
            num_hidden_layers (`int`, *optional*, defaults to 12):
                Number of decoder layers.
            ffn_dim (`int`, *optional*, defaults to 3072):
                Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
            num_attention_heads (`int`, *optional*, defaults to 12):
                Number of attention heads for each attention layer in the Transformer decoder.
            activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
                The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
                `"relu"`, `"silu"` and `"gelu_new"` are supported.
            max_position_embeddings (`int`, *optional*, defaults to 2048):
                The maximum sequence length that this model might ever be used with. Typically set this to something large
                just in case (e.g., 512 or 1024 or 2048).
            do_layer_norm_before (`bool`, *optional*, defaults to `True`):
                Whether to perform layer normalization before the attention block.
            word_embed_proj_dim (`int`, *optional*):
                `word_embed_proj_dim` can be set to down-project word embeddings, *e.g.* `opt-350m`. Defaults to
                `hidden_size`.
            dropout (`float`, *optional*, defaults to 0.1):
                The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            attention_dropout (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the attention probabilities.
            layerdrop (`float`, *optional*, defaults to 0.0):
                The LayerDrop probability. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556) for more
                details.
            init_std (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should return the last key/values attentions (not used by all models).
            enable_bias (`bool`, *optional*, defaults to `True`):
                Whether or not if the linear layers in the attention blocks should use the bias term.
            layer_norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
                Whether or not if the layer norms should have learnable parameters.

        Example:

        ```python
        >>> from transformers import OPTConfig, OPTModel

        >>> # Initializing a OPT facebook/opt-large style configuration
        >>> configuration = OPTConfig()

        >>> # Initializing a model (with random weights) from the facebook/opt-large style configuration
        >>> model = OPTModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```"""
    model_type = "opt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50272,
        hidden_size=768,
        num_hidden_layers=12,
        ffn_dim=3072,
        max_position_embeddings=2048,
        do_layer_norm_before=True,
        _remove_final_layer_norm=False,
        word_embed_proj_dim=None,
        dropout=0.1,
        attention_dropout=0.0,
        num_attention_heads=12,
        activation_function="relu",
        layerdrop=0.0,
        init_std=0.02,
        use_cache=False,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=2,
        enable_bias=True,
        layer_norm_elementwise_affine=True,
        lora_config: dict = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.word_embed_proj_dim = (
            word_embed_proj_dim if word_embed_proj_dim is not None else hidden_size
        )
        self.ffn_dim = ffn_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.do_layer_norm_before = do_layer_norm_before
        # We keep these variables at `True` for backward compatibility.
        self.enable_bias = enable_bias
        self.layer_norm_elementwise_affine = layer_norm_elementwise_affine
        if lora_config is not None:
            lora_config = parse_lora_config(lora_config, num_hidden_layers, num_attention_heads)
        self.lora_config = lora_config

        # Note that the only purpose of `_remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        self._remove_final_layer_norm = _remove_final_layer_norm

    def __setattr__(self, key, value):
        if key == "lora_config" and value is not None:
            value = parse_lora_config(
                config=value, num_hidden_layers=self.num_hidden_layers, num_heads=self.num_attention_heads
            )
        return super().__setattr__(key, value)

