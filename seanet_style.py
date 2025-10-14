import math
import torch
import torch.nn as nn
import numpy as np
import typing as tp # Import typing for type hints
# Example for a Depthwise Separable Conv1d replacement
class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# In SimplifiedResnetBlock and CustomEncoder/ClassificationDecoder:
# Replace nn.Conv1d with SeparableConv1d.
# Note: ReflectionPad1d would still be applied BEFORE SeparableConv1d's depthwise part.

# New LayerNorm Wrapper
class LayerNorm1dChannel(nn.Module):
    def __init__(self, num_channels: int, **kwargs):
        super().__init__()
        # LayerNorm expects the normalized_shape to be the last dimension(s)
        # We want to normalize over the channel dimension (dim 1)
        # So we'll permute to [B, L, C], apply LayerNorm(C), then permute back to [B, C, L]
        self.norm = nn.LayerNorm(num_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, L]
        # Permute to [B, L, C] for LayerNorm
        x_permuted = x.permute(0, 2, 1)
        # Apply LayerNorm on the last dimension (C)
        x_normalized = self.norm(x_permuted)
        # Permute back to [B, C, L]
        return x_normalized.permute(0, 2, 1)


# --- Simplified Model Definition (from previous interaction) ---
class _BaseModelModule(nn.Module):
    def __init__(self, activation: str = 'GELU', activation_params: dict = {},
                 norm: str = 'BatchNorm1d', norm_params: tp.Dict[str, tp.Any] = {}):
        super().__init__()
        self.act_fn = getattr(nn, activation)(**activation_params)
        self.norm_type = norm
        self.norm_params = norm_params

    def _get_norm_layer(self, dim: int):
        if self.norm_type == 'BatchNorm1d':
            return nn.BatchNorm1d(dim, **self.norm_params)
        elif self.norm_type == 'GroupNorm':
            num_groups = self.norm_params.get('num_groups', 32)
            return nn.GroupNorm(min(num_groups, dim), dim, **self.norm_params)
        elif self.norm_type == 'LayerNorm':
            # Use our custom wrapper for LayerNorm to handle 1D conv outputs
            return LayerNorm1dChannel(dim, **self.norm_params)
        elif self.norm_type == 'InstanceNorm1d': # Add this option
            # InstanceNorm1d normalizes across the TIME dimension (L)
            # for each channel (C) in each sample (B).
            return nn.InstanceNorm1d(dim, **self.norm_params)
        elif self.norm_type == 'Identity' or self.norm_type is None:
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported norm type: {self.norm_type}. Choose from 'BatchNorm1d', 'GroupNorm', 'Identity', 'LayerNorm'.")
            
class SimplifiedResnetBlock(_BaseModelModule):
    def __init__(self, dim: int, kernel_sizes: tp.List[int] = [3, 1], dilations: tp.List[int] = [1, 1],
                 activation: str = 'GELU', activation_params: dict = {},
                 norm: str = 'BatchNorm1d', norm_params: tp.Dict[str, tp.Any] = {},
                 compress: int = 2, true_skip: bool = False):
        super().__init__(activation, activation_params, norm, norm_params)
        assert len(kernel_sizes) == len(dilations), 'Number of kernel sizes should match number of dilations'
        
        hidden_dim = dim // compress
        
        block_layers = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden_dim
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden_dim
            
            padding = (kernel_size // 2) * dilation 
            
            block_layers += [
                self.act_fn,
                nn.Dropout1d(p=0.1),
                # Add ReflectionPad1d BEFORE Conv1d
                nn.ReflectionPad1d(padding), # Apply padding using ReflectionPad1d
                SeparableConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation, padding=0), # Set Conv1d padding to 0
                self._get_norm_layer(out_chs)
            ]
        self.block = nn.Sequential(*block_layers)
        
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                SeparableConv1d(dim, dim, kernel_size=1),
                self._get_norm_layer(dim)
            )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class CustomEncoder(_BaseModelModule):
    def __init__(self, in_channels: int = 23, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: tp.List[int] = [5, 4, 2],
                 activation: str = 'GELU', activation_params: dict = {},
                 norm: str = 'BatchNorm1d', norm_params: tp.Dict[str, tp.Any] = {}, 
                 kernel_size: int = 7, last_kernel_size: int = 7, residual_kernel_size: int = 3, 
                 dilation_base: int = 2, compress: int = 2, true_skip: bool = False):
        super().__init__(activation, activation_params, norm, norm_params)
        self.in_channels = in_channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios)) # Encoder ratios are reversed in the encoder init.
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios) # This is the total downsampling factor

        model_layers: tp.List[nn.Module] = [
            nn.ReflectionPad1d(kernel_size // 2), # Add padding
            SeparableConv1d(in_channels, n_filters, kernel_size, padding=0),
            self._get_norm_layer(n_filters)
        ]
        
        mult = 1
        for i, ratio in enumerate(self.ratios):
            for j in range(n_residual_layers):
                model_layers += [
                    SimplifiedResnetBlock(
                        mult * n_filters, kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base ** j, 1],
                        activation=activation, activation_params=activation_params,
                        norm=norm, norm_params=norm_params,
                        compress=compress, true_skip=true_skip)
                ]
            
            in_conv_channels = mult * n_filters
            out_conv_channels = mult * n_filters * 2
            
            model_layers += [
                self.act_fn,
                nn.ReflectionPad1d(kernel_size // 2), # Add padding
                SeparableConv1d(in_conv_channels, out_conv_channels,
                          kernel_size=ratio * 2, stride=ratio,
                          padding=0), 
                self._get_norm_layer(out_conv_channels)
            ]
            mult *= 2

        model_layers += [
            self.act_fn,
            nn.ReflectionPad1d(kernel_size // 2), # Add padding
            SeparableConv1d(mult * n_filters, dimension, last_kernel_size, padding=0),
            self._get_norm_layer(dimension)
        ]
        self.model = nn.Sequential(*model_layers)

    def forward(self, x):
        return self.model(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class ClassificationDecoder(_BaseModelModule):
    def __init__(self, out_channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: tp.List[int] = [2, 4, 5], # Decoder ratios (upsampling factors)
                 activation: str = 'GELU', activation_params: dict = {},
                 norm: str = 'BatchNorm1d', norm_params: tp.Dict[str, tp.Any] = {}, 
                 kernel_size: int = 7, last_kernel_size: int = 7, residual_kernel_size: int = 3, 
                 dilation_base: int = 2):
        super().__init__(activation, activation_params, norm, norm_params)
        self.dimension = dimension
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        
        # Calculate initial 'mult' matching the final 'mult' of the encoder
        # Encoder ratios are [5, 4, 2]. Encoder 'mult' starts at 1.
        # After ratio 5: mult = 2
        # After ratio 4: mult = 4
        # After ratio 2: mult = 8
        # So, the decoder should start with mult = 8
        mult = int(2 ** len(self.ratios)) # This is correctly 2^3 = 8 for [2,4,5]

        model_layers: tp.List[nn.Module] = [
            nn.ReflectionPad1d(kernel_size // 2), # Add padding
            SeparableConv1d(dimension, mult * n_filters, kernel_size, padding=0),
            self._get_norm_layer(mult * n_filters)
        ]

        for i, ratio in enumerate(self.ratios):
            in_conv_channels = mult * n_filters
            out_conv_channels = mult * n_filters // 2
            
            # --- FIX STARTS HERE ---
            # To ensure exact upsampling with ConvTranspose1d and avoid output_padding error:
            # Common pattern: kernel_size = stride, padding = 0, output_padding = 0
            # This makes output_length = (input_length - 1) * stride + stride + 0 = input_length * stride
            
            # If you must maintain the 'kernel_size = ratio * 2' from encoder,
            # and want padding to align, it's trickier.
            # However, the original encoder uses `padding=ratio` and `kernel_size=ratio*2`, which is fine for Conv1d.
            # But for ConvTranspose1d, this combination (`kernel_size=2*stride, padding=stride`)
            # always leads to `output_padding = stride`, which is forbidden.
            
            # The simplest valid fix for exact upsampling using ConvTranspose1d with `stride=ratio`
            # that satisfies the `output_padding < stride` constraint is:
            adjusted_kernel_size = ratio  # Changed from ratio * 2
            adjusted_padding = 0         # Changed from ratio
            # output_padding will be 0 here to achieve exact upsampling.
            # --- FIX ENDS HERE ---

            model_layers += [
                self.act_fn,
                nn.ConvTranspose1d(in_conv_channels, out_conv_channels,
                                   kernel_size=adjusted_kernel_size, stride=ratio, # Use adjusted kernel_size
                                   padding=adjusted_padding, output_padding=0),   # Use adjusted padding and output_padding=0
                self._get_norm_layer(out_conv_channels)
            ]
            
            for j in range(n_residual_layers):
                model_layers += [
                    SimplifiedResnetBlock(
                        mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base ** j, 1],
                        activation=activation, activation_params=activation_params,
                        norm=norm, norm_params=norm_params)
                ]
            mult //= 2

        model_layers += [
            self.act_fn,
            nn.ReflectionPad1d(last_kernel_size // 2),
            SeparableConv1d(n_filters, out_channels, last_kernel_size, padding=0)
        ]
        
        self.model = nn.Sequential(*model_layers)

    def forward(self, z):
        return self.model(z)


class SEANetTransformerClassifier(nn.Module):
    """
    Combines a SEANet-style Encoder, a Positional Encoding, a Transformer, 
    and a SEANet-style Decoder for per-time-step classification.
    """
    def __init__(self,
                 input_channels: int = 23,
                 sampling_rate: int = 250,
                 encoder_dimension: int = 128,
                 encoder_n_filters: int = 32,
                 encoder_ratios: tp.List[int] = [5, 4, 2],
                 transformer_n_heads: int = 8,
                 transformer_n_layers: int = 4,
                 transformer_dim_feedforward: int = 512,
                 transformer_dropout: float = 0.1,
                 decoder_out_channels: int = 1,
                 activation: str = 'GELU', 
                 activation_params: dict = {},
                 norm: str = 'BatchNorm1d', 
                 norm_params: tp.Dict[str, tp.Any] = {},
                 n_residual_layers: int = 1,
                 kernel_size: int = 7,
                 last_kernel_size: int = 7,
                 residual_kernel_size: int = 3,
                 dilation_base: int = 2,
                 compress: int = 2,
                 true_skip: bool = False
                ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.input_channels = input_channels
        self.encoder_dimension = encoder_dimension
        self.encoder_ratios = encoder_ratios
        self.n_residual_layers = n_residual_layers

        self.encoder = CustomEncoder(
            in_channels=input_channels,
            dimension=encoder_dimension,
            n_filters=encoder_n_filters,
            ratios=encoder_ratios,
            activation=activation, activation_params=activation_params,
            norm=norm, norm_params=norm_params,
            kernel_size=kernel_size, last_kernel_size=last_kernel_size,
            residual_kernel_size=residual_kernel_size, dilation_base=dilation_base,
            compress=compress, true_skip=true_skip
        )

        self.total_downsample_factor = np.prod(self.encoder.ratios)
        max_input_length_for_pe = math.ceil(100 * sampling_rate / self.total_downsample_factor) + 5
        self.positional_encoding = PositionalEncoding(encoder_dimension, max_len=max_input_length_for_pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dimension,
            nhead=transformer_n_heads,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_n_layers
        )

        self.decoder = ClassificationDecoder(
            out_channels=decoder_out_channels,
            dimension=encoder_dimension,
            n_filters=encoder_n_filters,
            ratios=[r for r in encoder_ratios], # Using encoder_ratios here for consistency with upsampling factors
            activation=activation, activation_params=activation_params,
            norm=norm, norm_params=norm_params,
            kernel_size=kernel_size, last_kernel_size=last_kernel_size,
            residual_kernel_size=residual_kernel_size, dilation_base=dilation_base
        )
    
    def forward(self, x: torch.Tensor, original_lengths: tp.List[int] = None) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != self.input_channels:
            raise ValueError(f"Input tensor must be [B, {self.input_channels}, T], but got {x.shape}")

        original_input_length = x.shape[-1]
        batch_size = x.shape[0]

        # 1. Encoder
        encoded_features = self.encoder(x) # Output: [B, encoder_dimension, T']
        
        # Calculate new mask for encoded features
        src_key_padding_mask_for_transformer = None
        if original_lengths is not None:
            downsampled_lengths = []
            for length in original_lengths:
                # Need to be very precise here with how encoder changes length.
                # Simplest is often to use the ratio directly:
                downsampled_length = math.ceil(length / self.total_downsample_factor)
                downsampled_length = min(downsampled_length, encoded_features.shape[-1])
                downsampled_lengths.append(downsampled_length)
            
            src_key_padding_mask_for_transformer = torch.ones(
                batch_size, encoded_features.shape[-1], dtype=torch.bool, device=x.device
            )
            for i, length in enumerate(downsampled_lengths):
                src_key_padding_mask_for_transformer[i, :length] = False
        

        # 2. Prepare for Transformer: [T', B, C]
        encoded_features_permuted = encoded_features.permute(2, 0, 1) 

        # 3. Add Positional Encoding
        encoded_features_with_pe = self.positional_encoding(encoded_features_permuted)

        # 4. Transformer
        transformer_output = self.transformer_encoder(
            encoded_features_with_pe,
            src_key_padding_mask=src_key_padding_mask_for_transformer
        )

        # 5. Prepare for Decoder: [B, C, T']
        transformer_output_permuted = transformer_output.permute(1, 2, 0) 

        # 6. Classification Decoder
        decoded_logits = self.decoder(transformer_output_permuted) 

        # 7. Trim or pad output to match original input length
        # This step is still important because encoder/decoder upsampling might not be *exactly* perfect for all input lengths
        # due to ceil/floor operations, or due to stride/kernel/padding interactions.
        output_length_after_decoder = decoded_logits.shape[-1]
        if output_length_after_decoder > original_input_length:
            decoded_logits = decoded_logits[..., :original_input_length]
        elif output_length_after_decoder < original_input_length:
            padding_needed = original_input_length - output_length_after_decoder
            decoded_logits = nn.functional.pad(decoded_logits, (0, padding_needed))

        return decoded_logits

import math
import torch
import torch.nn as nn
import numpy as np
import typing as tp # Import typing for type hints

# --- Re-use existing components ---
# SeparableConv1d, LayerNorm1dChannel, _BaseModelModule, SimplifiedResnetBlock, CustomEncoder, PositionalEncoding
# All these are used in the new class, so they need to be defined.
# I will not repeat the full code for them, but assume they are available from the previous block.
# The code below is a new class definition that leverages the old components.

# New Model for sequence-to-single-label classification
class SEANetTransformerEncoderClassifier(nn.Module):
    """
    Combines a SEANet-style Encoder, a Positional Encoding, a Transformer,
    and a final linear classifier for single-label classification.
    """
    def __init__(self,
                 input_channels: int = 23,
                 sampling_rate: int = 250,
                 encoder_dimension: int = 128,
                 encoder_n_filters: int = 32,
                 encoder_ratios: tp.List[int] = [5, 4, 2],
                 transformer_n_heads: int = 8,
                 transformer_n_layers: int = 4,
                 transformer_dim_feedforward: int = 512,
                 transformer_dropout: float = 0.1,
                 num_classes: int = 2,  # Specify the number of output classes
                 activation: str = 'GELU',
                 activation_params: dict = {},
                 norm: str = 'BatchNorm1d',
                 norm_params: tp.Dict[str, tp.Any] = {},
                 n_residual_layers: int = 1,
                 kernel_size: int = 7,
                 last_kernel_size: int = 7,
                 residual_kernel_size: int = 3,
                 dilation_base: int = 2,
                 compress: int = 2,
                 true_skip: bool = False
                ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.input_channels = input_channels
        self.encoder_dimension = encoder_dimension
        self.encoder_ratios = encoder_ratios

        # 1. Encoder: Same as before
        self.encoder = CustomEncoder(
            in_channels=input_channels,
            dimension=encoder_dimension,
            n_filters=encoder_n_filters,
            ratios=encoder_ratios,
            activation=activation, activation_params=activation_params,
            norm=norm, norm_params=norm_params,
            kernel_size=kernel_size, last_kernel_size=last_kernel_size,
            residual_kernel_size=residual_kernel_size, dilation_base=dilation_base,
            compress=compress, true_skip=true_skip
        )

        self.total_downsample_factor = np.prod(self.encoder.ratios)
        max_input_length_for_pe = math.ceil(100 * sampling_rate / self.total_downsample_factor) + 5
        
        # 2. Positional Encoding: Same as before
        self.positional_encoding = PositionalEncoding(encoder_dimension, max_len=max_input_length_for_pe)

        # 3. Transformer: Same as before
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dimension,
            nhead=transformer_n_heads,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_n_layers
        )

        # 4. Global Average Pooling (to aggregate the sequence)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 5. Final Classifier (a simple linear layer)
        self.classifier = nn.Sequential(
            _BaseModelModule._get_norm_layer(self, encoder_dimension), # Optional normalization
            _BaseModelModule(activation=activation, activation_params=activation_params).act_fn, # Activation
            nn.Linear(encoder_dimension, num_classes)
        )

    def forward(self, x: torch.Tensor, original_lengths: tp.List[int] = None) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != self.input_channels:
            raise ValueError(f"Input tensor must be [B, {self.input_channels}, T], but got {x.shape}")

        batch_size = x.shape[0]

        # 1. Encoder
        encoded_features = self.encoder(x) # [B, encoder_dimension, T']
        
        # Create a mask for the transformer if lengths are provided
        src_key_padding_mask_for_transformer = None
        if original_lengths is not None:
            downsampled_lengths = [min(math.ceil(l / self.total_downsample_factor), encoded_features.shape[-1]) for l in original_lengths]
            src_key_padding_mask_for_transformer = torch.ones(
                batch_size, encoded_features.shape[-1], dtype=torch.bool, device=x.device
            )
            for i, length in enumerate(downsampled_lengths):
                src_key_padding_mask_for_transformer[i, :length] = False

        # 2. Permute for Transformer: [T', B, C]
        encoded_features_permuted = encoded_features.permute(2, 0, 1)

        # 3. Add Positional Encoding
        encoded_features_with_pe = self.positional_encoding(encoded_features_permuted)

        # 4. Transformer
        # `src_key_padding_mask` is of shape [B, T']
        transformer_output = self.transformer_encoder(
            encoded_features_with_pe,
            src_key_padding_mask=src_key_padding_mask_for_transformer
        ) # [T', B, C]

        # 5. Permute back for global pooling: [B, C, T']
        transformer_output_permuted = transformer_output.permute(1, 2, 0)

        # 6. Global Average Pooling over the time dimension (T')
        # This collapses the sequence into a single vector.
        pooled_output = self.global_pool(transformer_output_permuted).squeeze(-1) # [B, C]

        # 7. Final Classification
        logits = self.classifier(pooled_output) # [B, num_classes]

        return logits

# --- Helper function for masking (as provided previously) ---
def create_mask(lengths, max_len, device):
    """
    Creates a boolean mask for sequences, where True indicates padded elements.
    Args:
        lengths (list or tensor): A list or tensor of actual sequence lengths.
        max_len (int): The maximum length of the padded sequences.
        device (torch.device): The device to create the mask on.
    Returns:
        torch.Tensor: A boolean mask of shape (batch_size, max_len), True for padding.
    """
    batch_size = len(lengths)
    mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=device)
    for i, l in enumerate(lengths):
        mask[i, :l] = False  # False means not masked (real data)
    return mask