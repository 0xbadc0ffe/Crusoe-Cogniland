"""Block-based network modules for DreamerV3."""

import jax.numpy as jnp
import flax.linen as nn


class BlockLinear(nn.Module):
    """
    Block-wise linear transformation for efficient grouped operations.

    Used in BlockGRU for DreamerV3's RSSM.
    """

    features: int               # Total output features
    groups: int                 # Number of groups/blocks
    use_bias: bool = True      # Whether to use bias
    kernel_init: nn.initializers.Initializer = nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform')
    bias_init: nn.initializers.Initializer = nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply block-wise linear transformation.

        Args:
            x: Input tensor of shape (..., features)

        Returns:
            Output tensor of shape (..., features)
        """
        assert self.features % self.groups == 0, (
            f"features ({self.features}) must be divisible by groups ({self.groups})"
        )

        input_dim = x.shape[-1]
        features_per_group = self.features // self.groups

        # Reshape input to separate groups
        # (..., input_dim) -> (..., groups, input_per_group)
        if input_dim % self.groups == 0:
            input_per_group = input_dim // self.groups
            x_grouped = x.reshape((*x.shape[:-1], self.groups, input_per_group))
        else:
            # If input not divisible by groups, broadcast to all groups
            x_grouped = x[..., None, :].repeat(self.groups, axis=-2)
            input_per_group = input_dim

        # Apply separate linear transformation to each group
        # Weight shape: (groups, input_per_group, features_per_group)
        kernel = self.param(
            'kernel',
            self.kernel_init,
            (self.groups, input_per_group, features_per_group)
        )

        # Cast kernel to match input dtype for mixed precision
        kernel = kernel.astype(x.dtype)

        # Batch matrix multiply
        # (..., groups, input_per_group) @ (groups, input_per_group, features_per_group)
        # -> (..., groups, features_per_group)
        out_grouped = jnp.einsum('...gi,gif->...gf', x_grouped, kernel)

        if self.use_bias:
            bias = self.param(
                'bias',
                self.bias_init,
                (self.groups, features_per_group)
            )
            # Cast bias to match input dtype
            bias = bias.astype(x.dtype)
            out_grouped = out_grouped + bias

        # Flatten groups back
        # (..., groups, features_per_group) -> (..., features)
        out = out_grouped.reshape((*x.shape[:-1], self.features))

        return out