"""Recurrent neural network modules for RL agents.

Provides reusable LSTM components with episode boundary reset handling,
designed for use with Flax's nn.scan for efficient sequence processing.

Key components:
    - LSTMState: NamedTuple for LSTM hidden state (c, h)
    - ScannedRNN: LSTM with built-in reset handling using nn.scan
"""

from typing import Tuple, NamedTuple
from functools import partial

import flax.linen as nn
import jax.numpy as jnp


class LSTMState(NamedTuple):
    """LSTM hidden state (c, h) - matches nn.OptimizedLSTMCell carry format.

    Attributes:
        c: Cell state with shape (batch, lstm_size)
        h: Hidden state with shape (batch, lstm_size)
    """
    c: jnp.ndarray
    h: jnp.ndarray


class ScannedRNN(nn.Module):
    """LSTM with built-in reset handling using nn.scan.

    This is the idiomatic Flax way to handle recurrent networks.
    Resets hidden state to zeros when done=True (episode boundary).

    The @nn.scan decorator on __call__ enables efficient sequence processing
    while sharing parameters with the single-step method.

    Attributes:
        hidden_size: Size of the LSTM hidden state.
    """
    hidden_size: int

    def setup(self):
        self.cell = nn.OptimizedLSTMCell(self.hidden_size)

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    def __call__(self, carry: LSTMState, x: Tuple[jnp.ndarray, jnp.ndarray]):
        """Scanned LSTM step with reset handling (for sequence processing).

        Args:
            carry: LSTM state (c, h)
            x: Tuple of (inputs, resets) where:
                - inputs: (batch, features)
                - resets: (batch,) boolean/float, True = reset state

        Returns:
            Tuple of (new_carry, output) where output = new_h
        """
        inputs, resets = x
        return self.step(carry, inputs, resets)

    def step(
        self,
        carry: LSTMState,
        inputs: jnp.ndarray,
        resets: jnp.ndarray = None,
    ) -> Tuple[LSTMState, jnp.ndarray]:
        """Single LSTM step with optional reset after update.

        Args:
            carry: Current LSTM state (c, h)
            inputs: Input features with shape (batch, features)
            resets: Optional boolean array (batch,) indicating episode boundaries.
                    If True, the hidden state is reset to zeros after the step.

        Returns:
            Tuple of (new_carry, output) where:
                - new_carry: Updated LSTMState (potentially reset)
                - output: LSTM output (same as new_carry.h)
        """
        new_lstm_carry, output = self.cell((carry.c, carry.h), inputs)
        new_carry = LSTMState(c=new_lstm_carry[0], h=new_lstm_carry[1])

        if resets is not None:
            resets = resets.astype(jnp.bool_)
            reset_mask = resets[:, None]
            zero = LSTMState(
                c=jnp.zeros_like(new_carry.c),
                h=jnp.zeros_like(new_carry.h),
            )
            new_carry = LSTMState(
                c=jnp.where(reset_mask, zero.c, new_carry.c),
                h=jnp.where(reset_mask, zero.h, new_carry.h),
            )

        return new_carry, output

    @staticmethod
    def initialize_carry(hidden_size: int, batch_size: int) -> LSTMState:
        """Initialize zero LSTM state.

        Args:
            hidden_size: Size of the LSTM hidden state
            batch_size: Batch size

        Returns:
            LSTMState with zeros for both c and h
        """
        return LSTMState(
            c=jnp.zeros((batch_size, hidden_size)),
            h=jnp.zeros((batch_size, hidden_size)),
        )
