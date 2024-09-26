from typing import Callable, Optional, Sequence, Any, Union


import torch
import torch.nn as nn
import torch.nn.functional as F


class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs.

    Attributes:
      posemb_shape: shape of the positional embeddings (batch_size, seq_len, emb_dim).
      posemb_init: positional embedding initializer.
    """

    def __init__(self, posemb_init=torch.zeros):
        super(AddPositionEmbs, self).__init__()
        self.posemb_init = posemb_init
        self.pos_embedding = nn.UninitializedParameter()

    def forward(self, inputs):
        """Applies the AddPositionEmbs module.

        Args:
          inputs: Inputs to the layer.

        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert (
            inputs.ndim == 3
        ), f"Number of dimensions should be 3, but it is: {inputs.ndim}"

        if isinstance(self.pos_embedding, nn.UninitializedParameter):
            pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
            self.pos_embedding = nn.Parameter(self.posemb_init(*pos_emb_shape))

        return inputs + self.pos_embedding


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(
        self,
        input_dim: int,
        mlp_dim: int,
        dtype=torch.float32,
        out_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super(MlpBlock, self).__init__()
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.actual_out_dim = input_dim if self.out_dim is None else self.out_dim

        self.dropout_rate = dropout_rate
        self.add_module("Dense_0", nn.Linear(self.input_dim, self.mlp_dim))
        self.add_module("GELU", nn.GELU())
        self.add_module("Dropout_0", nn.Dropout(self.dropout_rate))
        self.add_module("Dense_1", nn.Linear(self.mlp_dim, self.actual_out_dim))
        self.add_module("Dropout_1", nn.Dropout(self.dropout_rate))

    def forward(self, inputs):
        """Applies Transformer MlpBlock module."""
        x = self.Dense_0(inputs)
        x = self.GELU(x)
        x = self.Dropout_0(x)
        output = self.Dense_1(x)
        output = self.Dropout_1(output)
        return output


class MAPHead(nn.Module):
    """Multihead Attention Pooling.

    From https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py
    """

    def __init__(
        self,
        input_dim: int,
        mlp_dim: Optional[int] = None,
        num_heads: Optional[int] = 8,
        num_readouts: Optional[int] = 1,
    ):
        super(MAPHead, self).__init__()
        self.d = input_dim
        self.mlp_dim = mlp_dim if mlp_dim is not None else 4 * self.d
        self.num_heads = num_heads
        self.num_readouts = num_readouts

        self.probe = nn.Parameter(torch.ones(1, self.num_readouts, self.d))
        nn.init.xavier_uniform_(self.probe)

        self.add_module("MlpBlock_0", MlpBlock(self.d, self.mlp_dim, dropout_rate=0.0))

        self.add_module(
            "MultiHeadDotProductAttention_0",
            nn.MultiheadAttention(
                self.d, self.num_heads, dropout=0.0, batch_first=True
            ),
        )

        self.add_module("LayerNorm_0", nn.LayerNorm(self.d))

    def forward(self, x: torch.Tensor, mask=None):
        *batch_dims, l, d = x.shape
        assert d == self.d, f"Expected last dim to be {self.d} got {d}"

        x = x.reshape(-1, l, d)
        batch_size = x.shape[0]

        probe = self.probe.repeat(batch_size, 1, 1)

        if mask is not None:
            mask = mask.reshape(-1, l)
            # repeat mask for each head and readout
            mask = mask[:, None, None, :].expand(
                batch_size, self.num_heads, self.num_readouts, l
            )
            mask = mask.reshape(batch_size * self.num_heads, self.num_readouts, l)

            # convert mask to float because pytorch has a bug with bool
            mask = mask.to(torch.float32)
            mask = mask.masked_fill(mask == 0, float("-inf"))
            mask = mask.masked_fill(mask == 1, 0.0)

        out = self.MultiHeadDotProductAttention_0(probe, x, x, attn_mask=mask)[0]
        y = self.LayerNorm_0(out)
        out = out + self.MlpBlock_0(y)
        out = out.reshape(*batch_dims, self.num_readouts, self.d)

        return out


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiheadAttention
    """

    def __init__(
        self,
        input_dim: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
    ):
        super(Encoder1DBlock, self).__init__()
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.LayerNorm_0 = nn.LayerNorm(self.input_dim)
        self.MultiHeadDotProductAttention_0 = nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=self.num_heads,
            dropout=self.attention_dropout_rate,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.MlpBlock_0 = MlpBlock(
            input_dim=self.input_dim,
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
        )
        self.LayerNorm_1 = nn.LayerNorm(self.input_dim)

    def forward(self, inputs, attention_mask):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert (
            inputs.ndim == 3
        ), f"Expected (batch, seq, hidden) got {inputs.shape}"  # (batch, len, emb)

        x = self.LayerNorm_0(inputs)
        if attention_mask.ndim > 3:
            attention_mask = attention_mask.repeat(1, self.num_heads, 1, 1)
            # merge first two dimensions
            attention_mask = attention_mask.reshape(-1, *attention_mask.shape[2:])

        # cast the mask to float because pytorch has a bug with bool
        attention_mask = attention_mask.to(torch.float32)
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float("-inf"))
        attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)

        x = self.MultiHeadDotProductAttention_0(
            x, x, x, attn_mask=attention_mask.to(torch.float32)
        )[0]
        x = self.dropout(x)
        x = x + inputs

        # MLP block.
        y = self.LayerNorm_1(x)
        y = self.MlpBlock_0(y)

        return x + y


class Transformer(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block
      in_dim: input dimension of the transformer
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate in self attention.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        mlp_dim: int,
        num_attention_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        add_position_embedding: bool = False,
    ):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.add_position_embedding = add_position_embedding

        for lyr in range(self.num_layers):
            self.add_module(
                f"encoderblock_{lyr}",
                Encoder1DBlock(
                    input_dim=input_dim,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_attention_heads,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                ),
            )
        self.encoder_norm = nn.LayerNorm(self.input_dim)

        if self.add_position_embedding:
            posemb_init = nn.init.normal_(torch.empty(1, 1, self.input_dim), std=0.02)
            self.position_embedding = AddPositionEmbs(
                posemb_shape=posemb_init.shape, posemb_init=posemb_init
            )
            self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x, attention_mask):
        """Applies Transformer model on the inputs.

        Args:
          x: Inputs to the layer.
          attention_mask: Mask for attention computation.

        Returns:
          output of a transformer encoder.
        """
        assert x.ndim == 3  # (batch, len, emb)

        if self.add_position_embedding:
            x = self.position_embedding(x)
            x = self.dropout(x)

        for lyr in range(self.num_layers):
            x = getattr(self, f"encoderblock_{lyr}")(x, attention_mask)

        encoded = self.encoder_norm(x)

        return encoded
