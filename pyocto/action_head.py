from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from pyocto.layers import upsample_layer, dense_layer, ConvLayer
from pyocto.losses import ActionLoss
from pyocto.utils.action_space import normalise_quat


class ActionHead(nn.Module):
    """
    Action head for the octo model. Predicts the gripper translation, rotation and openness.
    """

    def __init__(
        self,
        token_embedding_size: int,
        decoder_features: List[int],
        decoder_out_features: int,
        latent_im_size: Tuple[int, int],
        max_steps: int,
        num_cameras: int,
    ):
        super().__init__()

        self.token_embedding_size = token_embedding_size
        self.latent_im_size = latent_im_size

        self.step_embedding = nn.Embedding(max_steps, token_embedding_size)
        self.cam_embedding = nn.Embedding(num_cameras, token_embedding_size)
        self.pix_embedding = nn.Embedding(
            np.prod(latent_im_size), self.token_embedding_size
        )

        self.trans_decoder = nn.ModuleList()
        self.trans_projection = ConvLayer(
            self.token_embedding_size,
            decoder_out_features,
            kernel_size=1,
            stride_size=1,
        )
        self.trans_decoder.append(
            ConvLayer(
                decoder_out_features * 2,
                decoder_features[0],
                kernel_size=1,
                stride_size=1,
            )
        )

        for i in range(len(decoder_features) - 1):
            self.trans_decoder.append(
                nn.Sequential(
                    *upsample_layer(decoder_features[i] * 2, decoder_features[i + 1])
                )
            )

        self.maps_to_coord = ConvLayer(
            in_channels=decoder_features[-1],
            out_channels=1,
            kernel_size=(1, 1),
            stride_size=(1, 1),
            apply_norm=False,
            apply_activation=False,
        )

        self.quat_projection = ConvLayer(
            self.token_embedding_size,
            decoder_out_features,
            kernel_size=1,
            stride_size=1,
        )
        quat_hidden_size = decoder_out_features * 3
        self.quat_decoder = nn.Sequential(
            ConvLayer(
                in_channels=quat_hidden_size,
                out_channels=quat_hidden_size // 2,
                kernel_size=(3, 3),
                stride_size=(2, 2),
            ),
            ConvLayer(
                in_channels=quat_hidden_size // 2,
                out_channels=quat_hidden_size // 4,
                kernel_size=(3, 3),
                stride_size=(2, 2),
            ),
            nn.AdaptiveAvgPool2d(1),
            Rearrange("b c h w -> b (c h w)"),
            *dense_layer(quat_hidden_size // 4, quat_hidden_size // 4),
            *dense_layer(quat_hidden_size // 4, 3 + 4 + 1, apply_activation=False),
        )

        self.loss_fn = ActionLoss()

    def forward(self, transformer_output, pc_obs, step_ids):
        visual_embeddings = transformer_output["visual_embeddings"]
        enc_fts = transformer_output["enc_fts"]

        new_im_height, new_im_width = (
            transformer_output["new_im_height"],
            transformer_output["new_im_width"],
        )

        device = visual_embeddings.device

        batch_size, n_cameras, _, im_height, im_width = pc_obs.size()

        step_embeds = self.step_embedding(step_ids)  # (B, T, C)
        cam_embeds = self.cam_embedding(
            torch.arange(n_cameras).long().to(device)
        )  # (N, C)
        pix_embeds = self.pix_embedding(
            torch.arange(np.prod(self.latent_im_size)).long().to(device)
        )  # (H * W, C)

        visual_embeddings = (
            visual_embeddings
            + einops.rearrange(step_embeds, "b c -> b 1 1 c")
            + einops.rearrange(cam_embeds, "n c -> 1 n 1 c")
            + einops.rearrange(pix_embeds, "l c -> 1 1 l c")
        )

        visual_embeddings = einops.rearrange(
            visual_embeddings,
            "b n (h w) c -> (b n) c h w",
            b=batch_size,
            n=n_cameras,
            h=new_im_height,
            w=new_im_width,
        )

        # predict the translation of the gripper
        enc_fts.reverse()

        x = self.trans_projection(visual_embeddings)
        if len(enc_fts) > 1:
            for i, layer in enumerate(self.trans_decoder):
                x = torch.cat([x, enc_fts[i]], dim=1)
                x = layer(x)
        else:
            for i, layer in enumerate(self.trans_decoder):
                x = layer(x)

        xt_heatmap = self.maps_to_coord(x)
        xt_heatmap = einops.rearrange(
            xt_heatmap, "(b n) c h w -> b (n c h w)", n=n_cameras, c=1
        )
        xt_heatmap = torch.softmax(xt_heatmap / 0.1, dim=-1)
        xt_heatmap = einops.rearrange(
            xt_heatmap,
            "b (n c h w) -> b n c h w",
            n=n_cameras,
            c=1,
            h=im_height,
            w=im_width,
        )

        xt = einops.reduce(pc_obs * xt_heatmap, "b n c h w -> b c", "sum")

        # predict the (translation_offset, rotation and openness) of the gripper
        if len(enc_fts) > 1:
            xg = self.quat_projection(visual_embeddings) + enc_fts[0]
        else:
            xg = self.quat_projection(visual_embeddings)
        xg = einops.rearrange(xg, "(b n) c h w -> b (n c) h w", n=n_cameras)
        xg = self.quat_decoder(xg)
        xt_offset = xg[..., :3]
        xr = normalise_quat(xg[..., 3:7])
        xo = xg[..., 7].unsqueeze(-1)

        actions = torch.cat([xt + xt_offset, xr, xo], dim=-1)

        return actions, xt_heatmap
