from typing import List, Union

import torch
import torch.nn as nn

from pyocto.backbone import OctoBackbone
from pyocto.action_head import ActionHead

import logging

logger = logging.getLogger(__name__)


class PyOcto(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.backbone = OctoBackbone()
        self.action_head = ActionHead(
            token_embedding_size=self.backbone.token_embedding_size,
            decoder_features=self.backbone.decoder_features[0],
            decoder_out_features=self.backbone.decoder_features[1],
            latent_im_size=self.backbone.latent_im_size,
            max_steps=20,
            num_cameras=3,
        )

    def forward(self, batch, compute_loss=False, return_heatmaps=False):
        """Input batch contains:
        - rgbs, pcds: (B, N, C, H, W) B: batch_size, N: #cameras
        - step_ids: (B, )
        - task_desc: (B, ) list of strings
        """

        batch = self.prepare_batch(batch)

        rgb_obs = batch["rgbs"]
        pc_obs = batch["pcds"]
        text_input = batch["task_desc"]
        step_ids = batch["step_ids"]

        transformer_output = self.backbone(text_input, rgb_obs)

        actions, xt_heatmap = self.action_head(
            transformer_output,
            pc_obs,
            step_ids,
        )

        if compute_loss:
            gt_actions = batch["actions"]
            losses = self.action_head.loss_fn.compute_loss(actions, gt_actions)
            if return_heatmaps:
                return losses, actions, xt_heatmap
            return losses, actions

        if return_heatmaps:
            return actions, xt_heatmap

        return actions

    def prepare_batch(self, batch):
        device = next(self.parameters()).device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_frozen_params(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    @property
    def frozen_params(self):
        return [p for p, t in self.named_parameters() if not t.requires_grad]

    @property
    def trainable_params(self):
        return [p for p, t in self.named_parameters() if t.requires_grad]

    def freeze(self, layers: Union[List[str], str]):
        if layers == "all":
            for p in self.parameters():
                p.requires_grad = False
        elif layers == "backbone":
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            for layer in layers:
                if isinstance(getattr(self, layer), nn.Module):
                    for p in getattr(self, layer).parameters():
                        p.requires_grad = False
                elif isinstance(getattr(self, layer), nn.Parameter):
                    getattr(self, layer).requires_grad = False

    def unfreeze(self, layers: Union[List[str], str]):
        if layers == "all":
            for p in self.parameters():
                p.requires_grad = True
            # the only exception will be the task_tokenizer
            logger.info("task_tokenizer will be exceptionnaly unfrozen")
            for p in self.backbone.task_tokenizer.parameters():
                p.requires_grad = False
        else:
            for layer in layers:
                if isinstance(getattr(self, layer), nn.Module):
                    for p in getattr(self, layer).parameters():
                        p.requires_grad = True
                elif isinstance(getattr(self, layer), nn.Parameter):
                    getattr(self, layer).requires_grad = True


if __name__ == "__main__":
    model = PyOcto()

    batch = {
        "rgbs": torch.randn(2, 3, 3, 256, 256),
        "pcds": torch.randn(2, 3, 3, 256, 256),
        "task_desc": ["pick up the object", "place the object"],
        "actions": torch.randn(2, 8),
        "step_ids": torch.randint(0, 10, (2,)),
    }

    loss, actions = model(batch, compute_loss=True)

    print(actions.shape)
    print(loss)
