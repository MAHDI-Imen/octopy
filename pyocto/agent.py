from typing import List
import torch
from pyocto.model import PyOcto
import torchvision.transforms.functional as transforms_f
import torchvision.transforms as transforms
import einops
import numpy as np


class Agent(object):
    def __init__(
        self,
        policy: PyOcto,
        cameras: List[str],
        resize: List[int] = (256, 256),
    ):
        self.policy = policy
        self.cameras = cameras
        self.resize = resize

    def predict_action(self, observation, task: str, step=int, return_batch=False):
        """
        Args:
            observation: RLBnech observation at step t. Must include camera views specified at initialization as well as their respective point clouds.
            task: text description of the task
            step: indice of curent timestep
            return_batch: whether or not to return the batch with heatmaps predicted included.
        Returns:
            action: predicted action as an 8D tensor.
            batch: batch of data used for prediction with heatmaps included if return_batch is True.

        """
        self.policy.eval()
        device = next(self.policy.parameters()).device
        batch = self.get_batch(observation, task, step)
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        action, heatmaps = self.policy(batch, return_heatmaps=True)

        batch["heatmaps"] = heatmaps
        if return_batch:
            return action[0], batch

        return action[0]

    def get_batch(self, observation, task: str, step=int):
        rgbs = self._extract_visual_input(observation, "rgb")
        pcds = self._extract_visual_input(observation, "point_cloud")
        step_ids = torch.tensor([step])
        task_desc = [task]
        batch = {
            "rgbs": rgbs,
            "pcds": pcds,
            "step_ids": step_ids,
            "task_desc": task_desc,
        }
        return batch

    def _extract_visual_input(self, observation, visual_input: str):
        visuals = [
            getattr(observation, f"{cam}_{visual_input}") for cam in self.cameras
        ]
        visuals = (
            torch.tensor(np.array(visuals)).float().permute(0, 3, 1, 2).unsqueeze(0)
        )

        # resize to 256x256
        visuals = einops.rearrange(visuals, "t n c h w -> (t n) c h w")

        visuals = transforms_f.resize(
            visuals, (256, 256), transforms.InterpolationMode.BILINEAR
        )

        # arrange back
        visuals = einops.rearrange(visuals, "(t n) c h w -> t n c h w", t=1)

        return visuals
