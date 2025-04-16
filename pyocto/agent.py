from typing import List
import torch
from pyocto.model import PyOcto
import torchvision.transforms.functional as transforms_f
import torchvision.transforms as transforms
import einops
import numpy as np

from tapas_gmm.utils.vision import construct_pointcloud_batch, construct_pointcloud
from tapas_gmm.utils.observation import collate


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

    def predict_action(self, observation, task: str, return_batch=False):
        """
        Args:
            observation: RLBnech observation at step t. Must include camera views specified at initialization as well as their respective point clouds.
            task: text description of the task
            return_batch: whether or not to return the batch with heatmaps predicted included.
        Returns:
            action: predicted action as an 8D tensor.
            batch: batch of data used for prediction with heatmaps included if return_batch is True.

        """
        self.policy.eval()
        device = next(self.policy.parameters()).device
        batch = self.get_batch(observation, task)
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        action, heatmaps = self.policy(batch, return_heatmaps=True)
        # action last dim should be 1 if positive and 0 if negative
        action[0, -1] = action[0, -1] > 0.0
        batch["heatmaps"] = heatmaps
        if return_batch:
            return action[0], batch

        return action[0]

    def get_batch(self, obs, task: str):
        rgbs = []
        pcds = []
        for cam in self.cameras:
            rgb = getattr(obs, cam + "_rgb").transpose((2, 0, 1)) / 255
            depth = getattr(obs, cam + "_depth")
            extr = obs.misc[cam + "_camera_extrinsics"]
            intr = obs.misc[cam + "_camera_intrinsics"].astype(float)
            pcd = construct_pointcloud(depth, extr, intr).transpose((2, 0, 1))
            rgbs.append(rgb)
            pcds.append(pcd)

        rgbs = torch.tensor(np.array(rgbs)).float().unsqueeze(0)
        pcds = torch.tensor(np.array(pcds)).float().unsqueeze(0)
        task_desc = [task]
        batch = {
            "rgbs": rgbs,
            "pcds": pcds,
            "task_desc": task_desc,
        }
        return batch
