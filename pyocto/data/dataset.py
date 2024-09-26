from typing import List, Dict, Optional

import os
import numpy as np
import einops

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

import lmdb
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

import pandas as pd


from pyocto.data.utils import pad_tensors, gen_seq_masks

from rlbench.tasks import *


class DataTransform(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, data) -> Dict[str, torch.Tensor]:
        """
        Inputs:
            data: dict
                - rgb: (T, N, C, H, W), N: num of cameras
                - pc: (T, N, C, H, W)
        """
        keys = list(data.keys())

        # Continuous range of scales
        sc = np.random.uniform(*self.scales)

        t, n, c, raw_h, raw_w = data[keys[0]].shape
        data = {k: v.flatten(0, 1) for k, v in data.items()}  # (t*n, h, w, c)
        resized_size = [int(raw_h * sc), int(raw_w * sc)]

        # Resize based on randomly sampled scale
        data = {
            k: transforms_f.resize(
                v, resized_size, transforms.InterpolationMode.BILINEAR
            )
            for k, v in data.items()
        }

        # Adding padding if crop size is smaller than the resized size
        if raw_h > resized_size[0] or raw_w > resized_size[1]:
            right_pad = max(raw_w - resized_size[1], 0)
            bottom_pad = max(raw_h - resized_size[0], 0)
            data = {
                k: transforms_f.pad(
                    v,
                    padding=[0, 0, right_pad, bottom_pad],
                    padding_mode="edge",
                )
                for k, v in data.items()
            }

        # Random Cropping
        i, j, h, w = transforms.RandomCrop.get_params(
            data[keys[0]], output_size=(raw_h, raw_w)
        )

        data = {k: transforms_f.crop(v, i, j, h, w) for k, v in data.items()}

        data = {
            k: einops.rearrange(v, "(t n) c h w -> t n c h w", t=t)
            for k, v in data.items()
        }

        return data


class KeystepDataset(Dataset):
    def __init__(
        self,
        data_dir,
        taskvars,
        cameras=("left_shoulder", "right_shoulder", "wrist"),
        is_training=False,
        task_desc=None,
        in_memory=False,
        include_last=False,
        resize=(256, 256),
    ):
        self.data_dir = data_dir
        self.taskvars = taskvars
        self.taskvar_to_id = {x: i for i, x in enumerate(self.taskvars)}
        self.cameras = cameras
        self.in_memory = in_memory
        self.is_training = is_training
        self.task_desc_dict = None
        self.resize = resize
        if self.in_memory:
            self.memory = {}
        self.include_last = include_last
        self._transform = DataTransform((0.75, 1.25))

        self.lmdb_envs, self.lmdb_txns = [], []
        self.episode_ids = []
        for i, taskvar in enumerate(self.taskvars):
            lmdb_env = lmdb.open(os.path.join(data_dir, taskvar), readonly=True)
            self.lmdb_envs.append(lmdb_env)
            lmdb_txn = lmdb_env.begin()
            self.lmdb_txns.append(lmdb_txn)
            keys = list(lmdb_txn.cursor().iternext(values=False))
            if b"stats" in keys:
                keys.remove(b"stats")
            # order keys by the episode number
            keys = sorted(
                keys, key=lambda x: int(x.decode("ascii").split("episode")[-1])
            )
            self.episode_ids.extend([(i, key) for key in keys])
            if self.in_memory:
                self.memory[f"taskvar{i}"] = {}

        if task_desc is not None:
            task_desc_lists = pd.read_csv(task_desc)
            self.task_desc_dict = {}
            for i, row in task_desc_lists.iterrows():
                self.task_desc_dict[row["name"]] = list(row["desc"].split(", "))

    def __exit__(self):
        for lmdb_env in self.lmdb_envs:
            lmdb_env.close()

    def __len__(self):
        return len(self.episode_ids)

    def get_taskvar_episode(self, taskvar_idx, episode_key):
        if self.in_memory:
            mem_key = f"taskvar{taskvar_idx}"
            if episode_key in self.memory[mem_key]:
                return self.memory[mem_key][episode_key]

        value = self.lmdb_txns[taskvar_idx].get(episode_key)
        value = msgpack.unpackb(value)
        if self.in_memory:
            self.memory[mem_key][episode_key] = value
        return value

    def __getitem__(self, idx):
        taskvar_idx, episode_key = self.episode_ids[idx]

        value = self.get_taskvar_episode(taskvar_idx, episode_key)
        rgbs = self._process_visual_input(value["rgb"])
        pcs = self._process_visual_input(value["pc"])
        # rgb, pcd: (T, N, C, H, W)

        num_steps, num_cameras, _, im_height, im_width = rgbs.size()

        outs = {"rgbs": rgbs, "pcds": pcs}

        if self.is_training:
            outs = self._transform(outs)

        outs["step_ids"] = torch.arange(0, num_steps).long()
        if not self.include_last:
            outs["actions"] = torch.tensor(value["action"][1:])
        else:
            action = value["action"][1:]
            action = np.concatenate([action, [action[-1]]], axis=0)
            outs["actions"] = torch.tensor(action)
        outs["episode_ids"] = episode_key.decode("ascii")
        outs["taskvars"] = self.taskvars[taskvar_idx]
        outs["taskvar_ids"] = taskvar_idx

        if self.task_desc_dict is not None:
            list_desc = self.task_desc_dict[outs["taskvars"]]
            # random choice of the description
            outs["task_desc"] = list_desc[np.random.randint(len(list_desc))]

        return outs

    def _process_visual_input(self, visual_input):
        if not self.include_last:
            visual_input = (
                torch.tensor(visual_input[:-1]).float().permute(0, 1, 4, 2, 3)
            )  # (T, N, C, H, W)
            T = visual_input.shape[0]

        else:
            visual_input = torch.tensor(visual_input).float().permute(0, 1, 4, 2, 3)

        if self.resize != (128, 128):
            # resize to 256x256
            visual_input = einops.rearrange(visual_input, "t n c h w -> (t n) c h w")

            visual_input = transforms_f.resize(
                visual_input, (256, 256), transforms.InterpolationMode.BILINEAR
            )

            # arrange back
            visual_input = einops.rearrange(
                visual_input, "(t n) c h w -> t n c h w", t=T
            )

            visual_input = visual_input[:, :3, ...]

        return visual_input


def stepwise_collate_fn(data: List[Dict]):
    batch = {}

    for key in data[0].keys():
        if key == "taskvar_ids":
            batch[key] = [
                torch.LongTensor([v["taskvar_ids"]] * len(v["step_ids"])) for v in data
            ]

        else:
            batch[key] = [v[key] for v in data]

    for key in ["rgbs", "pcds", "taskvar_ids", "step_ids", "actions"]:
        # e.g. rgbs: (B*T, N, C, H, W)
        batch[key] = torch.cat(batch[key], dim=0)

    if "task_desc" in batch:

        batch["task_desc"] = [
            v["task_desc"] for v in data for _ in range(len(v["step_ids"]))
        ]

    return batch


def episode_collate_fn(data: List[Dict]):
    batch = {}

    for key in data[0].keys():
        batch[key] = [v[key] for v in data]

    batch["taskvar_ids"] = torch.LongTensor(batch["taskvar_ids"])
    num_steps = [len(x["rgbs"]) for x in data]
    if "instr_embeds" in batch:
        num_ttokens = [len(x["instr_embeds"]) for x in data]

    for key in ["rgbs", "pcds", "step_ids", "actions"]:
        # e.g. rgbs: (B, T, N, C, H, W)
        batch[key] = pad_tensors(batch[key], lens=num_steps)

    if "instr_embeds" in batch:
        batch["instr_embeds"] = pad_tensors(batch["instr_embeds"], lens=num_ttokens)
        batch["txt_masks"] = torch.from_numpy(gen_seq_masks(num_ttokens))
    else:
        batch["txt_masks"] = torch.ones(len(num_steps), 1).bool()

    batch["step_masks"] = torch.from_numpy(gen_seq_masks(num_steps))

    return batch


if __name__ == "__main__":
    print("Testing KeystepDataset")
    import time
    from torch.utils.data import DataLoader

    data_dir = "/home/imahdi/pytorch_octo/data/keysteps/"
    taskvars = ["reach_target+0"]
    cameras = ["left_shoulder", "right_shoulder", "wrist"]
    print("data_dir: ", data_dir)

    dataset = KeystepDataset(data_dir, taskvars, cameras=cameras, is_training=False)

    data_loader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=stepwise_collate_fn,
        # collate_fn=episode_collate_fn,
    )

    print("Size of the dataset: ", len(dataset))
    print("Number of batchs: ", len(data_loader))

    st = time.time()
    for batch in data_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.size())
        break
    et = time.time()
    print("cost time: %.2fs" % (et - st))
