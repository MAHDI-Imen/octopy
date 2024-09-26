# pyocto
Pytorch Implementation of Octo for RLBench.

To run evaluations on RLBench make sure to look at instructions in [PyRep](https://github.com/stepjam/PyRep) and [RLBench](https://github.com/stepjam/RLBench) to install RLBench simulator (with VirtualGL in headless machines). Use the modified version of [RLBench](https://github.com/rjgpinel/RLBench) to support additional tasks.

To load the pretrained weights, make sure to install octo in your environment. 
Make sure to install the dependencies from the [octo repository](https://github.com/octo-models/octo). Then run the following command to install octo.
    
```bash
pip install git+https://github.com/octo-models/octo
```

## Octo Backbone
You can simply use the Octo backbone with the pretrained weights as follows:

```python
import torch
from pyocto.backbone import OctoBackbone
from pyocto.utils.transfer_weights import load_octo_backbone_weights

backbone = OctoBackbone()
load_octo_backbone_weights(backbone, pretrained_path="hf://rail-berkeley/octo-small-1.5")

B = 2; N = 3; C = 3; H = 256; W = 256 # N: number of cameras

example_batch = {
    "text_input": ["example text instruction"] * B,
    "rgb_obs": torch.randn(B, N, C, H, W),
}

output = backbone(**example_batch)
```

**Note**: If your calculation node does not have internet access. You can first download the files before running the job. Then, specify the folder path as the pretrained_path.

```python
import huggingface_hub
huggingface_repo_id ="rail-berkeley/octo-small-1.5"

folder_path = huggingface_hub.snapshot_download(huggingface_repo_id)
print(folder_path)
load_octo_backbone_weights(backbone, pretrained_path=folder_path)

```


## Custom Model with Octo Backbone
You can use the Octo backbone as a part of your custom model as follows:

```python
class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.backbone = OctoBackbone()
        ...


    def forward(self, text_input, rgb_obs):
        x = self.backbone(text_input, rgb_obs)
        ...
        return x

model = DummyModel()
load_octo_backbone_weights(model, backbone_submodule_name="backbone", pretrained_path="hf://rail-berkeley/octo-small-1.5")
```

## PyOcto model
If you want to use our full model with the action decoder for RLBench dataset, you can use the following code:

```python
import torch
from pyocto.model import PyOcto
from pyocto.utils.transfer_weights import load_octo_backbone_weights

model = PyOcto()
load_octo_backbone_weights(model, backbone_submodule_name="backbone", pretrained_path="hf://rail-berkeley/octo-small-1.5")
B = 2; N = 3; C = 3; H = 256; W = 256 # N: number of cameras
A = 8 # Action dimension
S = 20 # Max number of steps
batch = {
    "rgbs": torch.randn(B, N, C, H, W),
    "pcds": torch.randn(B, N, 3, 256, 256),
    "task_desc": ["pick up the object"] * B,
    "actions": torch.randn(B, A),
    "step_ids": torch.randint(0, S, (2,)),
}

loss, actions = model(batch, compute_loss=True)

print(actions.shape)
print(loss)
``` 

