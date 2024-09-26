import torch
import torch.nn as nn
from transformers import AutoConfig, T5EncoderModel, AutoTokenizer


# #############################################################################
# Vision Tokenizers
# #############################################################################
def normalize_images(img, img_norm_type="default"):
    if img_norm_type == "default":
        # put pixels in [-1, 1]
        return img.float() / 127.5 - 1.0
    elif img_norm_type == "imagenet":
        # put pixels in [0, 1]
        img = img.float() / 255
        assert img.shape[1] % 3 == 0, "images should have RGB channels!"

        # define pixel-wise mean/std stats calculated from ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        std = torch.tensor([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

        # normalize image
        return (img - mean) / std
    raise ValueError("Unsupported img_norm_type!")


def weight_standardize(weight, axis, eps=1e-5):
    mean = weight.mean(dim=axis, keepdim=True)
    std = weight.std(dim=axis, keepdim=True, unbiased=False)
    return (weight - mean) / (std + eps)


class StdConv(nn.Conv2d):
    """Convolution with weight standardization."""

    def __init__(
        self,
        in_channels,
        features,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        device=None,
        dtype=None,
    ):
        super(StdConv, self).__init__(
            in_channels,
            features,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        # Apply weight standardization to the kernel
        weight = weight_standardize(self.weight, axis=[1, 2, 3], eps=1e-5)
        return nn.functional.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class SmallStem(nn.Module):
    def __init__(
        self,
        patch_size=32,
        kernel_sizes=(3, 3, 3, 3),
        strides=(2, 2, 2, 2),
        features=(32, 96, 192, 384),
        padding=(1, 1, 1, 1),
        num_features=512,
        img_norm_type="default",
        **kwargs,
    ):
        super(SmallStem, self).__init__()
        self.patch_size = patch_size
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.features = features
        self.padding = padding
        self.num_features = num_features
        self.img_norm_type = img_norm_type

        num_groups = kwargs.get("num_groups", 32)
        for i, (k, s, f, p) in enumerate(
            zip(self.kernel_sizes, self.strides, self.features, self.padding)
        ):
            in_channels = 6 if i == 0 else self.features[i - 1]
            self.add_module(
                f"StdConv_{i}",
                StdConv(
                    in_channels=in_channels,
                    features=f,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                ),
            )
            self.add_module(
                f"GroupNorm_{i}", nn.GroupNorm(num_groups=num_groups, num_channels=f)
            )
            self.add_module(f"ReLU_{i}", nn.ReLU(inplace=True))

        self.embedding = nn.Conv2d(
            self.features[-1],
            self.num_features,
            kernel_size=self.patch_size // 16,
            stride=self.patch_size // 16,
            padding=0,
        )

    def forward(self, observations):
        x = normalize_images(observations, self.img_norm_type)
        enc_fts = []
        for i in range(len(self.features)):
            x = self._modules[f"StdConv_{i}"](x)
            x = self._modules[f"GroupNorm_{i}"](x)
            x = self._modules[f"ReLU_{i}"](x)
            enc_fts.append(x)
        x = self.embedding(x)
        enc_fts.append(x)
        return x, enc_fts


# #############################################################################
# Language Tokenizers
# #############################################################################
class TaskTokenizer(nn.Module):
    def __init__(self, tokenizer_kwargs):
        super(TaskTokenizer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.encoder = T5EncoderModel(AutoConfig.from_pretrained("t5-base"))
        self.tokenizer_kwargs = tokenizer_kwargs

    def forward(self, text_inputs: list[str], device=None):
        text_tokens = self.tokenizer(text_inputs, **self.tokenizer_kwargs)
        if device:
            text_tokens = text_tokens.to(device)

        task_tokens = self.encoder(**text_tokens).last_hidden_state
        return task_tokens
