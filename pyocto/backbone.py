from typing import Dict

import torch
import einops
import torch.nn as nn

from pyocto.backbone_components.transformer import Transformer
from pyocto.backbone_components.tokenizers import SmallStem, TaskTokenizer


class OctoBackbone(nn.Module):
    def __init__(self, config: Dict = None) -> None:
        super().__init__()
        if config is None:
            config = self.default_config()
        # Tokenizers
        self.task_tokenizer = TaskTokenizer(config["language_tokenizer_kwargs"])
        self.primary_tokenizer = SmallStem(**config["image_tokenizer_kwargs"])

        # Transformer Backbone
        self.transformer = Transformer(**config["transformer_kwargs"])

        self.token_embedding_size = config["transformer_kwargs"]["input_dim"]
        self.latent_im_size = config["latent_im_size"]
        task_max_length = config["language_tokenizer_kwargs"]["max_length"]
        max_horizon = config["max_horizon"]
        image_tokenizer_latent_dim = self.primary_tokenizer.num_features

        # projection layers
        self.task_language_projection = nn.Linear(768, self.token_embedding_size)
        self.obs_primary_projection = nn.Linear(
            image_tokenizer_latent_dim, self.token_embedding_size
        )
        # positional embeddings
        self.task_language_pos_embedding = nn.Parameter(
            torch.randn(1, task_max_length, self.token_embedding_size)
        )

        self.obs_primary_pos_embedding = nn.Parameter(
            torch.randn(1, max_horizon, self.latent_im_size, self.token_embedding_size)
        )

        self.readout_action_pos_embedding = nn.Parameter(
            torch.randn(1, max_horizon, 1, self.token_embedding_size)
        )

    def forward(
        self,
        text_input: list[str],
        rgb_obs: torch.Tensor,  # (B, N, C, H, W) N: #cameras
    ):
        """
        Returns the embeddings for the task, visual observations and readout action embedding as well as the encoded visual features and the new image height and width (visual tokens are flattened to a sequence of length new_height * new_width).
        """

        device = rgb_obs.device
        batch_size, n_cameras = rgb_obs.shape[:2]
        # Tokenize all input
        task_tokens = self.task_tokenizer(text_input, device)
        task_tokens = self.task_language_projection(task_tokens)
        task_tokens += self.task_language_pos_embedding.expand(batch_size, -1, -1)

        rgb_fts = einops.rearrange(rgb_obs, "b n c h w -> (b n) c h w")
        # pad images channel wise to 6 channels
        rgb_fts = torch.cat([rgb_fts, torch.zeros_like(rgb_fts)], dim=1)

        obs_tokens, enc_fts = self.primary_tokenizer(rgb_fts)
        new_im_height, new_im_width = obs_tokens.shape[-2:]
        obs_tokens = einops.rearrange(
            obs_tokens, "(b n) c h w -> (b n) (h w) c", b=batch_size, n=n_cameras
        )
        obs_tokens = self.obs_primary_projection(obs_tokens)

        obs_tokens += self.obs_primary_pos_embedding[:, 0].expand(
            batch_size * n_cameras, -1, -1
        )

        obs_tokens = einops.rearrange(
            obs_tokens,
            "(b n) (h w) c -> b (n h w) c",
            b=batch_size,
            n=n_cameras,
            h=new_im_height,
            w=new_im_width,
        )

        readout_tokens = torch.zeros(batch_size, 1, self.token_embedding_size).to(
            device
        )
        readout_tokens += self.readout_action_pos_embedding[:, 0].expand(
            batch_size, -1, -1
        )

        all_tokens = torch.cat([task_tokens, obs_tokens, readout_tokens], dim=1)

        n_ttokens = task_tokens.shape[1]
        n_vtokens = obs_tokens.shape[1]
        n_rtokens = readout_tokens.shape[1]

        attention_mask = self.gen_attention_mask(n_ttokens, n_vtokens, n_rtokens).to(
            device
        )

        outputs = self.transformer(all_tokens, attention_mask=attention_mask)

        task_embeddings = outputs[:, :n_ttokens]
        visual_embeddings = outputs[:, n_ttokens:-n_rtokens]
        readout_embeddings = outputs[:, -n_rtokens:]

        visual_embeddings = einops.rearrange(
            visual_embeddings,
            "b (n h w) c -> b n (h w) c",
            n=n_cameras,
            h=new_im_height,
            w=new_im_width,
        )

        return {
            "task_embeddings": task_embeddings,
            "visual_embeddings": visual_embeddings,
            "readout_embeddings": readout_embeddings,
            "enc_fts": enc_fts,
            "new_im_height": new_im_height,
            "new_im_width": new_im_width,
        }

    def gen_attention_mask(
        self,
        n_ttokens: int,
        n_vtokens: int,
        n_rtokens: int,
    ):
        """
        Generate a proper 2D attention mask in the case of one timestep.

        Args:
            n_ttokens: number of task tokens
            n_vtokens: number of visual tokens
            n_rtokens: number of readout tokens
        """
        n_total = n_ttokens + n_vtokens + n_rtokens
        mask = torch.zeros(n_total, n_total).bool()
        # task tokens attend only to themselves
        mask[:n_ttokens, :n_ttokens] = True
        # obs tokens attend to all previous tokens and themselves
        mask[n_ttokens : n_ttokens + n_vtokens, : n_ttokens + n_vtokens] = True
        # readout tokens attend to everything
        mask[-n_rtokens:, :] = True
        return mask

    @property
    def decoder_features(self) -> tuple[list[int], int]:
        """
        Returns the num of features for each deconv layer in the image decoder and the number of output features of the final layer.
        """
        features = list(self.primary_tokenizer.features)
        out_features = self.primary_tokenizer.num_features
        features.reverse()
        features.append(features[-1])
        return features, out_features

    def default_config(self):
        """
        Default configuration for the Octo backbone.
        """
        # can be obtained directly from the jax model config with some modification as follows:

        ## from octo.model.octo_model import OctoModel
        ## model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

        ## tokenizer_kwargs = model.config["text_processor"]["kwargs"]["tokenizer_kwargs"]
        ## tokenizer_kwargs["return_tensors"] = "pt"

        ## token_embedding_size = model.config["model"]["token_embedding_size"]

        ## transformer_kwargs = model.config["model"]["transformer_kwargs"]
        ## transformer_kwargs["input_dim"] = token_embedding_size

        ## max_horizon = model.config["model"]["max_horizon"]

        ## backbone_config = {
        ##     "image_tokenizer_kwargs": {"patch_size": 16},
        ##     "tokenizer_kwargs": tokenizer_kwargs,
        ##     "transformer_kwargs": transformer_kwargs,
        ##     "latent_im_size": 256,
        ##     "max_horizon": max_horizon,
        ## }

        backbone_config = {
            "image_tokenizer_kwargs": {"patch_size": 16},
            "language_tokenizer_kwargs": {
                "max_length": 16,
                "padding": "max_length",
                "return_tensors": "pt",
                "truncation": True,
            },
            "transformer_kwargs": {
                "input_dim": 384,
                "add_position_embedding": False,
                "attention_dropout_rate": 0.0,
                "dropout_rate": 0.0,
                "mlp_dim": 1536,
                "num_attention_heads": 6,
                "num_layers": 12,
            },
            "latent_im_size": 256,
            "max_horizon": 10,
        }
        return backbone_config


if __name__ == "__main__":
    model = OctoBackbone()
    B = 2  # batch size
    N = 3  # number of cameras
    C = 3  # number of channels
    H = 256  # height
    W = 256  # width

    example_batch = {
        "text_input": ["example text"] * B,
        "rgb_obs": torch.randn(B, N, C, H, W),
    }

    output = model(**example_batch)

    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        elif isinstance(v, list):
            print(k)
            for i, e in enumerate(v):
                print(f"\t ({i})", e.shape)
        else:
            print(k, v)
