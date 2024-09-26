from typing import Union

import torch

import jax
import numpy as np
import jax.numpy as jnp
from flax.traverse_util import flatten_dict

from pyocto.backbone import OctoBackbone


#####################
# Flax => PyTorch #
#####################
def load_flax_weights_in_pytorch_model(pt_model, flax_state):
    """Load flax checkpoints in a PyTorch model"""
    # check if we have bf16 weights
    is_type_bf16 = flatten_dict(
        jax.tree_util.tree_map(lambda x: x.dtype == jnp.bfloat16, flax_state)
    ).values()
    if any(is_type_bf16):
        # convert all weights to fp32 if the are bf16 since torch.from_numpy can-not handle bf16
        # and bf16 is not fully supported in PT yet.
        print(
            "Found ``bfloat16`` weights in Flax model. Casting all ``bfloat16`` weights to ``float32`` "
            "before loading those in PyTorch model."
        )
        flax_state = jax.tree_util.tree_map(
            lambda params: (
                params.astype(np.float32) if params.dtype == jnp.bfloat16 else params
            ),
            flax_state,
        )

    flax_state_dict = flatten_dict(flax_state)
    pt_model_dict = pt_model.state_dict()
    base_model_prefix = ""
    load_model_with_head_into_base_model = (base_model_prefix in flax_state) and (
        base_model_prefix not in {k.split(".")[0] for k in pt_model_dict.keys()}
    )
    load_base_model_into_model_with_head = (base_model_prefix not in flax_state) and (
        base_model_prefix in {k.split(".")[0] for k in pt_model_dict.keys()}
    )

    # keep track of unexpected & missing keys
    unexpected_keys = []
    missing_keys = set(pt_model_dict.keys())

    for flax_key_tuple, flax_tensor in flax_state_dict.items():
        has_base_model_prefix = flax_key_tuple[0] == base_model_prefix
        require_base_model_prefix = (
            ".".join((base_model_prefix,) + flax_key_tuple) in pt_model_dict
        )

        # adapt flax_key to prepare for loading from/to base model only
        if load_model_with_head_into_base_model and has_base_model_prefix:
            flax_key_tuple = flax_key_tuple[1:]
        elif load_base_model_into_model_with_head and require_base_model_prefix:
            flax_key_tuple = (base_model_prefix,) + flax_key_tuple

        # rename flax weights to PyTorch format
        if len(flax_key_tuple) >= 2 and flax_key_tuple[-2] == "out":
            flax_key_tuple = flax_key_tuple[:-2] + ("out_proj", flax_key_tuple[-1])
            if flax_tensor.ndim == 2:
                flax_tensor = flax_tensor.flatten()
            elif flax_tensor.ndim == 3:
                flax_tensor = flax_tensor.reshape(-1, flax_tensor.shape[2])

        if (
            flax_key_tuple[-1] == "kernel"
            and flax_tensor.ndim == 4
            and ".".join(flax_key_tuple) not in pt_model_dict
        ):
            # conv layer
            flax_key_tuple = flax_key_tuple[:-1] + ("weight",)
            flax_tensor = jnp.transpose(flax_tensor, (3, 2, 0, 1))

        elif (
            flax_key_tuple[-1] == "kernel"
            and flax_tensor.ndim == 2
            and ".".join(flax_key_tuple) not in pt_model_dict
        ):
            # linear layer
            flax_key_tuple = flax_key_tuple[:-1] + ("weight",)
            flax_tensor = flax_tensor.T
        elif flax_key_tuple[-1] in ["scale", "embedding"]:
            flax_key_tuple = flax_key_tuple[:-1] + ("weight",)

        # adding batch stats from flax batch norm to pt
        elif "mean" in flax_key_tuple[-1]:
            flax_key_tuple = flax_key_tuple[:-1] + ("running_mean",)
        elif "var" in flax_key_tuple[-1]:
            flax_key_tuple = flax_key_tuple[:-1] + ("running_var",)

        if "batch_stats" in flax_state:
            flax_key = ".".join(
                flax_key_tuple[1:]
            )  # Remove the params/batch_stats header
        else:
            flax_key = ".".join(flax_key_tuple)

        # We also need to look at `pt_model_dict` and see if there are keys requiring further transformation.
        special_pt_names = {}
        # New `weight_norm` from https://github.com/huggingface/transformers/pull/24030
        for key in pt_model_dict:
            key_components = key.split(".")
            name = None
            if key_components[-3::2] == ["parametrizations", "original0"]:
                name = key_components[-2] + "_g"
            elif key_components[-3::2] == ["parametrizations", "original1"]:
                name = key_components[-2] + "_v"
            if name is not None:
                key_components = key_components[:-3] + [name]
                key_to_check = ".".join(key_components)
                special_pt_names[key_to_check] = key

        if flax_key in special_pt_names:
            flax_key = special_pt_names[flax_key]

        if flax_key in pt_model_dict:
            if flax_tensor.shape != pt_model_dict[flax_key].shape:
                raise ValueError(
                    f"Flax checkpoint seems to be incorrect. Weight {flax_key_tuple} was expected "
                    f"to be of shape {pt_model_dict[flax_key].shape}, but is {flax_tensor.shape}."
                )
            else:
                # add weight to pytorch dict
                flax_tensor = (
                    np.asarray(flax_tensor)
                    if not isinstance(flax_tensor, np.ndarray)
                    else flax_tensor
                )
                pt_model_dict[flax_key] = torch.tensor(flax_tensor)
                # remove from missing keys
                missing_keys.remove(flax_key)
        else:
            # weight is not expected by PyTorch model
            unexpected_keys.append(flax_key)

        # deal with MultiHeadAttentionLayers Seperately
        # filter from missing keys the ones that have in_proj and out_proj in them
        in_proj_attention_layers = [k for k in missing_keys if "in_proj" in k]

        for in_proj_layer in in_proj_attention_layers:
            layer_name = in_proj_layer.split(".")[:-1]
            if "bias" in in_proj_layer:
                # take all the name previous to the .
                q_key = tuple(layer_name + ["query", "bias"])
                k_key = tuple(layer_name + ["key", "bias"])
                v_key = tuple(layer_name + ["value", "bias"])
                if (
                    ".".join(q_key) in unexpected_keys
                    and ".".join(k_key) in unexpected_keys
                    and ".".join(v_key) in unexpected_keys
                ):
                    in_proj_bias = np.concatenate(
                        [
                            flax_state_dict[q_key].flatten(),
                            flax_state_dict[k_key].flatten(),
                            flax_state_dict[v_key].flatten(),
                        ],
                        axis=0,
                    )

                    pt_model_dict[in_proj_layer] = torch.from_numpy(in_proj_bias)
                    missing_keys.remove(in_proj_layer)
                    unexpected_keys.remove(".".join(q_key))
                    unexpected_keys.remove(".".join(k_key))
                    unexpected_keys.remove(".".join(v_key))

            elif "weight" in in_proj_layer:
                q_key = tuple(layer_name + ["query", "kernel"])
                k_key = tuple(layer_name + ["key", "kernel"])
                v_key = tuple(layer_name + ["value", "kernel"])

                if (
                    ".".join(q_key) in unexpected_keys
                    and ".".join(k_key) in unexpected_keys
                    and ".".join(v_key) in unexpected_keys
                ):
                    embed_dim = flax_state_dict[q_key].shape[0]
                    in_proj_weight = np.concatenate(
                        [
                            flax_state_dict[q_key].reshape(embed_dim, -1).T,
                            flax_state_dict[k_key].reshape(embed_dim, -1).T,
                            flax_state_dict[v_key].reshape(embed_dim, -1).T,
                        ],
                        axis=0,
                    )

                    pt_model_dict[in_proj_layer] = torch.from_numpy(in_proj_weight)

                    missing_keys.remove(in_proj_layer)
                    unexpected_keys.remove(".".join(q_key))
                    unexpected_keys.remove(".".join(k_key))
                    unexpected_keys.remove(".".join(v_key))

    pt_model.load_state_dict(pt_model_dict)

    # re-transform missing_keys to list
    missing_keys = list(missing_keys)

    if len(unexpected_keys) > 0:
        print(
            "Some weights of the Flax model were not used when initializing the PyTorch model"
            f" {pt_model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing"
            f" {pt_model.__class__.__name__} from a Flax model trained on another task or with another architecture"
            " (e.g. initializing a BertForSequenceClassification model from a FlaxBertForPreTraining model).\n- This"
            f" IS NOT expected if you are initializing {pt_model.__class__.__name__} from a Flax model that you expect"
            " to be exactly identical (e.g. initializing a BertForSequenceClassification model from a"
            " FlaxBertForSequenceClassification model)."
        )
    else:
        print(
            f"All Flax model weights were used when initializing {pt_model.__class__.__name__}.\n"
        )
    if len(missing_keys) > 0:
        print(
            f"Some weights of {pt_model.__class__.__name__} were not initialized from the Flax model and are newly"
            f" initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to"
            " use it for predictions and inference."
        )
    else:
        print(
            f"All the weights of {pt_model.__class__.__name__} were initialized from the Flax model.\n"
        )

    return pt_model


def get_octo_flax_params(pretrained_path: str = "hf://rail-berkeley/octo-small-1.5"):
    try:
        from octo.model.octo_model import OctoModel
    except ImportError:
        print(
            "Jax Octo module not found. Please install octo from https://github.com/octo-models/octo"
        )
    jax_model = OctoModel.load_pretrained(pretrained_path)
    jax_params = jax_model.params
    # A mapping to adjust the names of parameters in the JAX model to the PyTorch model allowing for more flexibility in the model architecture.
    # Note: we only use the primary tokenizer in our case. The wrist tokenizer can be added but the backbone architecture should be adjusted accordingly.
    jax_weigths = {
        "primary_tokenizer": jax_params["octo_transformer"][
            "observation_tokenizers_primary"
        ]["SmallStem16_0"],
        # "wrist_tokenizer": jax_params["octo_transformer"]["observation_tokenizers_wrist"]["SmallStem16_0"],
        "task_tokenizer.encoder": jax_params["octo_transformer"][
            "task_tokenizers_language"
        ]["hf_model"],
        "transformer": jax_params["octo_transformer"]["BlockTransformer_0"][
            "Transformer_0"
        ],
        "task_language_projection": jax_params["octo_transformer"][
            "task_language_projection"
        ],
        "obs_primary_projection": jax_params["octo_transformer"][
            "obs_primary_projection"
        ],
        # "obs_wrist_projection": jax_params["octo_transformer"]["obs_wrist_projection"],
        "task_language_pos_embedding": jax_params["octo_transformer"][
            "task_language_pos_embedding"
        ],
        "obs_primary_pos_embedding": jax_params["octo_transformer"][
            "obs_primary_pos_embedding"
        ],
        # "obs_wrist_pos_embedding": jax_params["octo_transformer"]["obs_wrist_pos_embedding"],
        "readout_action_pos_embedding": jax_params["octo_transformer"][
            "readout_action_pos_embedding"
        ],
    }
    return jax_weigths


def load_octo_backbone_weights(
    model: Union[torch.nn.Module, OctoBackbone],
    backbone_submodule_name: str = None,
    **kwargs,
):
    """
    Load Octo model weights in a OctoBackbone module or PyTorch model with OctoBackbone as submodule.
    Args:
        model: PyTorch model with OctoBackbone as submodule or OctoBackbone class
        backbone_submodule_name: Name of the OctoBackbone submodule in the model
    Returns:
        model: PyTorch model with Octo model weights loaded
    Example usage:
        class Model(nn.Module):
            def __init__(self):
                self.backbone = OctoBackbone()
                ...

            def forward(input):
                ...
                # Check backbone implementation for details on input and output.
                output = self.backbone(text_input, rgb_obs, device)
                ...

        model = Model()
        model = load_octo_backbone_weights(model, "backbone")
    """

    jax_params = get_octo_flax_params(**kwargs)

    if isinstance(model, OctoBackbone):
        return load_flax_weights_in_pytorch_model(model, jax_params)

    assert (
        backbone_submodule_name is not None
    ), "Model is not an OctoBackbone instance, but backbone_submodule_name is not provided."

    assert backbone_submodule_name in [
        n for (n, _) in model.named_children()
    ], f"Backbone submodule name {backbone_submodule_name} not found in model state dict keys. Did you provide the correct module name?"

    assert isinstance(
        model.get_submodule(backbone_submodule_name), OctoBackbone
    ), f"Backbone submodule is not an instance of OctoBackbone."

    backbone = model.get_submodule(backbone_submodule_name)
    load_flax_weights_in_pytorch_model(backbone, jax_params)

    return model
