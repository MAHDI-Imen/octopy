import os

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf, DictConfig

import wandb
from tqdm import tqdm

from pyocto.model import PyOcto
from pyocto.utils.eval_utils import evaluate
from pyocto.environement import set_up_rlbench_env
from pyocto.utils.train_utils import (
    convert_params,
    setup_model_training_strategy,
    set_up_optimizer,
    set_up_scheduler,
    set_up_logging,
    train_epoch,
)

from tapas_gmm.behavior_cloning import (
    BCDataConfig,
    DataLoaderConfig,
)
from tapas_gmm.dataset.bc_keypose_cached import BCKeyPoseCachedDataset as Dataset
from tapas_gmm.utils.observation import ObservationConfig, MaskTypes
from conf._machine import data_naming_config

from tapas_gmm.utils.misc import (
    load_scene_data,
)
from tapas_gmm.utils.data_loading import (
    DataLoaderConfig,
    build_infinte_data_iterators,
)
from tapas_gmm.utils.observation import collate


def main(config: DictConfig):
    TRAINING_STRATEGY = config["training"]["strategy"]
    PRETRAINED_PATH = config["training"].get(
        "pretrained_path", "hf://rail-berkeley/octo-small-1.5"
    )
    LR = config["training"]["lr"]
    BACKBONE_LR = config["training"].get("backbone_lr", config["training"]["lr"])
    EPOCHS = config["training"]["epochs"]
    BATCH_SIZE = config["training"]["batch_size"]

    TASK = config["data"]["task"]
    TASKVARS = config["data"]["taskvars"]
    CAMERAS = config["data"]["cameras"]
    TASK_DESC = config["data"]["task_desc"]
    IM_SIZE = config["data"]["image_size"]
    RESIZE = None if IM_SIZE == 256 else (IM_SIZE, IM_SIZE)

    PROJECT_NAME = config["logging"]["project_name"]
    RUN_NAME = config["logging"]["run_name"]
    LOGGING_MODE = config["logging"]["mode"]
    USE_WANDB = config["logging"]["wandb"]

    SAVE_DIR = config["checkpoints"]["save_dir"]
    SAVE_RATE = config["checkpoints"]["save_rate"]

    EVALUATE = config["evaluation"]["evaluate"]
    EVALUATION_RATE = config["evaluation"]["evaluation_rate"]
    EVALUATION_SEED = config["evaluation"]["seed"]
    NUM_EPISODES = config["evaluation"]["num_episodes"]
    MAX_STEPS = config["evaluation"]["max_steps"]

    ############################################################################
    # Setup Environment
    ############################################################################
    if EVALUATE:
        env = set_up_rlbench_env(headless=True, cameras=CAMERAS)
        env.launch()

    ############################################################################
    # Setup device
    ############################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}", flush=True)
    print(f"Training strategy: {TRAINING_STRATEGY}", flush=True)

    ############################################################################
    # Load the model and checkpoint
    ############################################################################
    print(f"Loading model", flush=True)
    model = PyOcto()
    setup_model_training_strategy(model, TRAINING_STRATEGY, PRETRAINED_PATH)
    model.to(device)

    print(f"Total params: {convert_params(model.num_params)}", flush=True)
    print(f"Trainable params: {convert_params(model.num_trainable_params)}", flush=True)
    print(
        f"Non-trainable params: {convert_params(model.num_frozen_params)}", flush=True
    )

    ############################################################################
    # Load the dataset
    ############################################################################
    print("Loading dataset", flush=True)

    data_naming = data_naming_config
    data_naming.task = config["data"]["task"]
    data_naming.feedback_type = config["data"]["feedback_type"]
    scene_data = load_scene_data(data_naming)

    bc_data = BCDataConfig(
        fragment_length=1,
        cameras=CAMERAS,
        mask_type=MaskTypes.GT,
        subsample_to_common_length=True,
    )
    bc_data = Dataset(scene_data, bc_data)

    data_loader = DataLoader(
        bc_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )

    print(f"Number of keypoints: {len(bc_data)}", flush=True)
    print(f"Number of batches: {len(data_loader)}", flush=True)

    ############################################################################
    # Setup the optimizer
    ############################################################################
    optimizer = set_up_optimizer(model, TRAINING_STRATEGY, LR, BACKBONE_LR)
    scheduler = set_up_scheduler(optimizer, EPOCHS)

    ############################################################################
    # Logging
    ############################################################################
    print(f"Project: {PROJECT_NAME} | Run: {RUN_NAME}", flush=True)
    if USE_WANDB:
        set_up_logging(config, model, PROJECT_NAME, RUN_NAME, LOGGING_MODE)

    ############################################################################
    # Checkpointing
    ############################################################################
    if not os.path.exists(f"{SAVE_DIR}/{RUN_NAME}"):
        os.makedirs(f"{SAVE_DIR}/{RUN_NAME}")

    ############################################################################
    # Training loop
    ############################################################################
    print("Starting training", flush=True)
    pbar = tqdm(range(1, EPOCHS + 1), total=EPOCHS)

    for epoch in pbar:
        logs = {}
        average_losses = train_epoch(model, optimizer, data_loader, TASK_DESC, CAMERAS)
        logs.update(average_losses)
        pbar.set_description(f"Epoch {epoch} | Loss: {average_losses['total']:.4f}")

        scheduler.step()
        logs.update({"lr": scheduler.get_last_lr()[0]})

        if EVALUATE and epoch % EVALUATION_RATE == 0:
            results = evaluate(
                env,
                model,
                EVALUATION_SEED,
                NUM_EPISODES,
                MAX_STEPS,
                TASKVARS,
                CAMERAS,
                RESIZE,
                TASK_DESC,
            )
            for task_name, result in results.items():
                logs.update({task_name: result})

        if USE_WANDB:
            wandb.log(logs, step=epoch)

        if epoch % SAVE_RATE == 0:
            torch.save(
                model.state_dict(), f"{SAVE_DIR}/{RUN_NAME}/checkpoint_{epoch}.pt"
            )

    if EVALUATE:
        env.shutdown()


if __name__ == "__main__":
    config_path = "pyocto/scripts/configs/config.yaml"
    config = OmegaConf.load(config_path)
    main(config)
