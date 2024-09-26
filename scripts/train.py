import os

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf, DictConfig

import wandb
from tqdm import tqdm

from pyocto.model import PyOcto
from pyocto.utils.eval_utils import evaluate
from pyocto.environement import set_up_rlbench_env
from pyocto.data.dataset import KeystepDataset, stepwise_collate_fn
from pyocto.utils.train_utils import (
    convert_params,
    setup_model_training_strategy,
    set_up_optimizer,
    set_up_scheduler,
    set_up_logging,
    train_epoch,
)


def main(config: DictConfig):
    TRAINING_STRATEGY = config["training"]["strategy"]
    PRETRAINED_PATH = config["training"].get(
        "pretrained_path", "hf://rail-berkeley/octo-small-1.5"
    )
    LR = config["training"]["lr"]
    BACKBONE_LR = config["training"].get("backbone_lr", config["training"]["lr"])
    EPOCHS = config["training"]["epochs"]
    BATCH_SIZE = config["training"]["batch_size"]

    DATA_DIR = config["data"]["data_dir"]
    TASKVARS = config["data"]["taskvars"]
    CAMERAS = config["data"]["cameras"]
    TASK_DESC = config["data"]["task_desc"]
    IM_SIZE = config["data"]["image_size"]
    RESIZE = None if IM_SIZE == 128 else (IM_SIZE, IM_SIZE)

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
        env = set_up_rlbench_env(headless=True)
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
    dataset = KeystepDataset(
        DATA_DIR,
        TASKVARS,
        cameras=CAMERAS,
        is_training=True,
        task_desc=TASK_DESC,
        resize=RESIZE,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=stepwise_collate_fn,
        shuffle=True,
    )

    print(f"Number of keypoints: {len(dataset)}", flush=True)
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
        average_losses = train_epoch(model, optimizer, data_loader)
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
    config_path = "scripts/configs/config.yaml"
    config = OmegaConf.load(config_path)
    main(config)
