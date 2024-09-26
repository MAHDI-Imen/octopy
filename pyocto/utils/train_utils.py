import torch
from torch.optim import AdamW
from omegaconf import OmegaConf
from pyocto.utils.transfer_weights import load_octo_backbone_weights


def convert_params(num_params):
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}G"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def setup_model_training_strategy(
    model: torch.nn.Module,
    training_strategy: str,
    pretrained_path: str,
) -> None:
    """
    Preapre model parameters according to the given training strategy.
    Args:
        model: Pytorch model that has the OctoBackbone as a submodule.
        training_strategy:
            - "scratch": Train all weights except the language transformer
            - "action_head": Freeze the backbone weights and train only the action head.
            - "finetune": Train all weights except the language transformer, with a lower learning rate for the backbone
    """
    if training_strategy in ["finetune", "action_head"]:
        print("Loading checkpoint", flush=True)
        load_octo_backbone_weights(model, "backbone", pretrained_path=pretrained_path)

    if training_strategy in ["scratch", "finetune"]:
        model.unfreeze("all")
    elif training_strategy == "action_head":
        model.freeze("backbone")


def set_up_optimizer(model, training_strategy, lr, backbone_lr):
    if training_strategy == "finetune":
        backbone_params = [p for n, p in model.backbone.named_parameters()]
        rest_params = [
            p for n, p in model.named_parameters() if not n.startswith("backbone")
        ]

        optimizer = AdamW(
            [
                {"params": rest_params, "lr": lr},
                {"params": backbone_params, "lr": backbone_lr},
            ]
        )
    else:
        optimizer = AdamW(model.parameters(), lr=lr)

    return optimizer


def set_up_scheduler(optimizer, epochs):
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, end_factor=1, total_iters=20
    )
    cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_annealing_scheduler],
        milestones=[20],
    )

    return scheduler


def set_up_logging(config, model, PROJECT_NAME, RUN_NAME, LOGGING_MODE):
    import wandb

    wandb.init(
        project=PROJECT_NAME,
        name=RUN_NAME,
        config=OmegaConf.to_container(config, resolve=True),
        mode=LOGGING_MODE,
    )
    wandb.watch(model)


def train_epoch(model, optimizer, data_loader):
    average_losses = {}
    model.train()

    for batch in data_loader:
        losses, actions = model(batch, compute_loss=True)
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        for k in losses.keys():
            average_losses[k] = average_losses.get(k, 0) + losses[k].item()

    for k in average_losses.keys():
        average_losses[k] /= len(data_loader)

    return average_losses
