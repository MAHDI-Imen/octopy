import torch


def extract_input(batch, task_desc, cameras):
    """
    Extracts the input data from the batch for the model.
    """
    rgbs = []
    pcds = []
    for cam in cameras:
        rgbs.append(batch.cameras[cam]["rgb"])
        pcds.append(batch.cameras[cam]["pcd"])

    rgbs = torch.stack(rgbs, dim=1)
    pcds = torch.stack(pcds, dim=1)
    # convert to float32
    rgbs = rgbs.to(torch.float32)
    pcds = pcds.to(torch.float32)

    actions = batch.action
    # convert to float32
    actions = actions.to(torch.float32)
    task_desc = [task_desc] * rgbs.shape[0]
    batch = {
        "task_desc": task_desc,
        "rgbs": rgbs,
        "pcds": pcds,
        "actions": actions,
    }
    return batch
