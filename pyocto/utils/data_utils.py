import torch


def extract_input(batch, task_desc):
    """
    Extracts the input data from the batch for the model.
    """

    wrist_rgb = batch.cameras["wrist"]["rgb"]
    wrist_pcd = batch.cameras["wrist"]["pcd"]
    left_shoulder_rgb = batch.cameras["left_shoulder"]["rgb"]
    left_shoulder_pcd = batch.cameras["left_shoulder"]["pcd"]
    right_shoulder_rgb = batch.cameras["right_shoulder"]["rgb"]
    right_shoulder_pcd = batch.cameras["right_shoulder"]["pcd"]

    rgbs = torch.stack(
        [
            left_shoulder_rgb,
            right_shoulder_rgb,
            wrist_rgb,
        ],
        dim=1,
    )
    pcds = torch.stack(
        [
            left_shoulder_pcd,
            right_shoulder_pcd,
            wrist_pcd,
        ],
        dim=1,
    )
    # convert to float32
    rgbs = rgbs.to(torch.float32)
    pcds = pcds.to(torch.float32)

    actions = batch.action
    # convert to float32
    actions = actions.to(torch.float32)
    batch = {
        "task_desc": [task_desc] * rgbs.shape[0],
        "rgbs": rgbs,
        "pcds": pcds,
        "actions": actions,
    }

    return batch
