import torch
from pyocto.environement import set_up_rlbench_env
from pyocto.model import PyOcto
from pyocto.utils.eval_utils import evaluate


def validate(
    checkpoint_path, TASKVARS, CAMERAS, RESIZE, seed=200, num_episodes=10, max_steps=20
):
    env = set_up_rlbench_env(headless=False)

    model = PyOcto()
    model.load_state_dict(torch.load(checkpoint_path))

    results = evaluate(
        env, model, seed, num_episodes, max_steps, TASKVARS, CAMERAS, RESIZE
    )
    return results


if __name__ == "__main__":
    checkpoint_path = "/path/to/chrckpoint.pt"
    TASKVARS = ["pick_up_cup+0"]
    CAMERAS = ["left_shoulder", "right_shoulder", "wrist"]
    RESIZE = (256, 256)
    results = validate(checkpoint_path, TASKVARS, CAMERAS, RESIZE)
    print(results)
