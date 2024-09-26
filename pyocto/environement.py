from typing import List

import numpy as np

from pyrep.const import RenderMode
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *


def get_task_from_task_name(env: Environment, task_name: str):
    if env._pyrep is None:
        print("Launching the environement")
        env.launch()
    # task_name examples: "reach_target", "reach_target+1", "pick_up_cup"
    if "+" in task_name:
        # in this case the variation is specified
        variation = int(task_name.split("+")[1])
        task_name = task_name.split("+")[0]
    else:
        variation = 0

    task_class = globals()[task_name.replace("_", " ").title().replace(" ", "")]
    task = env.get_task(task_class)
    task.set_variation(variation)

    return task


def set_up_rlbench_env(
    headless: bool = True,
    cameras: List[str] = ["left_shoulder", "right_shoulder", "wrist"],
):
    observation = ObservationConfig()
    observation.set_all(True)
    for cam in cameras:
        getattr(observation, f"{cam}_camera").render_mode = RenderMode.OPENGL
        getattr(observation, f"{cam}_camera").image_size = (128, 128)

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=False),
            gripper_action_mode=Discrete(),
        ),
        obs_config=observation,
        headless=headless,
    )

    return env


def run_episode(task, agent, max_steps):
    descriptions, obs = task.reset()
    instruction = np.random.choice(descriptions)
    for step in range(max_steps):
        action = agent.predict_action(obs, instruction, step)
        obs, reward, terminate = task.step(action.cpu().detach())
        if reward > 0.0:
            break
    return reward


if __name__ == "__main__":
    task_name = "pick_up_cup+9"
    env = set_up_rlbench_env(headless=True)
    task = get_task_from_task_name(env, task_name)
    descriptions, obs = task.reset()
    print(descriptions[0])
    env.shutdown()
