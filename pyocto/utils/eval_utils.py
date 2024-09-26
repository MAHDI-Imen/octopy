import numpy as np

from pyocto.agent import Agent
from pyocto.environement import run_episode
from pyocto.environement import set_up_rlbench_env, get_task_from_task_name


def evaluate(env, model, seed, NUM_EPISODES, MAX_STEPS, TASKVARS, CAMERAS, RESIZE):
    agent = Agent(model, CAMERAS, resize=RESIZE)
    results = {}
    for task_name in TASKVARS:
        results[task_name] = 0
        task = get_task_from_task_name(env, task_name)
        np.random.seed(seed)
        for episode in range(NUM_EPISODES):
            try:
                reward = run_episode(task, agent, max_steps=MAX_STEPS)
                results[task_name] += reward
            except Exception as e:
                print(e)
                continue
        results[task_name] /= NUM_EPISODES
        results[task_name] *= 100
    return results
