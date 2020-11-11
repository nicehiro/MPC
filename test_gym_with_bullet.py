import time
import argparse
import multiprocessing

import gym
import numpy as np
import pybullet_envs

from pybullet_envs.stable_baselines.utils import TimeFeatureWrapper
import pybullet_envs.bullet.racecarGymEnv as e


if __name__ == '__main__':
    # env = TimeFeatureWrapper(gym.make('KukaBulletEnv-v0'))
    env = e.RacecarGymEnv(isDiscrete=False, renders=True)
    env.render(mode='human')

    try:
        # Use deterministic actions for evaluation
        episode_rewards, episode_lengths = [], []
        while True:
            env.reset()
            while True:
                action = env.action_space.sample()
                obs, reward, done, _info = env.step(action)
                env.render(mode='human')
                dt = 1. / 240.
                time.sleep(dt)
    except KeyboardInterrupt:
        pass

    # Close process
    env.close()
