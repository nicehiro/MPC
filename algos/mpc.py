import gym
from gym import Env

import numpy as np
import torch

from sample_policies.random_sample import RandomSample
from sample_policies.cem_sample import CEMSample
from models.dynamic_model import DynamicModel
from models.reward_model import RewardModel


class MPC:
    samplers = {"Random": RandomSample, "CEM": CEMSample}

    def __init__(self, config, dynamic, rewards):
        mpc_config = config
        self.is_cem = True if mpc_config["sampler"] == "CEM" else False
        self.dynamic = dynamic
        self.rewards = rewards
        self.timesteps = mpc_config["timesteps"]
        self.sample_times = mpc_config["sample_times"]
        self.num_elites = mpc_config["num_elites"]
        self.upper_bound = mpc_config["upper_bound"]
        self.lower_bound = mpc_config["lower_bound"]
        self.action_dim = mpc_config["action_dim"]
        self.is_discrete = mpc_config["is_discrete"]
        self.sampler = self.samplers[mpc_config["sampler"]](
            self.upper_bound, self.lower_bound, self.action_dim, self.is_discrete
        )
        # mu is mean and sigma is 0 at the first timestep
        self.mu = (self.lower_bound + self.upper_bound) / 2
        self.sigma = np.ones_like(self.mu)
        self.alpha = mpc_config["alpha"]

    def act(self, state):
        """
        Get first action from best plan.

        @param state: first state
        :return: action
        """
        best_plan = self.plan(state)
        return best_plan[0]

    def _calc_rewards(self, first_state, actions):
        """
        Calc rewards of trajectories.

        @param first_state: first state
        @param actions: sample_times * timesteps * action_dim
        :return: rewards sample_times * 1
        """
        rewards = []
        for actions_traj in actions:
            state = first_state
            rewards_of_traj = 0
            for action in actions_traj:
                next_state = self.dynamic.predict(state, action) + state
                reward = self.rewards.predict(state, action)
                state = next_state
                rewards_of_traj += reward
            rewards.append(rewards_of_traj)
        return np.array(rewards)

    def plan(self, state):
        # actions shape: sample_times * timesteps * self.action_dim (1000 * 15 * 1)
        actions = self.sampler.sample_n(
            sample_nums=self.sample_times,
            timesteps=self.timesteps,
            mu=self.mu,
            sigma=self.sigma,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
        )
        rewards = self._calc_rewards(state, actions)
        best_plan = actions[np.argmax(rewards)]
        # update mu and mean
        if self.is_cem:
            idx = np.argsort(rewards)
            elites = actions[idx.squeeze()][: self.num_elites]
            new_mean = torch.mean(elites, dim=1, keepdim=True)
            new_mean = torch.mean(new_mean, dim=0, keepdim=True)
            new_var = torch.var(elites, dim=1, keepdim=True)
            new_var = torch.var(new_var, dim=0, keepdim=True)
            self.mu = (
                self.alpha * self.mu
                + (1 - self.alpha) * torch.squeeze(new_mean).numpy()
            )
            self.sigma = (
                self.alpha * self.sigma
                + (1 - self.alpha) * torch.squeeze(new_var).numpy()
            )
        return best_plan
