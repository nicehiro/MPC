import gym
from gym import Env
import copy

import numpy as np
import torch
import logging

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
        self.gamma = mpc_config["gamma"]
        self.max_iters = mpc_config["max_iters"]
        self.epsilon = mpc_config["epsilon"]
        self.sampler = self.samplers[mpc_config["sampler"]](
            self.upper_bound, self.lower_bound, self.action_dim, self.is_discrete
        )
        self.use_cem = True if mpc_config["sampler"] == "CEM" else False
        self.mu = self.sigma = None
        self.alpha = mpc_config["alpha"]

    def act(self, state):
        """
        Get first action from best plan.

        @param state: first state
        :return: action
        """
        best_plan = self.cem_plan(state) if self.use_cem else self.random_plan(state)
        return best_plan[0]

    def _calc_rewards(self, first_state, actions):
        """
        Calc rewards of trajectories.

        @param first_state: first state
        @param actions: sample_times * timesteps * action_dim
        :return: rewards sample_times * 1
        """
        rewards = np.zeros(self.sample_times)
        states = np.tile(first_state, (self.sample_times, 1))
        for timestep in range(self.timesteps):
            action = actions[:, timestep, :]
            states_next = self.dynamic.predict(states, action) + states
            reward = self.rewards.predict(states, action)
            reward = reward.reshape(rewards.shape)
            rewards += (self.gamma ** timestep) * reward
            states = copy.deepcopy(states_next)
        return rewards

    def reset(self):
        self.mu = np.tile(
            (self.upper_bound + self.lower_bound) / 2, [self.timesteps, 1]
        )
        self.sigma = np.tile(
            np.square(self.lower_bound - self.upper_bound) / 16, [self.timesteps, 1]
        )

    def random_plan(self, state):
        """
        Random planning.
        """
        # logging.info("Mu: {0}, Sigma: {1}".format(self.mu, self.sigma))
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
        return best_plan

    def cem_plan(self, state):
        """
        CEM planning.

        Update mu and sigma for each planning to chose best plan.
        """
        t = 0
        while t < self.max_iters and np.max(self.sigma) > self.epsilon:
            actions = self.sampler.sample_n(
                sample_nums=self.sample_times,
                timesteps=self.timesteps,
                mu=self.mu,
                sigma=self.sigma,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
            )
            rewards = self._calc_rewards(state, actions)
            self.update_mu_and_sigma(rewards, actions)
            t += 1
        return self.mu

    def update_mu_and_sigma(self, rewards, actions):
        # update mu and mean
        if self.is_cem:
            idx = np.argsort(rewards, axis=0)[::-1]
            elites = actions[idx.squeeze()][: self.num_elites]
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)
            self.mu = self.alpha * self.mu + (1 - self.alpha) * new_mean
            self.sigma = self.alpha * self.sigma + (1 - self.alpha) * new_var
