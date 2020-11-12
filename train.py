import os
import logging

import gym
import yaml
from gym.spaces.discrete import Discrete

from algos.mpc import MPC
from models.dynamic_model import DynamicModel
from models.reward_model import RewardModel


def load_config(path):
    if os.path.isfile(path):
        f = open(path)
        return yaml.load(f, Loader=yaml.FullLoader)
    raise Exception("Not found config file.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = load_config("./configs/train.yml")
    # env = gym.make("MountainCarContinuous-v0")
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    if type(env.action_space) is Discrete:
        action_dim = 1
        upper_bound = env.action_space.n
        lower_bound = 0
        is_discrete = True
    else:
        action_dim = env.action_space.shape[0]
        upper_bound = env.action_space.high
        lower_bound = env.action_space.low
        is_discrete = False

    nn_config = config["NN_config"]
    nn_config["state_dim"] = state_dim
    nn_config["action_dim"] = action_dim
    dynamic = DynamicModel(nn_config)

    reward_config = config["reward_config"]
    reward_config["state_dim"] = state_dim
    reward_config["action_dim"] = action_dim
    reward_model = RewardModel(reward_config)

    mpc_config = config["mpc_config"]
    mpc_config["upper_bound"] = upper_bound
    mpc_config["lower_bound"] = lower_bound
    mpc_config["action_dim"] = action_dim
    mpc_config["is_discrete"] = is_discrete
    mpc = MPC(mpc_config, dynamic, reward_model)

    if not (
        nn_config["model_config"]["load_model"]
        and reward_config["model_config"]["load_model"]
    ):
        pretrained_episode = 40
        logging.info(
            "Random sample actions from env and fit the dynamic and reward model."
        )
        for _ in range(pretrained_episode):
            o = env.reset()
            d = False
            while not d:
                # env.render()
                a = env.action_space.sample()
                o_, r, d, _ = env.step(a)
                logging.debug(
                    "Random sampled state-action pairs: {}".format([o, a, r, d, o_])
                )
                dynamic.add_dataset([0, o, a, o_ - o])
                reward_model.add_dataset([0, o, a, [r]])
                o = o_
    # fit model
    if not nn_config["model_config"]["load_model"]:
        logging.info("Fitting Dynamic model.")
        dynamic.fit()
    if not reward_config["model_config"]["load_model"]:
        logging.info("Fitting Reward model.")
        reward_model.fit()
    # test model
    test_episodes = 3
    test_epochs = 20
    for _ in range(test_episodes):
        for epo in range(test_epochs):
            o = env.reset()
            d = False
            t = 0
            while not d:
                env.render()
                a = mpc.act(o)
                if is_discrete:
                    a = a.item()
                logging.info("Predict action: {}".format(a))
                o_, r, d, _ = env.step(a)
                dynamic.add_dataset([0, o, a, o_ - o])
                reward_model.add_dataset([0, o, a, [r]])
                o = o
                if t > 100:
                    break
            env.close()
            logging.info("Episode done!")

        if not nn_config["model_config"]["load_model"]:
            logging.info("Fitting Dynamic model.")
            dynamic.fit()
        if not reward_config["model_config"]["load_model"]:
            logging.info("Fitting Reward model.")
            reward_model.fit()
