from models.fnn import FNN
from utils.torch_utils import CUDA

import torch
import numpy as np


class RewardModel(FNN):

    def __init__(self, reward_config):
        model_config = reward_config['model_config']
        training_config = reward_config['training_config']
        state_dim = reward_config['state_dim']
        action_dim = reward_config['action_dim']
        super(RewardModel, self).__init__(
            load_model=model_config['load_model'],
            model_path=model_config['model_path'],
            input_dim=state_dim+action_dim,
            output_dim=1,
            hidden_sizes=model_config['hidden_sizes'],
            activation=model_config['activation'],
            batch_size=training_config['batch_size'],
            epochs_n=training_config['n_epochs'],
            lr=training_config['learning_rate'],
            save_model_flag=training_config['save_model_flag'],
            validation_flag=training_config['validation_flag'],
            save_model_path=training_config['save_model_path'],
            validation_freq=training_config['validation_freq']
        )

    def make_dataset(self, data):
        """
        Make dataset.

        @param data: training data, list of [id, state, action, reward]
        :return: None
        """
        for d in data:
            _, state, action, state_ = d
            feature = np.concatenate((state, action), axis=0)
            feature = CUDA(torch.tensor(feature))
            label = CUDA(torch.tensor(state))
            self.dataset.append([feature, label])

    def predict(self, state, action):
        """
        Give predict reward of state and action.

        @param state: state
        @param action: action
        :return: predict state
        """
        feature = np.concatenate((state, action), axis=0)
        return super().predict(feature)
