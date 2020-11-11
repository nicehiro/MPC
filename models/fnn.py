from models.model import Model
from utils.net_utils import MLPRegression
from utils.torch_utils import CUDA, CPU

import torch
import torch.nn as nn

import numpy as np

import logging


class FNN(Model):
    activations = {'Tanh': nn.Tanh(),
                   'ReLU': nn.ReLU()}

    def __init__(self, **kwargs):
        """
        Init FNN.

        :param kwargs:
        """
        super(FNN, self).__init__()
        self.model = CUDA(MLPRegression(kwargs['input_dim'], kwargs['output_dim'],
                                        kwargs['hidden_sizes'], self.activations[kwargs['activation']]))
        self.dataset = []
        self.batch_size = kwargs['batch_size']
        self.epochs_n = kwargs['epochs_n']
        self.lr = kwargs['lr']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss(reduction='mean')
        self.save_model_flag = kwargs['save_model_flag']
        self.save_model_path = kwargs['save_model_path']
        self.validation_flag = kwargs['validation_flag']
        self.validation_freq = kwargs['validation_freq']
        if kwargs['load_model']:
            self.load_model()

    def predict(self, input):
        """
        Give predict value of input.

        :param input: the input of net.
        :return: predict value.
        """
        input = CUDA(input)
        with torch.no_grad():
            delta_state = self.model(torch.tensor(input, dtype=torch.float))
            delta_state = CPU(delta_state).numpy()
            logging.debug('Predict value: {0}'.format(delta_state))
        return delta_state

    def fit(self):
        """
        Train model with dataset.

        :return: Loss
        """
        train_loader = torch.utils.data.DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size)
        for epoch in range(self.epochs_n):
            loss_of_epoch = []
            for feature, label in train_loader:
                self.optimizer.zero_grad()
                output = self.model(feature)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()
                loss_of_epoch.append(loss)
            if self.validation_flag and (epoch + 1) % self.validation_freq == 0:
                logging.info('Training ({0}/{1}): Loss: {2}'
                             .format(epoch+1, self.epochs_n, np.mean(loss_of_epoch, dtype=np.float)))
                self.save_model()
        if self.save_model_flag:
            self.save_model()
        return np.mean(loss_of_epoch, dtype=np.float)

    def reset_dataset(self):
        self.dataset = []

    def add_dataset(self, data):
        """
        Add data to dataset.

        @param data: [id, state, action, next_state]
        :return: None
        """
        _, state, action, state_ = data
        if type(action) is not np.ndarray:
            action = np.array([action])
        feature = np.concatenate((state, action), axis=0)
        feature = CUDA(torch.tensor(feature)).float()
        label = CUDA(torch.tensor(state_)).float()
        self.dataset.append([feature, label])

    def save_model(self):
        """
        Save model.

        :return: None
        """
        torch.save(self.model.state_dict(), self.save_model_path)

    def load_model(self):
        """
        Load model.

        :return: None
        """
        self.model.load_state_dict(torch.load(self.save_model_path))
