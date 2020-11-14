import torch

from sample_policies.sample import Sample
from torch.distributions.uniform import Uniform


class RandomSample(Sample):
    def __init__(self, upper_bound, lower_bound, action_dim, is_discrete=False):
        """
        Random sample actions.
        """
        self.upper_bound = torch.tensor(upper_bound)
        self.lower_bound = torch.tensor(lower_bound)
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        if is_discrete:
            self.sampler = RandIntSampler(self.lower_bound, self.upper_bound)
        else:
            self.sampler = Uniform(self.lower_bound, self.upper_bound)

    def sample(self, timesteps):
        shape = (timesteps, self.action_dim)
        actions = self.sampler.sample(shape).cpu().numpy()
        return actions

    def sample_n(self, sample_nums, timesteps, **kwargs):
        shape = (
            (sample_nums, timesteps, self.action_dim)
            if self.is_discrete
            else (sample_nums, timesteps)
        )
        actions = self.sampler.sample(shape).cpu().numpy()
        return actions


class RandIntSampler:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self, shape):
        return torch.randint(self.low, self.high, shape)
