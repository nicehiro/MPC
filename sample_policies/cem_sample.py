import torch

from sample_policies.sample import Sample

from torch.distributions.normal import Normal
import numpy as np


class CEMSample(Sample):

    def __init__(self, upper_bound, lower_bound, action_dim):
        self.sampler = self.__truncated_normal
        self.upper_bound = torch.tensor(upper_bound)
        self.lower_bound = torch.tensor(lower_bound)
        self.action_dim = action_dim

    def __truncated_normal(self, shape, mu, sigma, a, b):
        """
        Truncated normal distribution sample method.
        Use inverse transform sampling.

        @param shape: dim of output
        @param mu: mean of data
        @param sigma: var of data
        @param a: left bound of ouput
        @param b: right bound of output

        :return: list of actions
        """
        normal = Normal(0, 1)
        U = torch.rand(shape)
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        # cdf: cumulative distribution function
        alpha_normal_cdf = normal.cdf(alpha)
        beta_normal_cdf = normal.cdf(beta)
        p = alpha_normal_cdf + U * (beta_normal_cdf - alpha_normal_cdf)
        p = p.numpy()
        one = np.array(1, dtype=p.dtype)
        epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
        v = np.clip(2 * p - 1, epsilon - 1, 1 + epsilon)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))
        x = torch.clamp(x, a[0], b[0])
        return x

    def sample(self, timesteps, mu, sigma, a, b):
        shape = (timesteps, self.action_dim)
        return self.sampler(shape, mu, sigma, a, b)

    def sample_n(self, sample_nums, timesteps, **kwargs):
        shape = (sample_nums, timesteps, self.action_dim)
        return self.sampler(shape, kwargs['mu'], kwargs['sigma'], kwargs['a'], kwargs['b'])
