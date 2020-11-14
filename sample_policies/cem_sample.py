import torch

from sample_policies.sample import Sample

from torch.distributions.normal import Normal
import numpy as np

import scipy.stats as stats


class CEMSample(Sample):
    def __init__(self, upper_bound, lower_bound, action_dim, is_discrete=False):
        """
        Cross Entropy Method to sample actions.
        """
        self.upper_bound = torch.tensor(upper_bound)
        self.lower_bound = torch.tensor(lower_bound)
        self.is_discrete = is_discrete
        self.action_dim = action_dim
        self.sampler = self.__truncated_normal

    def __truncated_normal(self, shape, mu, sigma, a, b):
        """
        Truncated normal distribution cross entropy sample method.
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
        v = np.clip(2 * p - 1, epsilon - one, one - epsilon)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v)).numpy()
        x = np.clip(x, a[0].numpy(), b[0].numpy())
        # x = torch.clamp(torch.tensor(x), a, b)
        return x

    def sample(self, timesteps, mu, sigma, a, b):
        shape = timesteps
        return self.sampler(shape, mu, sigma, a, b)

    def sample_n(self, sample_nums, timesteps, **kwargs):
        shape = (sample_nums, timesteps, self.action_dim)
        return self.sampler(
            shape, kwargs["mu"], kwargs["sigma"], self.lower_bound, self.upper_bound
        )
        # return self.__truncated_normal_scipy(
        #     sample_nums, timesteps, kwargs["mu"], kwargs["sigma"]
        # )

    def __truncated_normal_scipy(self, sample_nums, timesteps, mean, sigma):
        """
        Scipy truncated normal method.
        """
        shape = (sample_nums, timesteps, self.action_dim)
        lb_dist, ub_dist = (
            mean - self.lower_bound.numpy(),
            self.upper_bound.numpy() - mean,
        )
        constrained_var = np.minimum(
            np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), sigma
        )
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))
        samples = X.rvs(size=shape) * np.sqrt(constrained_var) + mean
        return samples.astype(np.float32)
