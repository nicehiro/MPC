from models.model import Model


class Sample:

    def __init__(self):
        pass

    def sample(self, timesteps, **kwargs):
        """
        Sample a list of actions.

        :return: timesteps * action_dim
        """
        raise NotImplementedError('You must implement function sample!')

    def sample_n(self, sample_nums, timesteps, **kwargs):
        """
        Sample actions of `sample_nums' trajectories.

        @param sample_times: sample nums
        @param timesteps: how many steps each trajectory should make

        :return: sample_nums * timesteps * action_dim
        """
        raise NotImplementedError('You must implement this function!')
