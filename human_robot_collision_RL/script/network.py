# adapted from https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Network(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(Network, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "img":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                n_input_channels = observation_space["img"].shape[0]
                #65,552 - 16 = 65,536
                #62,017 - 16 = 62,001
                #65,552 - 16 = 65,536
                extractors[key] = nn.Sequential(
                    nn.Conv2d(n_input_channels,4,8,padding="same"),
                    nn.ReLU(),
                    nn.MaxPool2d(4),
                    #nn.Conv2d(4,4,8,padding="same"),
                    #nn.ReLU(),
                    #nn.MaxPool2d(4),
                    nn.Conv2d(4,4,4,padding="same"),
                    nn.ReLU(),
                    nn.MaxPool2d(4),#2
                    nn.Flatten(),
                    #nn.Linear(16,64),
                    #nn.ReLU(),
                    #nn.Linear(64,64),
                    #nn.ReLU(),
                    #nn.Linear(64,2),
                    #nn.Tanh()
                    )

                total_concat_size += 256#subspace.shape[1] // 32 * subspace.shape[2] // 32
            elif key == "vec":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)