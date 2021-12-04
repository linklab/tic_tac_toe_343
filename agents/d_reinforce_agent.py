import random
import torch
from torch import optim
import numpy as np
from torch.distributions import Categorical

from common.b_models_and_buffer import Policy, Transition, ReplayBuffer
from common.d_utils import AGENT_TYPE


class TTTAgentReinforce:
    def __init__(self, name, env, gamma, learning_rate):
        self.name = name
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.agent_type = AGENT_TYPE.REINFORCE.value

        self.buffer = ReplayBuffer(capacity=100_000)

        self.policy = Policy(n_features=12, n_actions=12)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.episode_reward_lst = []

        self.time_steps = 0
        self.training_time_steps = 0

        self.model = self.policy

    def get_action(self, state, epsilon=0.0, mode="TRAIN"):
        available_actions = state.get_available_actions()
        unavailable_actions = list(set(self.env.ALL_ACTIONS) - set(available_actions))

        action = None

        # TODO

        return action.item()

    def learning(self, state, action, next_state, reward, done):
        loss = 0.0
        self.buffer.append(
            Transition(state.data.flatten(), action, next_state.data.flatten(), reward, done)
        )
        if not done or len(self.buffer) == 0:
            return loss

        # sample all from buffer
        batch = self.buffer.sample(batch_size=-1)

        # observations.shape: torch.Size([32, 4]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])
        observations, actions, _, rewards, _ = batch

        # TODO

        return loss.item()
