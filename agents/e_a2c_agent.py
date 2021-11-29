import random
import torch
from torch import optim
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

from common.b_models_and_buffer import ActorCritic, ReplayBuffer


class TTTAgentA2C:
    def __init__(self, name, env, gamma, learning_rate, batch_size):
        self.name = name
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(capacity=batch_size)

        self.actor_critic_model = ActorCritic(n_features=12, n_actions=12)
        self.optimizer = optim.Adam(self.actor_critic_model.parameters(), lr=learning_rate)

        # init rewards
        self.episode_reward_lst = []

        self.time_steps = 0
        self.training_time_steps = 0

    def get_action(self, state, mode="TRAIN"):
        available_actions = state.get_available_actions()
        action = random.choice(available_actions)

        # TODO

        return action

    def learning(self, state, action, next_state, reward, done):
        loss = 0.0

        # TODO

        return loss
