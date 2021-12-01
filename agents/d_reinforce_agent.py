import random
import torch
from torch import optim
import numpy as np

from common.b_models_and_buffer import Policy


class TTTAgentReinforce:
    def __init__(self, name, env, gamma, learning_rate):
        self.name = name
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.buffer = []

        self.policy = Policy(n_features=12, n_actions=12)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.episode_reward_lst = []

        self.time_steps = 0
        self.training_time_steps = 0

    def get_action(self, state, epsilon=0.0, mode="TRAIN"):
        available_actions = state.get_available_actions()
        action = random.choice(available_actions)

        # TODO

        return action

    def learning(self, state, action, next_state, reward, done):
        loss = 0.0

        # TODO

        return loss
