import random
import torch
from torch import optim
import numpy as np

from common.b_models_and_buffer import QNet, ReplayBuffer


class TTTAgentDqn:
    def __init__(
            self, name, env, gamma=0.99, learning_rate=0.001,
            replay_buffer_size=10_000, batch_size=32, target_sync_step_interval=500,
            min_buffer_size_for_training=100
    ):
        self.name = name
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_sync_step_interval = target_sync_step_interval
        self.replay_buffer_size = replay_buffer_size
        self.min_buffer_size_for_training = min_buffer_size_for_training

        # network
        self.q = QNet(n_features=12, n_actions=12)
        self.target_q = QNet(n_features=12, n_actions=12)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

        # agent
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.episode_reward_lst = []

        self.time_steps = 0
        self.training_time_steps = 0

    def get_action(self, state, epsilon=0.0):
        available_actions = state.get_available_actions()
        action = random.choice(available_actions)

        # TODO

        return action

    def learning(self, state, action, next_state, reward, done):
        loss = 0.0

        # TODO

        return loss