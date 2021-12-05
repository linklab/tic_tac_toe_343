import random
import torch
from torch import optim
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

from common.b_models_and_buffer import ActorCritic, ReplayBuffer, Transition
from common.d_utils import AGENT_TYPE


class TTTAgentA2C:
    def __init__(self, name, env, gamma=0.99, learning_rate=0.00001, batch_size=32):
        self.name = name
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(capacity=batch_size)
        self.agent_type = AGENT_TYPE.A2C.value

        self.actor_critic_model = ActorCritic(n_features=12, n_actions=12)
        self.optimizer = optim.Adam(self.actor_critic_model.parameters(), lr=learning_rate)

        # init rewards
        self.episode_reward_lst = []

        self.time_steps = 0
        self.training_time_steps = 0

        self.model = self.actor_critic_model

    def get_action(self, state, epsilon=0.0, mode="TRAIN"):
        available_actions = state.get_available_actions()
        unavailable_actions = list(set(self.env.ALL_ACTIONS) - set(available_actions))
        obs = state.data.flatten()
        action = None

        # TODO

        return action.item()

    def learning(self, state, action, next_state, reward, done):
        loss = 0.0

        self.buffer.append(
            Transition(state.data.flatten(), action, next_state.data.flatten(), reward, done)
        )

        if len(self.buffer) < self.batch_size:
            return loss

        # sample all from buffer
        batch = self.buffer.sample(batch_size=-1)

        observations, actions, next_observations, rewards, dones = batch

        # TODO

        self.training_time_steps += 1

        return loss.item()
