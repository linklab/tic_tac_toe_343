import enum

import torch
import os

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


class AGENT_TYPE(enum.Enum):
    DQN = 0
    REINFORCE = 1
    A2C = 2


class PLAY_TYPE(enum.Enum):
    FIRST = 0
    BACK = 1
    SELF = 2


def model_save(model, agent_type, play_type, last_episode):
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, "{0}_{1}_{2}.pth".format(
            agent_type, play_type, last_episode
        ))
    )


def model_load(model, file_name):
    model_params = torch.load(
        os.path.join(MODEL_DIR, file_name)
    )
    model.load_state_dict(model_params)
