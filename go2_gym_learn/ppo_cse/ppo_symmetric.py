import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from go2_gym_learn.ppo_cse.ppo import PPO
from go2_gym_learn.ppo_cse.actor_critic_symmetric import ActorCriticSymm
from go2_gym_learn.ppo_cse import RolloutStorage
from go2_gym_learn.ppo_cse import caches


class PPOEMLP_Args(PrefixProto):
    # algorithm
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-3  # 5.e-4
    adaptation_module_learning_rate = 1.e-3
    num_adaptation_module_substeps = 1
    schedule = 'adaptive'  # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.

    selective_adaptation_module_loss = False


class PPOEMLP(PPO):
    actor_critic: ActorCriticSymm

    def __init__(self, actor_critic, device='cpu'):

        self.device = device

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=PPOEMLP_Args.learning_rate)
        self.adaptation_module_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                      lr=PPOEMLP_Args.adaptation_module_learning_rate)
        if self.actor_critic.decoder:
            self.decoder_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                          lr=PPOEMLP_Args.adaptation_module_learning_rate)
        self.transition = RolloutStorage.Transition()

        self.learning_rate = PPOEMLP_Args.learning_rate