import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal
from .network_manager import NetworkManager

from .config import RunnerArgs
from .config import PPO_Args
from .config import AC_Args

from .ms_gnn import MS_GNN

from .actor_critic import get_activation

from go2_gym_learn.ppo_cse import ActorCritic

class ActorCriticMSGNN(ActorCritic):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 network_architecture="gnn",
                 **kwargs):
        if kwargs:
            print("ActorCriticMSGNN.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = AC_Args.use_decoder
        super().__init__(num_obs, num_privileged_obs, num_obs_history, num_actions, network_architecture)

        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        activation = get_activation(AC_Args.activation)

        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.num_obs_history, AC_Args.adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(AC_Args.adaptation_module_branch_hidden_dims)):
            if l == len(AC_Args.adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
            else:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l],
                              AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)
        
        print(f"Adaptation Module: {self.adaptation_module}")
        
        # self.network_manager = NetworkManager()
        # self.architecture = self.network_manager.get_architecture(
        #     name=AC_Args.network_architecture,
        #     input_dim=self.num_obs_history + self.num_privileged_obs,
        #     action_dim=num_actions,
        #     **kwargs)
        # self._create_networks()
        
        self.actor_body = self.create_actor(
            hidden_dim=AC_Args.msgnn.hidden_dim,
            num_layers=AC_Args.msgnn.num_layers,
            num_envs=AC_Args.msgnn.num_envs,
            num_env_mini_batch=AC_Args.msgnn.num_env_mini_batch,
            activation=get_activation(AC_Args.msgnn.activation)
        )
        self.critic_body = self.create_critic(
            hidden_dim=AC_Args.msgnn.hidden_dim,
            num_layers=AC_Args.msgnn.num_layers,
            num_envs=AC_Args.msgnn.num_envs,
            num_env_mini_batch=AC_Args.msgnn.num_env_mini_batch,
            activation=get_activation(AC_Args.msgnn.activation)
        )
        print(f"Actor MS-GNN: {self.actor_body}")
        print(f"Critic MS-GNN: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
    def create_actor(self, hidden_dim: int, num_layers: int, num_envs: int, num_env_mini_batch: int, activation: nn.Module) -> nn.Module:
        return MS_GNN(
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            activation_fn=activation,
            num_envs=num_envs,
            num_env_mini_batch=num_env_mini_batch,
            is_critic=False
        )
    
    def create_critic(self, hidden_dim: int, num_layers: int, num_envs: int, num_env_mini_batch: int, activation: nn.Module) -> nn.Module:
        return MS_GNN(
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            activation_fn=activation,
            num_envs=num_envs,
            num_env_mini_batch=num_env_mini_batch,
            is_critic=True
        )

    # def _create_networks(self):
    #     """Create the networks for the actor and critic"""
    #     # actor network
    #     self.actor_body = self.architecture.create_actor()
    #     # critic network
    #     self.critic_body = self.architecture.create_critic()
    #     print(f"Actor {self.architecture.get_name()}: {self.actor_body}")
    #     print(f"Critic {self.architecture.get_name()}: {self.critic_body}")
