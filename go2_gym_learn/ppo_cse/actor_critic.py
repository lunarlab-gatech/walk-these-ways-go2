import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal
from .network_manager import NetworkManager

from .config import RunnerArgs
from .config import PPO_Args

class AC_Args(PrefixProto, cli=False):
    init_noise_std = 1.0
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    adaptation_module_branch_hidden_dims = [256, 128]
    use_decoder = False
    
    # Network architecture selection
    network_architecture = "mlp"
    
    # MLP specific configuration
    class mlp(PrefixProto, cli=False):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
    
    # GNN specific configuration
    class gnn(PrefixProto, cli=False):
        hidden_dim = 128
        num_layers = 8
        activation = 'elu'
        num_envs = 4096 # NOTE: this should be set to the number of environments in the environment
        num_env_mini_batch = num_envs * RunnerArgs.num_steps_per_env // PPO_Args.num_mini_batches
    
    # EMLP specific configuration
    class emlp(PrefixProto, cli=False):
        hidden_dims = [512, 256, 128]
        group_type = "SO3"
        activation = 'elu'


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 network_architecture="mlp",
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = AC_Args.use_decoder
        super().__init__()

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
        
        self.network_manager = NetworkManager()
        self.architecture = self.network_manager.get_architecture(
            name=AC_Args.network_architecture,
            input_dim=self.num_obs_history + self.num_privileged_obs,
            action_dim=num_actions,
            **kwargs)
        
        self._create_networks()

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _create_networks(self):
        """Create the networks for the actor and critic"""
        # actor network
        self.actor_body = self.architecture.create_actor()
        # critic network
        self.critic_body = self.architecture.create_critic()
        print(f"Actor {self.architecture.get_name()}: {self.actor_body}")
        print(f"Critic {self.architecture.get_name()}: {self.critic_body}")

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observation_history):
        latent = self.adaptation_module(observation_history)
        mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observation_history, **kwargs): # NOTE: this is the actor network
        self.update_distribution(observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs_history"], policy_info=policy_info)

    def act_student(self, observation_history, policy_info={}):
        latent = self.adaptation_module(observation_history)
        actions_mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, observation_history, privileged_info, policy_info={}):
        actions_mean = self.actor_body(torch.cat((observation_history, privileged_info), dim=-1))
        policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, observation_history, privileged_observations, **kwargs): # NOTE: this is the critic network
        value = self.critic_body(torch.cat((observation_history, privileged_observations), dim=-1))
        return value

    def get_student_latent(self, observation_history):
        return self.adaptation_module(observation_history)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
