import numpy as np

import escnn
from escnn.nn import FieldType, EquivariantModule, GeometricTensor
from hydra import compose, initialize

# from morpho_symm.nn.EMLP import EMLP
from morpho_symm.utils.robot_utils import load_symmetric_system
from morpho_symm.nn.test_EMLP import get_kinematic_three_rep_two, get_ground_reaction_forces_rep_two, get_friction_rep

import torch
import torch.nn as nn
from torch.distributions import Normal



class SimpleEMLP(EquivariantModule):
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 hidden_dims = [256, 256, 256],
                 bias: bool = True,
                 actor: bool = True,
                 activation: str = "ReLU"):
        super().__init__()
        self.out_type = out_type
        gspace = in_type.gspace
        group = gspace.fibergroup
        
        layer_in_type = in_type
        self.net = escnn.nn.SequentialModule()
        for n in range(len(hidden_dims)):
            layer_out_type = FieldType(gspace, [group.regular_representation] * int((hidden_dims[n] / group.order())))

            self.net.add_module(f"linear_{n}: in={layer_in_type.size}-out={layer_out_type.size}",
                             escnn.nn.Linear(layer_in_type, layer_out_type, bias=bias))
            self.net.add_module(f"act_{n}", self.get_activation(activation, layer_out_type))

            layer_in_type = layer_out_type

        if actor: 
            self.net.add_module(f"linear_{len(hidden_dims)}: in={layer_in_type.size}-out={out_type.size}",
                                escnn.nn.Linear(layer_in_type, out_type, bias=bias))
            self.extra_layer = None
        else:
            num_inv_features = len(layer_in_type.irreps)
            self.extra_layer = torch.nn.Linear(num_inv_features, out_type.size, bias=False)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        x= self.net(x)
        if self.extra_layer:
            x = self.extra_layer(x.tensor)
        return x

    @staticmethod
    def get_activation(activation: str, hidden_type: FieldType) -> EquivariantModule:
        if activation.lower() == "relu":
            return escnn.nn.ReLU(hidden_type)
        elif activation.lower() == "elu":
            return escnn.nn.ELU(hidden_type)
        elif activation.lower() == "lrelu":
            return escnn.nn.LeakyReLU(hidden_type)
        else:
            raise NotImplementedError

    def evaluate_output_shape(self, input_shape):
        """Returns the output shape of the model given an input shape."""
        batch_size = input_shape[0]
        return batch_size, self.out_type.size

    def export(self):
        """Exports the model to a torch.nn.Sequential instance."""
        sequential = nn.Sequential()
        for name, module in self.net.named_children():
            sequential.add_module(name, module.export())
        return sequential




from params_proto import PrefixProto
class ACS_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_module_branch_hidden_dims = [256, 128]

    use_decoder = False


G = None
from go2_gym_learn.ppo_cse.actor_critic import get_activation
class ActorCriticEMLP(nn.Module):
    is_recurrent = False

    def __init__(self,
        num_obs,
        num_privileged_obs,
        num_obs_history,
        num_actions,
        **kwargs
    ):
        if kwargs:
            print("ActorCriticEMLP.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = ACS_Args.use_decoder
        super().__init__()

        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        activation = ACS_Args.activation


        global G
        # Load robot instance and its symmetry group
        initialize(config_path="../../MorphoSymm/morpho_symm/cfg/robot", version_base='1.3')
        robot_name = 'go2'  # or any of the robots in the library (see `/morpho_symm/cfg/robot`)
        robot_cfg = compose(config_name=f"{robot_name}.yaml")
        robot, G = load_symmetric_system(robot_cfg=robot_cfg)
        # We use ESCNN to handle the group/representation-theoretic concepts and for the construction of equivariant neural networks.
        gspace = escnn.gspaces.no_base_space(G)
        # Get the relevant group representations.
        rep_QJ = G.representations["Q_js"]  # Used to transform joint-space position coordinates q_js ∈ Q_js
        rep_TqQJ = G.representations["TqQ_js"]  # Used to transform joint-space velocity coordinates v_js ∈ TqQ_js
        rep_O3 = G.representations["Rd"]  # Used to transform the linear momentum l ∈ R3
        rep_O3_pseudo = G.representations["Rd_pseudo"]  # Used to transform the angular momentum k ∈ R3
        trivial_rep = G.trivial_representation
        rep_kin_three = get_kinematic_three_rep_two(G)
        rep_friction = get_friction_rep(G, rep_kin_three)

        num_obs_history_length = self.num_obs_history // num_obs
        base_transition = ([
            rep_O3, # projected gravity. 3
            rep_O3_pseudo, # command for x_linvel, y_linvel, yaw_angvel. 3 # TODO: this should be aligned with the definition of obs_buf in legged_robot.py
            trivial_rep, # command for body height. 1
            trivial_rep, # command for frequency. 1
            trivial_rep, # command for phases, 1
            trivial_rep, # command for offsets, 1
            trivial_rep, # command for bounds, 1
            trivial_rep, # command for durations, 1
            trivial_rep, # command for desired_footswing_height, 1
            trivial_rep, # command for limit_body_pitch (see Cfg.commands), 1
            trivial_rep, # command for limit_body_roll (see Cfg.commands), 1
            trivial_rep, # command for limit_stance_width (see Cfg.commands), 1
            trivial_rep, # command for limit_stance_length (see Cfg.commands), 1 
            trivial_rep, # command for limit_aux_reward (see Cfg.commands), 1
            rep_TqQJ, # dof pos. 12
            rep_TqQJ, # dof vel. 12
            rep_TqQJ, # action. 12
            rep_TqQJ, # last_action. 12
            rep_kin_three, 
            rep_kin_three, # clock inputs. 4
            ]) * num_obs_history_length #
        rep_extra_obs = [
            trivial_rep, # priv_observe_friction, 1
            trivial_rep, # priv_observe_restitution, 1
        ]
        
        in_field_type = FieldType(gspace, base_transition + rep_extra_obs) # Inlcude privilege obs for this task
        # Representation of y := [l, k] ∈ R3 x R3            =>    ρ_Y_js(g) := ρ_O3(g) ⊕ ρ_O3pseudo(g)  | g ∈ G
        out_field_type = FieldType(gspace, [rep_QJ])
        
        self.gspace = gspace
        self.in_field_type = in_field_type
        self.out_field_type = out_field_type
        # one dimensional field type for critic
        critic_out_field_type = FieldType(gspace, [G.trivial_representation])

        # Adaptation module if not using EMLP
        activation_func = get_activation(activation)
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.num_obs_history, ACS_Args.adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation_func)
        for l in range(len(ACS_Args.adaptation_module_branch_hidden_dims)):
            if l == len(ACS_Args.adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(ACS_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
            else:
                adaptation_module_layers.append(
                    nn.Linear(ACS_Args.adaptation_module_branch_hidden_dims[l],
                              ACS_Args.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation_func)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)


        # Construct the equivariant MLP
        self.actor_body = SimpleEMLP(
            in_field_type, 
            out_field_type,
            hidden_dims = ACS_Args.actor_hidden_dims, 
            activation = activation
        )

        self.critic_body = SimpleEMLP(
            in_field_type, 
            critic_out_field_type,
            hidden_dims = ACS_Args.critic_hidden_dims,
            activation=activation
        )

        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor EMLP: {self.actor_body}")
        print(f"Critic EMLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(ACS_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False


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

        '''
            Transform the input into GeometricTensor
        '''
        ob = self.in_field_type(torch.cat((observation_history, latent), dim=-1))
        mean = self.actor_body(ob).tensor
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observation_history, **kwargs):
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
        '''
            Transform the input into GeometricTensor
        '''
        ob = self.in_field_type(torch.cat((observation_history, latent), dim=-1))
        actions_mean = self.actor_body(ob).tensor
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, observation_history, privileged_info, policy_info={}):
        '''
            Transform the input into GeometricTensor
        '''
        ob = self.in_field_type(torch.cat((observation_history, privileged_info), dim=-1))

        actions_mean = self.actor_body(ob).tensor
        policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, observation_history, privileged_observations, **kwargs):
        '''
            Transform the input into GeometricTensor
        '''
        ob = self.in_field_type(torch.cat((observation_history, privileged_observations), dim=-1))

        value = self.critic_body(ob).tensor
        return value

    def get_student_latent(self, observation_history):
        return self.adaptation_module(observation_history)