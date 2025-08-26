import torch
import torch.nn as nn
from torch_geometric.nn import Linear, GraphConv
# from copy import deepcopy

class MS_GNN(torch.nn.Module):
    """
    Standard GNN for the C2 graph structure with 14 nodes (2 base + 12 joint)
    """
    def __init__(self,
                 hidden_channels: int, 
                 num_layers: int, 
                 activation_fn = nn.ELU(), 
                 num_envs: int = 4096,
                 num_env_mini_batch: int = 4096,
                 is_critic: bool = False):
        """
        Implementation of a MS-GNN model for C2 structure.

        Parameters:
            hidden_channels (int): Size of the node embeddings in the graph.
            num_layers (int): Number of message-passing layers.
            activation_fn (class): The activation function used between layers.
        """
        super().__init__()
        self.activation = activation_fn
        self.batch_size = num_envs
        self.mini_batch_size = num_env_mini_batch
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

         # Total number of nodes (combining base and joint)
        self.num_base_nodes = 2
        self.num_front_joint_nodes = 6
        self.num_rear_joint_nodes = 6
        self.num_nodes = self.num_base_nodes + self.num_front_joint_nodes + self.num_rear_joint_nodes
        
        # Create a single edge_index for all connections
        self.edge_index = self._create_edges_index()    
        self.is_critic = is_critic
        # self.has_privileged_obs = True
        
        if True:
            '''                
            Observation structure for walking gait tasks:
            | Observation                   | Index | Node         |
            |-------------------------------|-------|--------------|
            | Projected gravity             | 0:3   | base         |
            | Commands                      | 3:18  | base         |
            | Joint positions               | 18:30 | joint        |
            | Joint velocities              | 30:42 | joint        |
            | Joint current actions         | 42:54 | joint        |
            | Joint last actions            | 54:66 | joint        |
            | Phase input                   | 66:70 | joint        |
            |-------------------------------|-------|--------------|
            | Ground friction coefficient   | 0     | base, joint  |
            | Ground restitution coefficient| 1     | base, joint  |
                
            Feature Assignment
            | Node  | Node      | Common             | Privileged    |
            | Index | Name      | Feature Index      | Feature Index |
            |-------|-----------|--------------------|---------------|
            | 0     | base-L    |  *range(18)        | 0, 1          |
            | 1     | base-R    |  *range(18)        | 0, 1          |
            | 2     | FL-hip    | 18, 30, 42, 54, 66 | 0, 1          |
            | 3     | FL-thigh  | 19, 31, 43, 55, 66 | 0, 1          |
            | 4     | FL-knee   | 20, 32, 44, 56, 66 | 0, 1          |
            | 5     | FR-hip    | 21, 33, 45, 57, 67 | 0, 1          |
            | 6     | FR-thigh  | 22, 34, 46, 58, 67 | 0, 1          |
            | 7     | FR-knee   | 23, 35, 47, 59, 67 | 0, 1          |
            | 8     | RL-hip    | 24, 36, 48, 60, 68 | 0, 1          |
            | 9     | RL-thigh  | 25, 37, 49, 61, 68 | 0, 1          |
            | 10    | RL-knee   | 26, 38, 50, 62, 68 | 0, 1          |
            | 11    | RR-hip    | 27, 39, 51, 63, 69 | 0, 1          |
            | 12    | RR-thigh  | 28, 40, 52, 64, 69 | 0, 1          |
            | 13    | RR-knee   | 29, 41, 53, 65, 69 | 0, 1          |
            '''
            self.num_timesteps = 30   # obs_history_length
            self.dim_common_obs = 70  # num_obs
            self.node_dict = {
                0: {'name': 'base-L', 'common': [*range(18)], 'privileged': [0, 1]},
                1: {'name': 'base-R', 'common': [*range(18)], 'privileged': [0, 1]},
                2: {'name': 'FL-hip', 'common': [18, 30, 42, 54, 66], 'privileged': [0, 1]},
                3: {'name': 'FL-thigh', 'common': [19, 31, 43, 55, 66], 'privileged': [0, 1]},
                4: {'name': 'FL-knee', 'common': [20, 32, 44, 56, 66], 'privileged': [0, 1]},
                5: {'name': 'FR-hip', 'common': [21, 33, 45, 57, 67], 'privileged': [0, 1]},
                6: {'name': 'FR-thigh', 'common': [22, 34, 46, 58, 67], 'privileged': [0, 1]},
                7: {'name': 'FR-knee', 'common': [23, 35, 47, 59, 67], 'privileged': [0, 1]},
                8: {'name': 'RL-hip', 'common': [24, 36, 48, 60, 68], 'privileged': [0, 1]},
                9: {'name': 'RL-thigh', 'common': [25, 37, 49, 61, 68], 'privileged': [0, 1]},
                10: {'name': 'RL-knee', 'common': [26, 38, 50, 62, 68], 'privileged': [0, 1]},
                11: {'name': 'RR-hip', 'common': [27, 39, 51, 63, 69], 'privileged': [0, 1]},
                12: {'name': 'RR-thigh', 'common': [28, 40, 52, 64, 69], 'privileged': [0, 1]},
                13: {'name': 'RR-knee', 'common': [29, 41, 53, 65, 69], 'privileged': [0, 1]},
            }
            self.node_type_dict = {
                'base': {'node_indices': [0, 1], 'common_input_dim': -1, 'privileged_input_dim': -1},
                'F-joint': {'node_indices': [2, 3, 4, 5, 6, 7], 'common_input_dim': -1, 'privileged_input_dim': -1},
                'R-joint': {'node_indices': [8, 9, 10, 11, 12, 13], 'common_input_dim': -1, 'privileged_input_dim': -1}
            }
            for node_type, node_values in self.node_type_dict.items():
                self.node_type_dict[node_type]['common_input_dim'] = len(self.node_dict[node_values['node_indices'][0]]['common'])
                self.node_type_dict[node_type]['privileged_input_dim'] = len(self.node_dict[node_values['node_indices'][0]]['privileged'])
            
            # Define the group action
            linear_weights_base_e = torch.ones((self.node_type_dict['base']['common_input_dim']), dtype=torch.float32).repeat(self.num_timesteps)
            linear_weights_base_gs = torch.tensor([
                                                    1, -1, 1, # gravity 
                                                    1, -1, -1, # commands
                                                    1, 1, 
                                                    1, 1, 1, 1, 
                                                    1, 1, -1, 1, 1, 
                                                    1
                                                ], dtype=torch.float32).repeat(self.num_timesteps)
            linear_weights_F_joint_e = torch.ones((self.node_type_dict['F-joint']['common_input_dim']), dtype=torch.float32).repeat(self.num_timesteps)
            linear_weights_F_joint_gs = torch.tensor([-1, -1, -1, -1, 1 # pos, velocity, last_action, curr_action, phase input
                                                      ], dtype=torch.float32).repeat(self.num_timesteps)
            self.permutation_F_joint_indices = [4 + i * 5 for i in range(self.num_timesteps)]
            linear_weights_R_joint_e = torch.ones((self.node_type_dict['R-joint']['common_input_dim']), dtype=torch.float32).repeat(self.num_timesteps)
            linear_weights_R_joint_gs = torch.tensor([-1, -1, -1, -1, 1 # pos, velocity, last_action, curr_action, phase input
                                                      ], dtype=torch.float32).repeat(self.num_timesteps)
            self.permutation_R_joint_indices = [4 + i * 5 for i in range(self.num_timesteps)]
            
            # if self.is_critic:
            linear_weights_base_e = torch.cat((linear_weights_base_e, torch.ones((self.node_type_dict['base']['privileged_input_dim']))), dim=0)
            linear_weights_base_gs = torch.cat((linear_weights_base_gs, 
                                                torch.tensor([1, 1] # contact information
                                                            )), dim=0)
            linear_weights_F_joint_e = torch.cat((linear_weights_F_joint_e, torch.ones((self.node_type_dict['F-joint']['privileged_input_dim']))), dim=0)
            linear_weights_F_joint_gs = torch.cat((linear_weights_F_joint_gs, torch.ones((self.node_type_dict['F-joint']['privileged_input_dim']))), dim=0)
            linear_weights_R_joint_e = torch.cat((linear_weights_R_joint_e, torch.ones((self.node_type_dict['R-joint']['privileged_input_dim']))), dim=0)
            linear_weights_R_joint_gs = torch.cat((linear_weights_R_joint_gs, torch.ones((self.node_type_dict['R-joint']['privileged_input_dim']))), dim=0)
            
            self.linear_weights_base = torch.cat((linear_weights_base_e, linear_weights_base_gs), dim=0)
            self.linear_weights_front_joint = torch.cat((
                linear_weights_F_joint_e, linear_weights_F_joint_e, linear_weights_F_joint_e, 
                linear_weights_F_joint_gs, linear_weights_F_joint_e, linear_weights_F_joint_e), dim=0)
            self.linear_weights_rear_joint = torch.cat((
                linear_weights_R_joint_e, linear_weights_R_joint_e, linear_weights_R_joint_e, 
                linear_weights_R_joint_gs, linear_weights_R_joint_e, linear_weights_R_joint_e), dim=0)  
        
        self.len_common_obs = self.num_timesteps * self.dim_common_obs
        
        # Used for the final output layer MS decoder.
        self.linear_weights_joint_actions = torch.tensor([1, 1, 1, 
                                                            -1, 1, 1, 
                                                            1, 1, 1, 
                                                            -1, 1, 1], dtype=torch.float32)
        
        if 'cuda' in self.device.type:
            self.linear_weights_base = self.linear_weights_base.to(self.device)
            self.linear_weights_front_joint = self.linear_weights_front_joint.to(self.device)
            self.linear_weights_rear_joint = self.linear_weights_rear_joint.to(self.device)
            self.linear_weights_joint_actions = self.linear_weights_joint_actions.to(self.device)
        
        # Create separate encoders for base, F-joint, R-joint nodes
        self.base_encoder = nn.Linear(self.node_type_dict['base']['common_input_dim']*self.num_timesteps + self.node_type_dict['base']['privileged_input_dim'], hidden_channels)
        self.F_joint_encoder = nn.Linear(self.node_type_dict['F-joint']['common_input_dim']*self.num_timesteps + self.node_type_dict['F-joint']['privileged_input_dim'], hidden_channels)
        self.R_joint_encoder = nn.Linear(self.node_type_dict['R-joint']['common_input_dim']*self.num_timesteps + self.node_type_dict['R-joint']['privileged_input_dim'], hidden_channels)

        # Create standard graph convolutions for each layer
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # Use standard GraphConv for all edges
            self.convs.append(GraphConv(hidden_channels, hidden_channels))

        # Output layer
        self.out_channels_per_node = 1
        
        if self.is_critic:
            self.decoder = nn.Sequential(
                Linear(hidden_channels * self.num_nodes, hidden_channels),
                self.activation,
                Linear(hidden_channels, 1)
            )
        else:
            self.decoder = Linear(hidden_channels, self.out_channels_per_node)

        # Create batched edge indices for common batch sizes
        self.edge_index_batch = self._create_edge_index_batch(self.batch_size).to(self.device)
        self.edge_index_batch_mini = self._create_edge_index_batch(self.mini_batch_size).to(self.device)
        
        # Create batched joint indices for common batch sizes
        self.joint_indices_batch = self._create_joint_indices_batch(self.batch_size).to(self.device)
        self.joint_indices_batch_mini = self._create_joint_indices_batch(self.mini_batch_size).to(self.device)

    def _create_edge_index_batch(self, batch_size):
        '''
        Create edge index for batch size
        '''
        edge_indices = []
        for i in range(batch_size):
            # Add edges with batch-specific offsets
            node_offset = i * self.num_nodes
            edge_offset = torch.ones(2, self.edge_index.shape[1], dtype=torch.int64) * node_offset
            edge_indices.append(self.edge_index + edge_offset)
        
        # Concatenate all edge indices
        edge_index_batch = torch.cat(edge_indices, dim=1)
        return edge_index_batch

    def _create_edges_index(self):
        # Define all connections in a single edge_index tensor        
        node_2_node = torch.tensor([[0, 0, 2, 3, 1, 5, 6, 0, 8, 9, 1, 11, 12],
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])

        edge_index = torch.cat([
            node_2_node, node_2_node.flip(0)
        ], dim=1)
        
        return edge_index
    
    def _create_joint_indices_batch(self, batch_size):
        '''
        Create joint indices for batch size
        Extract only joint nodes (indices 2-13)
        '''
        joint_indices = torch.arange(self.num_base_nodes, self.num_nodes)
        joint_indices = joint_indices.repeat(batch_size) + torch.arange(0, batch_size * self.num_nodes, self.num_nodes).repeat_interleave(self.num_nodes - self.num_base_nodes)

        return joint_indices
   
    def _obs_to_graph_features(self, obs_all):
        """
        Convert observations into features for each node in a batch of graphs.
        """
        batch_size = obs_all.shape[0]
        common_obs = obs_all[:, :self.len_common_obs].clone()
        common_obs = common_obs.reshape(batch_size, self.num_timesteps, -1)
        
        base_feature_list = []
        for node_idx in self.node_type_dict['base']['node_indices']:
            base_feature_list.append(common_obs[:, :, self.node_dict[node_idx]['common']].unsqueeze(1).flatten(start_dim=2)) # [b, 1, T * dim]
        base_feature = torch.cat(base_feature_list, dim=1) # [b, 2, T * dim]
        
        front_joint_feature_list = []
        for node_idx in self.node_type_dict['F-joint']['node_indices']:
            front_joint_feature_list.append(common_obs[:, :, self.node_dict[node_idx]['common']].unsqueeze(1).flatten(start_dim=2)) # [b, 1, T * dim]
        front_joint_feature = torch.cat(front_joint_feature_list, dim=1) # [b, 6, T * dim]
        
        rear_joint_feature_list = []
        for node_idx in self.node_type_dict['R-joint']['node_indices']:
            rear_joint_feature_list.append(common_obs[:, :, self.node_dict[node_idx]['common']].unsqueeze(1).flatten(start_dim=2)) # [b, 1, T * dim]
        rear_joint_feature = torch.cat(rear_joint_feature_list, dim=1) # [b, 6, T * dim]
        
        # if self.is_critic:
        privileged_obs = obs_all[:, self.len_common_obs:].clone()
        privileged_base_feature_list = []
        for node_idx in self.node_type_dict['base']['node_indices']:
            privileged_base_feature_list.append(privileged_obs[:, self.node_dict[node_idx]['privileged']].unsqueeze(1)) # [b, 1, dim]
        privileged_base_feature = torch.cat(privileged_base_feature_list, dim=1) # [b, 2, dim]
        base_feature = torch.cat((base_feature, privileged_base_feature), dim=-1)
        
        privileged_front_joint_feature_list = []
        for node_idx in self.node_type_dict['F-joint']['node_indices']:
            privileged_front_joint_feature_list.append(privileged_obs[:, self.node_dict[node_idx]['privileged']].unsqueeze(1)) # [b, 1, dim]
        privileged_front_joint_feature = torch.cat(privileged_front_joint_feature_list, dim=1) # [b, 6, dim]
        front_joint_feature = torch.cat((front_joint_feature, privileged_front_joint_feature), dim=-1)
        
        privileged_rear_joint_feature_list = []
        for node_idx in self.node_type_dict['R-joint']['node_indices']:
            privileged_rear_joint_feature_list.append(privileged_obs[:, self.node_dict[node_idx]['privileged']].unsqueeze(1)) # [b, 1, dim]
        privileged_rear_joint_feature = torch.cat(privileged_rear_joint_feature_list, dim=1) # [b, 6, dim]
        rear_joint_feature = torch.cat((rear_joint_feature, privileged_rear_joint_feature), dim=-1)

        if batch_size == self.batch_size:
            edge_index = self.edge_index_batch
        elif batch_size == self.mini_batch_size:
            edge_index = self.edge_index_batch_mini
        else:
            print(f"-------Unknown batch size: {batch_size}-------")
            edge_index = self._create_edge_index_batch(batch_size).to(self.device)
        
        # Shape: [b, 2, T*19 +2], [b, 6, T*5 +2], [b, 6, T*5 +2], [2, b*13]
        return base_feature, front_joint_feature, rear_joint_feature, edge_index

    def forward(self, obs):
        batch_size = obs.shape[0]
        
        if batch_size == 0:
            # return empty tensor with the same shape as expected output
            if self.is_critic:
                return torch.empty(0, 1, device=self.device)
            else:
                return torch.empty(0, 12, device=self.device)

        base_feature, front_joint_feature, rear_joint_feature, edge_index = self._obs_to_graph_features(obs)
        
        base_feature, front_joint_feature, rear_joint_feature = self.apply_symmetry(base_feature, front_joint_feature, rear_joint_feature)
        # Initial feature encoding - separate for base and joint nodes
        base_encoded = self.activation(self.base_encoder(base_feature)).reshape(batch_size, self.num_base_nodes, -1)  # [batch_size, 2, hidden_channels]
        front_joint_encoded = self.activation(self.F_joint_encoder(front_joint_feature)).reshape(batch_size, self.num_front_joint_nodes, -1)  # [batch_size, 6, hidden_channels]
        rear_joint_encoded = self.activation(self.R_joint_encoder(rear_joint_feature)).reshape(batch_size, self.num_rear_joint_nodes, -1)  # [batch_size, 6, hidden_channels]
    
        x = torch.cat((base_encoded, front_joint_encoded, rear_joint_encoded), dim=1).reshape(batch_size * self.num_nodes, -1)  # [batch_size * 14, hidden_channels]
        
        # Message passing layers
        for conv in self.convs:
            # Apply convolution
            x_new = conv(x, edge_index)
            x_new = self.activation(x_new)
            x = x_new
            
            # x = self.activation(conv(x, edge_index) + x)  # Residual
        
        if self.is_critic:
            # For critic, use all node embeddings
            x_reshaped = x.reshape(batch_size, -1)  # [batch_size, num_nodes * hidden_channels]
            final_output = self.decoder(x_reshaped)  # [batch_size, 1]
        else: # actor
            if batch_size == self.batch_size:
                joint_x = x[self.joint_indices_batch]  # [batch_size * 12, hidden_channels]
            elif batch_size == self.mini_batch_size:
                joint_x = x[self.joint_indices_batch_mini]  # [batch_size * 12, hidden_channels]
            else:
                print(f"-------Unknown batch size: {batch_size}-------")
                joint_x = x[self._create_joint_indices_batch(batch_size).to(self.device)]  # [batch_size * 12, hidden_channels]
            
            final_output = self.decoder(joint_x).reshape(batch_size, 12)  # [batch_size, 12]
            # # For actor, use only joint node embeddings to produce actions
            # # Extract only joint nodes (indices 2-13)
            # joint_indices = torch.arange(self.num_base_nodes, self.num_nodes).to(self.device)
            # joint_indices = joint_indices.repeat(batch_size) + torch.arange(0, batch_size * self.num_nodes, self.num_nodes).to(self.device).repeat_interleave(self.num_nodes - self.num_base_nodes)
            # joint_x = x[joint_indices]  # [batch_size * 12, hidden_channels]
            
            # final_output = self.decoder(joint_x).reshape(batch_size, self.num_nodes - self.num_base_nodes)  # [batch_size, 12]
            # final_output = self.ms_joint_decoder(final_output)

        return final_output
    
    def apply_symmetry(self, base_feature, front_joint_feature, rear_joint_feature):
        """
        Apply the symmetry to the node features
        ----------------------------------------
        Input:
        :param base_feature: shape [b, 2, T*18+2]
        :param front_joint_feature: shape [b, 6, T*5+2]
        :param rear_joint_feature: shape [b, 6, T*5+2]
        ----------------------------------------
        Output:
        :param base_feature: shape [b, 2, T*18+2]
        :param front_joint_feature: shape [b, 6, T*5+2]
        :param rear_joint_feature: shape [b, 6, T*5+2]
        """
        batch_size, num_base, num_variables_base = base_feature.shape
        base_feature = base_feature.reshape(batch_size, -1)
        base_feature = base_feature * self.linear_weights_base.view(1, -1)
        base_feature = base_feature.reshape(batch_size, num_base, num_variables_base).reshape(batch_size*num_base, num_variables_base)
        
        _, num_front_joint, num_variables_front_joint = front_joint_feature.shape
        front_joint_feature = front_joint_feature.reshape(batch_size, -1)
        front_joint_feature = front_joint_feature * self.linear_weights_front_joint.view(1, -1)
        front_joint_feature = front_joint_feature.reshape(batch_size, num_front_joint, num_variables_front_joint)
        # if self.is_critic:
        # Permute the FL and FR joints' contact information
        temp = front_joint_feature[:, :3, self.permutation_F_joint_indices].clone()
        front_joint_feature[:, :3, self.permutation_F_joint_indices] = front_joint_feature[:, 3:, self.permutation_F_joint_indices]
        front_joint_feature[:, 3:, self.permutation_F_joint_indices] = temp
        front_joint_feature = front_joint_feature.reshape(batch_size*num_front_joint, num_variables_front_joint)
        
        _, num_rear_joint, num_variables_rear_joint = rear_joint_feature.shape
        rear_joint_feature = rear_joint_feature.reshape(batch_size, -1)
        rear_joint_feature = rear_joint_feature * self.linear_weights_rear_joint.view(1, -1)
        rear_joint_feature = rear_joint_feature.reshape(batch_size, num_rear_joint, num_variables_rear_joint)
        # Permute the RL and RR joints' phase input and contact information
        temp = rear_joint_feature[:, :3, self.permutation_R_joint_indices].clone()
        rear_joint_feature[:, :3, self.permutation_R_joint_indices] = rear_joint_feature[:, 3:, self.permutation_R_joint_indices]
        rear_joint_feature[:, 3:, self.permutation_R_joint_indices] = temp
        rear_joint_feature = rear_joint_feature.reshape(batch_size*num_rear_joint, num_variables_rear_joint)
        
        return base_feature, front_joint_feature, rear_joint_feature
      
    def ms_joint_decoder(self, x):
        """
        Apply morphological symmetry to the final output for the joint nodes
        """
        return self.linear_weights_joint_actions.view(1, -1) * x
    
    def reset_parameters(self):
        """Reset all learnable parameters"""
        self.base_encoder.reset_parameters()
        self.F_joint_encoder.reset_parameters()
        self.R_joint_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if isinstance(self.decoder, nn.Sequential):
            for layer in self.decoder:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        else:
            self.decoder.reset_parameters()
