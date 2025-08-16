import torch
import torch.nn as nn
from torch_geometric.nn import Linear, GraphConv
from copy import deepcopy

class myGNN(torch.nn.Module):
    """
    Standard GNN for the graph structure with 13 nodes (1 base + 12 joint)
    """
    def __init__(self, hidden_channels: int, num_layers: int, 
                 activation_fn = nn.ELU(), batch_size: int = 4096, is_critic: bool = False):
        """
        Implementation of a standard GNN model for the graph structure.

        Parameters:
            hidden_channels (int): Size of the node embeddings in the graph.
            num_layers (int): Number of message-passing layers.
            activation_fn (class): The activation function used between layers.
        """
        super().__init__()
        self.activation = activation_fn
        self.batch_size_default = batch_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

         # Total number of nodes (combining base and joint)
        self.num_base_nodes = 1
        self.num_front_joint_nodes = 6
        self.num_rear_joint_nodes = 6
        self.num_nodes = self.num_base_nodes + self.num_front_joint_nodes + self.num_rear_joint_nodes
        
        # Create a single edge_index for all connections
        self.edge_index = self._create_edges_index()
        
        self.is_critic = is_critic
        
        if True: # walking gait tasks
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
            | 0     | base      |  *range(18)        | 0, 1          |
            | 1     | FL-hip    | 18, 30, 42, 54, 66 | 0, 1          |
            | 2     | FL-thigh  | 19, 31, 43, 55, 66 | 0, 1          |
            | 3     | FL-knee   | 20, 32, 44, 56, 66 | 0, 1          |
            | 4     | FR-hip    | 21, 33, 45, 57, 67 | 0, 1          |
            | 5     | FR-thigh  | 22, 34, 46, 58, 67 | 0, 1          |
            | 6     | FR-knee   | 23, 35, 47, 59, 67 | 0, 1          |
            | 7     | RL-hip    | 24, 36, 48, 60, 68 | 0, 1          |
            | 8     | RL-thigh  | 25, 37, 49, 61, 68 | 0, 1          |
            | 9     | RL-knee   | 26, 38, 50, 62, 68 | 0, 1          |
            | 10    | RR-hip    | 27, 39, 51, 63, 69 | 0, 1          |
            | 11    | RR-thigh  | 28, 40, 52, 64, 69 | 0, 1          |
            | 12    | RR-knee   | 29, 41, 53, 65, 69 | 0, 1          |
            '''
            self.num_timesteps = 30   # obs_history_length
            self.dim_common_obs = 70  # num_obs
            self.node_dict = {
                0: {'name': 'base', 'common': [*range(18)], 'privileged': [0, 1]},
                1: {'name': 'FL-hip', 'common': [18, 30, 42, 54, 66], 'privileged': [0, 1]},
                2: {'name': 'FL-thigh', 'common': [19, 31, 43, 55, 66], 'privileged': [0, 1]},
                3: {'name': 'FL-knee', 'common': [20, 32, 44, 56, 66], 'privileged': [0, 1]},
                4: {'name': 'FR-hip', 'common': [21, 33, 45, 57, 67], 'privileged': [0, 1]},
                5: {'name': 'FR-thigh', 'common': [22, 34, 46, 58, 67], 'privileged': [0, 1]},
                6: {'name': 'FR-knee', 'common': [23, 35, 47, 59, 67], 'privileged': [0, 1]},
                7: {'name': 'RL-hip', 'common': [24, 36, 48, 60, 68], 'privileged': [0, 1]},
                8: {'name': 'RL-thigh', 'common': [25, 37, 49, 61, 68], 'privileged': [0, 1]},
                9: {'name': 'RL-knee', 'common': [26, 38, 50, 62, 68], 'privileged': [0, 1]},
                10: {'name': 'RR-hip', 'common': [27, 39, 51, 63, 69], 'privileged': [0, 1]},
                11: {'name': 'RR-thigh', 'common': [28, 40, 52, 64, 69], 'privileged': [0, 1]},
                12: {'name': 'RR-knee', 'common': [29, 41, 53, 65, 69], 'privileged': [0, 1]},
            }
            self.node_type_dict = {
                'base': {'node_indices': [0], 'common_input_dim': -1, 'privileged_input_dim': -1},
                'F-joint': {'node_indices': [1, 2, 3, 4, 5, 6], 'common_input_dim': -1, 'privileged_input_dim': -1},
                'R-joint': {'node_indices': [7, 8, 9, 10, 11, 12], 'common_input_dim': -1, 'privileged_input_dim': -1}
            }
            for node_type, node_values in self.node_type_dict.items():
                self.node_type_dict[node_type]['common_input_dim'] = len(self.node_dict[node_values['node_indices'][0]]['common'])
                self.node_type_dict[node_type]['privileged_input_dim'] = len(self.node_dict[node_values['node_indices'][0]]['privileged'])
        
        self.len_common_obs = self.num_timesteps * self.dim_common_obs
        
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
        self.edge_index_batch_default = self._create_edge_index_batch(self.batch_size_default).to(self.device)
        self.edge_index_batch_5 = self._create_edge_index_batch(5).to(self.device)
        self.edge_index_batch_1024 = self._create_edge_index_batch(1024).to(self.device)
        self.edge_index_batch_4096 = self._create_edge_index_batch(4096).to(self.device)
        self.edge_index_batch_6144 = self._create_edge_index_batch(6144).to(self.device)
        self.edge_index_batch_24576 = self._create_edge_index_batch(24576).to(self.device)

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
        edge_index_batch = torch.cat(edge_indices, dim=1).to(self.device)
        return edge_index_batch

    def _create_edges_index(self):
        # Define all connections in a single edge_index tensor        
        node_2_node = torch.tensor([[0, 0, 0, 0, 1, 2, 4, 5, 7, 8, 10, 11],
                                    [1, 4, 7, 10, 2, 3, 5, 6, 8, 9, 11, 12]])

        edge_index = torch.cat([
            node_2_node, node_2_node.flip(0)
        ], dim=1)
        
        return edge_index
    
    def _obs_to_graph_features(self, obs_all):
        """
        Convert observations into features for each node in a batch of graphs.
        """
        batch_size = obs_all.shape[0]
        common_obs = deepcopy(obs_all[:, :self.len_common_obs])
        common_obs = common_obs.reshape(batch_size, self.num_timesteps, -1)
        
        base_feature_list = []
        for node_idx in self.node_type_dict['base']['node_indices']:
            base_feature_list.append(common_obs[:, :, self.node_dict[node_idx]['common']].unsqueeze(1).flatten(start_dim=2)) # [b, 1, T * dim]
        base_feature = torch.cat(base_feature_list, dim=1) # [b, 1, T * dim]
        
        front_joint_feature_list = []
        for node_idx in self.node_type_dict['F-joint']['node_indices']:
            front_joint_feature_list.append(common_obs[:, :, self.node_dict[node_idx]['common']].unsqueeze(1).flatten(start_dim=2)) # [b, 1, T * dim]
        front_joint_feature = torch.cat(front_joint_feature_list, dim=1) # [b, 6, T * dim]
        
        rear_joint_feature_list = []
        for node_idx in self.node_type_dict['R-joint']['node_indices']:
            rear_joint_feature_list.append(common_obs[:, :, self.node_dict[node_idx]['common']].unsqueeze(1).flatten(start_dim=2)) # [b, 1, T * dim]
        rear_joint_feature = torch.cat(rear_joint_feature_list, dim=1) # [b, 6, T * dim]
        
        # if self.is_critic:
        privileged_obs = deepcopy(obs_all[:, self.len_common_obs:])
        privileged_base_feature_list = []
        for node_idx in self.node_type_dict['base']['node_indices']:
            privileged_base_feature_list.append(privileged_obs[:, self.node_dict[node_idx]['privileged']].unsqueeze(1)) # [b, 1, dim]
        privileged_base_feature = torch.cat(privileged_base_feature_list, dim=1) # [b, 1, dim]
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
            
        # TODO: this hard code is not good, need to be improved
        if batch_size == 5:
            edge_index = self.edge_index_batch_5
        elif batch_size == 1024:
            edge_index = self.edge_index_batch_1024
        elif batch_size == 4096:
            edge_index = self.edge_index_batch_4096
        elif batch_size == 6144:
            edge_index = self.edge_index_batch_6144
        elif batch_size == 24576:
            edge_index = self.edge_index_batch_24576
        else:
            print(f"-------Unknown batch size: {batch_size}-------")
            edge_index = self._create_edge_index_batch(batch_size).to(self.device)
        
        # Shape: [b, 1, T*19 (+23 if critic)], [b, 6, T*3 (+6 if critic)], [b, 6, T*4 (+5 if critic)], [2, b*13]
        return base_feature, front_joint_feature, rear_joint_feature, edge_index
    
    def forward(self, obs):
        batch_size = obs.shape[0]

        base_feature, front_joint_feature, rear_joint_feature, edge_index = self._obs_to_graph_features(obs)
        
        _, num_base, num_variables_base = base_feature.shape
        base_feature = base_feature.reshape(batch_size * num_base, num_variables_base)
        
        _, num_front_joint, num_variables_front_joint = front_joint_feature.shape
        front_joint_feature = front_joint_feature.reshape(batch_size * num_front_joint, num_variables_front_joint)
        
        _, num_rear_joint, num_variables_rear_joint = rear_joint_feature.shape
        rear_joint_feature = rear_joint_feature.reshape(batch_size * num_rear_joint, num_variables_rear_joint)

        # Initial feature encoding - separate for base and joint nodes
        base_encoded = self.activation(self.base_encoder(base_feature)).reshape(batch_size, self.num_base_nodes, -1)  # [batch_size, 1, hidden_channels]
        front_joint_encoded = self.activation(self.F_joint_encoder(front_joint_feature)).reshape(batch_size, self.num_front_joint_nodes, -1)  # [batch_size, 6, hidden_channels]
        rear_joint_encoded = self.activation(self.R_joint_encoder(rear_joint_feature)).reshape(batch_size, self.num_rear_joint_nodes, -1)  # [batch_size, 6, hidden_channels]
        
        x = torch.cat((base_encoded, front_joint_encoded, rear_joint_encoded), dim=1).reshape(batch_size * self.num_nodes, -1)  # [batch_size * 13, hidden_channels]
        
        # Message passing layers
        for conv in self.convs:
            # Apply convolution
            x_new = conv(x, edge_index)
            x_new = self.activation(x_new)
            x = x_new
        
        if self.is_critic:
            # For critic, use all node embeddings
            x_reshaped = x.reshape(batch_size, -1)  # [batch_size, num_nodes * hidden_channels]
            final_output = self.decoder(x_reshaped)  # [batch_size, 1]
        else:
            # For actor, use only joint node embeddings to produce actions
            # Extract only joint nodes (indices 2-13)
            joint_indices = torch.arange(self.num_base_nodes, self.num_nodes).to(self.device)
            joint_indices = joint_indices.repeat(batch_size) + torch.arange(0, batch_size * self.num_nodes, self.num_nodes).to(self.device).repeat_interleave(self.num_nodes - self.num_base_nodes)
            joint_x = x[joint_indices]  # [batch_size * 12, hidden_channels]
            
            final_output = self.decoder(joint_x).reshape(batch_size, 12)  # [batch_size, 12]

        return final_output
    
    def reset_parameters(self):
        """Reset all learnable parameters"""
        self.encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if isinstance(self.decoder, nn.Sequential):
            for layer in self.decoder:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        else:
            self.decoder.reset_parameters()
