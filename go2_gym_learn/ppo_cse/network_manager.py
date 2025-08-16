from abc import ABC, abstractmethod
from typing import Dict, Type
import torch.nn as nn

class NetworkArchitecture(ABC):
    """Abstract base class for network architectures"""
    
    @abstractmethod
    def create_actor(self, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
        """Create the actor network"""
        pass
    
    @abstractmethod
    def create_critic(self, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
        """Create the critic network"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the network architecture"""
        pass

class MLPArchitecture(NetworkArchitecture):
    """Standard MLP architecture"""
    
    def __init__(self, input_dim: int, action_dim: int, actor_hidden_dims: list, critic_hidden_dims: list, activation=nn.ReLU):
        '''
        input_dim: the dimension of the input
        action_dim: the dimension of the action
        actor_hidden_dims: [list], the dimension of the hidden layers for actor
        critic_hidden_dims: [list], the dimension of the hidden layers for critic
        activation: the activation function
        '''
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims = critic_hidden_dims
        self.activation = activation
    
    def create_actor(self, **kwargs) -> nn.Module:
        layers = []
        layers.append(nn.Linear(self.input_dim, self.actor_hidden_dims[0]))
        layers.append(self.activation())
        
        for l in range(len(self.actor_hidden_dims)):
            if l == len(self.actor_hidden_dims) - 1:
                layers.append(nn.Linear(self.actor_hidden_dims[l], self.action_dim))
            else:
                layers.append(nn.Linear(self.actor_hidden_dims[l], self.actor_hidden_dims[l + 1]))
                layers.append(self.activation())
        
        return nn.Sequential(*layers)
    
    def create_critic(self, **kwargs) -> nn.Module:
        layers = []
        layers.append(nn.Linear(self.input_dim, self.critic_hidden_dims[0]))
        layers.append(self.activation())
        
        for l in range(len(self.critic_hidden_dims)):
            if l == len(self.critic_hidden_dims) - 1:
                layers.append(nn.Linear(self.critic_hidden_dims[l], 1))
            else:
                layers.append(nn.Linear(self.critic_hidden_dims[l], self.critic_hidden_dims[l + 1]))
                layers.append(self.activation())
        
        return nn.Sequential(*layers)
        
    def get_name(self) -> str:
        return "MLP"

class GNNArchitecture(NetworkArchitecture):
    """GNN architecture"""
    
    def __init__(self, hidden_dim: int, num_layers: int, activation=nn.ELU):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
    
    def create_actor(self, **kwargs) -> nn.Module:
        from .my_gnn import myGNN
        return myGNN(
            hidden_channels=self.hidden_dim,
            num_layers=self.num_layers,
            activation_fn=self.activation,
            is_critic=False
        )
    
    def create_critic(self, **kwargs) -> nn.Module:
        from .my_gnn import myGNN
        return myGNN(
            hidden_channels=self.hidden_dim,
            num_layers=self.num_layers,
            activation_fn=self.activation,
            is_critic=True
        )
    
    def get_name(self) -> str:
        return "GNN"

class EMLPArchitecture(NetworkArchitecture):
    """EMLP architecture (Equivariant MLP)"""
    
    def __init__(self, hidden_dims: list, group_type: str = "SO3"):
        self.hidden_dims = hidden_dims
        self.group_type = group_type
    
    def create_actor(self, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
        # Here you should create your EMLP implementation
        # Currently returns a placeholder
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], output_dim)
        )
    
    def create_critic(self, input_dim: int, output_dim: int, **kwargs) -> nn.Module:
        return self.create_actor(input_dim, output_dim, **kwargs)
    
    def get_name(self) -> str:
        return "EMLP"

class NetworkManager:
    """Network architecture manager"""
    
    def __init__(self):
        self.architectures: Dict[str, Type[NetworkArchitecture]] = {}
        self._register_default_architectures()
    
    def _register_default_architectures(self):
        """Register default network architectures"""
        self.register_architecture("mlp", MLPArchitecture)
        self.register_architecture("gnn", GNNArchitecture)
        self.register_architecture("emlp", EMLPArchitecture)
    
    def register_architecture(self, name: str, architecture_class: Type[NetworkArchitecture]):
        """Register a new network architecture"""
        self.architectures[name] = architecture_class
    
    def get_architecture(self, name: str, **kwargs) -> NetworkArchitecture:
        """Get an instance of the specified network architecture"""
        if name not in self.architectures:
            raise ValueError(f"Unknown architecture: {name}. Available: {list(self.architectures.keys())}")
        
        architecture_class = self.architectures[name]
        
        # automatically extract and validate parameters based on the architecture type
        validated_kwargs = self._validate_and_extract_params(name, **kwargs)
        
        return architecture_class(**validated_kwargs)
    
    def _validate_and_extract_params(self, architecture_name: str, **kwargs) -> dict:
        """Validate and extract architecture-specific parameters"""
        if architecture_name == "mlp":
            required_params = ['input_dim', 'action_dim', 'actor_hidden_dims', 'critic_hidden_dims']
            return self._extract_mlp_params(required_params, **kwargs)
        
        elif architecture_name == "gnn":
            required_params = ['hidden_dim', 'num_layers']
            return self._extract_gnn_params(required_params, **kwargs)
        
        elif architecture_name == "emlp":
            raise NotImplementedError("EMLP is not implemented yet")
        
        else:
            raise ValueError(f"Unknown architecture: {architecture_name}")
    
    def _extract_mlp_params(self, required_params: list, **kwargs) -> dict:
        """Extract MLP parameters"""
        # Get MLP config from AC_Args
        from .actor_critic import AC_Args
        
        return {
            'input_dim': kwargs.get('input_dim'),
            'action_dim': kwargs.get('action_dim'),
            'actor_hidden_dims': AC_Args.mlp.actor_hidden_dims,
            'critic_hidden_dims': AC_Args.mlp.critic_hidden_dims,
            'activation': self._get_activation(AC_Args.mlp.activation)
        }
    
    def _extract_gnn_params(self, required_params: list, **kwargs) -> dict:
        """Extract GNN parameters"""
        from .actor_critic import AC_Args
        
        return {
            'hidden_dim': AC_Args.gnn.hidden_dim,
            'num_layers': AC_Args.gnn.num_layers,
            'activation': self._get_activation(AC_Args.gnn.activation)
        }
    
    def _extract_emlp_params(self, required_params: list, **kwargs) -> dict:
        """Extract EMLP parameters"""
        from .actor_critic import AC_Args
        
        return {
            'hidden_dims': AC_Args.emlp.hidden_dims,
            'group_type': AC_Args.emlp.group_type,
            'activation': self._get_activation(AC_Args.emlp.activation)
        }
    
    def _get_activation(self, activation_name: str):
        """Get activation function by name"""
        activation_map = {
            'elu': nn.ELU(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activation_map.get(activation_name, nn.ELU)
    
    def list_architectures(self) -> list:
        """List all available network architectures"""
        return list(self.architectures.keys())