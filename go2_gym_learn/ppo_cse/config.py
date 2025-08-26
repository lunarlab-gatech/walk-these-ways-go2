from params_proto import PrefixProto

class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 100
    log_freq = 10

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True
    
class PPO_Args(PrefixProto):
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
        hidden_dim = 64
        num_layers = 8
        activation = 'elu'
        num_envs = 4096 # NOTE: this should be set to the number of environments in the environment
        num_env_mini_batch = num_envs * RunnerArgs.num_steps_per_env // PPO_Args.num_mini_batches
        
    # EMLP specific configuration
    class emlp(PrefixProto, cli=False):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        
    # MS-GNN specific configuration
    class msgnn(PrefixProto, cli=False):
        hidden_dim = 128
        num_layers = 8
        activation = 'elu'
        num_envs = 4096 # NOTE: this should be set to the number of environments in the environment
        num_env_mini_batch = num_envs * RunnerArgs.num_steps_per_env // PPO_Args.num_mini_batches