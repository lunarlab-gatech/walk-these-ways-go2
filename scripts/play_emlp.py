import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from go2_gym.envs import *
from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.go2.go2_config import config_go2
from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm
import os

import re
def extract_index(filename):
    # For sorting the frames when generating gif.
    dirname, basename = os.path.split(filename)
    name, ext = os.path.splitext(basename)
    match = re.search(r'(\d+)', name)
    return int(match.group(1)) if match else float('inf')



def load_policy(logdir, env):
    import os
    from go2_gym_learn.ppo_cse import PPOEMLPRunner

    runner = PPOEMLPRunner(env, device = 'cpu')

    runner.alg.actor_critic.adaptation_module.load_state_dict(torch.load(os.path.join(logdir, 'checkpoints/adaptation_module.pkl')), strict = False)
    runner.alg.actor_critic.actor_body.load_state_dict(torch.load(os.path.join(logdir, 'checkpoints/actor_body.pkl')), strict = False)

    adaptation_module = runner.alg.actor_critic.adaptation_module
    adaptation_module.eval()

    actor_body = runner.alg.actor_critic.actor_body
    actor_body.eval()
    
    in_field_type = runner.alg.actor_critic.in_field_type

    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        x = in_field_type(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        action = actor_body.forward(x).tensor
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[-1] # Latest

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        # print(pkl_cfg.keys())
        print(pkl_cfg.items())
        cfg = pkl_cfg["Cfg"]

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                # print(key, Cfg.key)
                if isinstance(cfg[key], dict):
                    for key2, value2 in cfg[key].items():
                        setattr(getattr(Cfg, key), key2, value2)

    Cfg.record.record = True
    Cfg.record.folder = 'exported_image/'
    if not os.path.exists(Cfg.record.folder):
        os.makedirs(Cfg.record.folder)

    Cfg.commands.yaw_command_curriculum = True

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 5
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    # default control_typw is "actuator_net", you can also switch it to "P" to enable joint PD control
    Cfg.control.control_type = "actuator_net" 
    Cfg.asset.flip_visual_attachments = True


    from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go2_gym_learn.ppo_cse.actor_critic_emlp import ActorCriticEMLP

    policy = load_policy(logdir, env)

    return env, policy


def play_go2(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go2_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    # TODO: 修改label
    # label = "gait-conditioned-agility/pretrain-v0/train"
    # label = "gait-conditioned-agility/pretrain-go2/train"
    label = "gait-conditioned-agility/2025-08-08/train_emlp"

    env, policy = load_env(label, headless=headless)

    num_eval_steps = 500 #250
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    # x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, 0.0, 0.0
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, -0.5, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0 #3.0
    
    gait_name = list(gaits.keys())[2]
    save_name = f"{gait_name}_cmds=[{x_vel_cmd},{y_vel_cmd},{yaw_vel_cmd}]"

    # gait = torch.tensor(gaits["trotting"])
    gait = torch.tensor(gaits[gait_name])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))
    measured_y_vels = np.zeros(num_eval_steps)
    target_y_vels = np.ones(num_eval_steps) * y_vel_cmd
    measured_yaw_vels = np.zeros(num_eval_steps)
    target_yaw_vels = np.ones(num_eval_steps) * yaw_vel_cmd
    ###### -----------ldt---------------
    joint_torques = np.zeros((num_eval_steps, 12))

    obs = env.reset()

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        measured_y_vels[i] = env.base_lin_vel[0, 1]
        measured_yaw_vels[i] = env.base_ang_vel[0, 2]
        joint_positions[i] = env.dof_pos[0, :].cpu()
        ###### -----------ldt---------------
        # joint_torques[i] = env.torques.detach().cpu().numpy()

    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(1, 1, figsize=(12, 5))
    # axs.plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_yaw_vels, color='black', linestyle="-", label="Measured")
    # axs.plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_yaw_vels, color='black', linestyle="--", label="Desired")
    # axs.legend()
    # axs.set_title("Yaw Angular Velocity")
    # axs.set_xlabel("Time (s)")
    # axs.set_ylabel("Velocity (rad/s)")

    axs.plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_y_vels, color='black', linestyle="-", label="Measured")
    axs.plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_y_vels, color='black', linestyle="--", label="Desired")
    axs.legend()
    axs.set_title("Side Linear Velocity")
    axs.set_xlabel("Time (s)")
    axs.set_ylabel("Velocity (m/s)")

    # axs.plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    # axs.plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    # axs.legend()
    # axs.set_title("Forward Linear Velocity")
    # axs.set_xlabel("Time (s)")
    # axs.set_ylabel("Velocity (m/s)")

    # axs[2].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    # axs[2].set_title("Joint Positions")
    # axs[2].set_xlabel("Time (s)")
    # axs[2].set_ylabel("Joint Position (rad)")

    # axs[3].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_torques, linestyle="-", label="Measured")
    # axs[3].set_title("Joint Torques")
    # axs[3].set_xlabel("Time (s)")
    # axs[3].set_ylabel("Joint Torques (Nm)")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{save_name}.png')
    plt.close()



    if Cfg.record.record:
        from PIL import Image
        import glob

        # List of image file paths (frames)
        frames = list(sorted(glob.glob(os.path.join(Cfg.record.folder, '*.png')), key = extract_index))
        # print(frames)

        # Open images and convert to RGB (if needed)
        images = [Image.open(f).convert('RGBA') for f in frames]
        save_gif_path = f'{save_name}.gif'

        # Save as GIF
        images[0].save(
            save_gif_path,
            save_all=True,
            append_images=images[1:],
            duration=100,  # Duration per frame in ms
            loop=0         # 0 means infinite loop
        )
        print(f'Save gif to {save_gif_path}!')

        for f in frames:
            os.remove(f)


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go2(headless=True)
