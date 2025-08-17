import time
from collections import deque
import copy
import os
import wandb
import torch
from params_proto import PrefixProto

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from go2_gym_learn.ppo.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


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


class Runner:

    def __init__(self, env, device='cpu', training_root=None):
        from .ppo import PPO

        self.device = device
        self.env = env
        self.training_root = training_root

        actor_critic = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions,
                                      ).to(self.device)

        # TODO: resume from checkpoint using wandb, check if it works
        if RunnerArgs.resume:
            # load pretrained weights from resume_path
            # Use wandb's restore functionality
            api = wandb.Api()
            run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{RunnerArgs.resume_path}")
            checkpoint_file = run.file("checkpoints/ac_weights_last.pt")
            checkpoint_file.download()
            weights = torch.load("checkpoints/ac_weights_last.pt")
            actor_critic.load_state_dict(state_dict=weights)

            if hasattr(self.env, "curricula") and RunnerArgs.resume_curriculum:
                # load curriculum state
                distribution_file = run.file("curriculum/distribution.pkl")
                distribution_file.download()
                import pickle
                with open("curriculum/distribution.pkl", "rb") as f:
                    distributions = pickle.load(f)
                distribution_last = distributions[-1]["distribution"]
                gait_names = [key[8:] if key.startswith("weights_") else None for key in distribution_last.keys()]
                for gait_id, gait_name in enumerate(self.env.category_names):
                    self.env.curricula[gait_id].weights = distribution_last[f"weights_{gait_name}"]
                    print(gait_name)

        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = 0

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500, eval_expert=False):        
        # Create all necessary directories at the start of training
        import os
        print("Creating necessary directories...")
        print(f"Training root: {self.training_root}")
        os.makedirs("./tmp/legged_data", exist_ok=True)
        print("Directories created successfully!")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs])
                    if eval_expert:
                        actions_eval = self.alg.actor_critic.act_teacher(obs_history[num_train_envs:],
                                                                         privileged_obs[num_train_envs:])
                    else:
                        actions_eval = self.alg.actor_critic.act_student(obs_history[num_train_envs:])
                    ret = self.env.step(torch.cat((actions_train, actions_eval), dim=0))
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)

                    if 'train/episode' in infos:
                        # Log training metrics with wandb
                        wandb.log({
                            "train/episode": infos['train/episode']
                        }, step=it)

                    if 'eval/episode' in infos:
                        # Log evaluation metrics with wandb
                        wandb.log({
                            "eval/episode": infos['eval/episode']
                        }, step=it)

                    if 'curriculum' in infos:

                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                        new_ids_eval = new_ids[new_ids >= num_train_envs]
                        rewbuffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())
                        lenbuffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_eval] = 0
                        cur_episode_length[new_ids_eval] = 0

                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])

                if it % curriculum_dump_freq == 0:
                    # First save to local file
                    import pickle
                    
                    # Save curriculum info
                    curriculum_info = {
                        "iteration": it,
                        **caches.slot_cache.get_summary(),
                        **caches.dist_cache.get_summary()
                    }
                    
                    curriculum_file_path = f"{self.training_root}/curriculum/info_{it}.pkl"
                    with open(curriculum_file_path, "wb") as f:
                        pickle.dump(curriculum_info, f)
                    
                    # Upload file to wandb
                    wandb.save(curriculum_file_path)
                    
                    # Save distribution info
                    if 'curriculum/distribution' in infos:
                        distribution_info = {
                            "iteration": it,
                            "distribution": distribution
                        }
                        
                        distribution_file_path = f"{self.training_root}/curriculum/distribution_{it}.pkl"
                        with open(distribution_file_path, "wb") as f:
                            pickle.dump(distribution_info, f)
                        
                        # Upload file to wandb
                        wandb.save(distribution_file_path)

            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            # Log training metrics with wandb
            wandb.log({
                "time_elapsed": time.time() - start,
                "time_iter": learn_time,
                "adaptation_loss": mean_adaptation_module_loss,
                "mean_value_loss": mean_value_loss,
                "mean_surrogate_loss": mean_surrogate_loss,
                "mean_decoder_loss": mean_decoder_loss,
                "mean_decoder_loss_student": mean_decoder_loss_student,
                "mean_decoder_test_loss": mean_decoder_test_loss,
                "mean_decoder_test_loss_student": mean_decoder_test_loss_student,
                "mean_adaptation_module_test_loss": mean_adaptation_module_test_loss,
                "collection_time": collection_time,
                "learn_time": learn_time
            }, step=it)

            if RunnerArgs.save_video_interval:
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            
            # Log at log frequency with wandb
            if it % RunnerArgs.log_freq == 0:
                wandb.log({
                    "timesteps": self.tot_timesteps,
                    "iterations": it
                }, step=it)

            if it % RunnerArgs.save_interval == 0:
                # Create necessary directories
                # os.makedirs("checkpoints", exist_ok=True)
                # os.makedirs("videos", exist_ok=True)
                
                # Save checkpoint
                checkpoint_path = f"{self.training_root}/checkpoints/ac_weights_{it:06d}.pt"
                torch.save(self.alg.actor_critic.state_dict(), checkpoint_path)
                
                # Copy as latest checkpoint
                latest_checkpoint_path = f"{self.training_root}/checkpoints/ac_weights_last.pt"
                torch.save(self.alg.actor_critic.state_dict(), latest_checkpoint_path)

                path = f'{self.training_root}/tmp/legged_data'
                os.makedirs(path, exist_ok=True)

                # Save model state dicts
                adaptation_module_path = f'{path}/adaptation_module_latest.pt'
                torch.save(self.alg.actor_critic.adaptation_module.state_dict(), adaptation_module_path)

                body_path = f'{path}/body_latest.pt'
                torch.save(self.alg.actor_critic.actor_body.state_dict(), body_path)

                # Save files with wandb
                wandb.save(checkpoint_path)
                wandb.save(adaptation_module_path)
                wandb.save(body_path)
                
            self.current_learning_iteration += num_learning_iterations

        # Final save
        # os.makedirs(f"{self.training_root}/checkpoints", exist_ok=True)
        # os.makedirs(f"{self.training_root}/videos", exist_ok=True)

        final_checkpoint_path = f"{self.training_root}/checkpoints/ac_weights_{it:06d}.pt"
        torch.save(self.alg.actor_critic.state_dict(), final_checkpoint_path)
        latest_checkpoint_path = f"{self.training_root}/checkpoints/ac_weights_last.pt"
        torch.save(self.alg.actor_critic.state_dict(), latest_checkpoint_path)

        path = f'{self.training_root}/tmp/legged_data'
        os.makedirs(path, exist_ok=True)

        adaptation_module_path = f'{path}/adaptation_module_latest.pt'
        torch.save(self.alg.actor_critic.adaptation_module.state_dict(), adaptation_module_path)

        body_path = f'{path}/body_latest.pt'
        torch.save(self.alg.actor_critic.actor_body.state_dict(), body_path)

        # Save final files with wandb
        wandb.save(final_checkpoint_path)
        wandb.save(adaptation_module_path)
        wandb.save(body_path)

    def log_video(self, it):
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()
            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            
            # Create video directory first
            # os.makedirs(f"{self.training_root}/videos", exist_ok=True)
            
            try:
                # Use imageio for better compatibility (same as ml_logger)
                import tempfile
                import imageio
                from skimage import img_as_ubyte
                
                # Convert frames to video format using imageio
                video_path = f"{self.training_root}/videos/{it:05d}.mp4"
                
                # Convert frames to uint8 format (same as ml_logger)
                frame_stack = img_as_ubyte(frames)
                
                # Save video using imageio (same method as ml_logger)
                try:
                    imageio.v3.imwrite(video_path, frame_stack, fps=1/self.env.dt)
                except imageio.core.NeedDownloadError:
                    # Download ffmpeg if needed (same as ml_logger)
                    imageio.plugins.ffmpeg.download()
                    imageio.v3.imwrite(video_path, frame_stack, fps=1/self.env.dt)
                
                # Check if video file was created successfully
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    # Upload to wandb
                    wandb.log({"video": wandb.Video(video_path, fps=1/self.env.dt, format="mp4")}, step=it)
                    print(f"Video saved successfully: {video_path}")
                else:
                    print(f"Warning: Video file was not created or is empty: {video_path}")
                    
            except Exception as e:
                print(f"Error saving video: {e}")

        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()
            if len(frames) > 0:
                self.env.pause_recording_eval()
                print("LOGGING EVAL VIDEO")
                
                # Create video directory (already created above, but ensure it exists)
                # os.makedirs(f"{self.training_root}/videos", exist_ok=True)
                
                try:
                    # Log evaluation video with wandb using imageio
                    video_path = f"{self.training_root}/videos/{it:05d}_eval.mp4"
                    
                    # Convert frames to uint8 format (same as ml_logger)
                    frame_stack = img_as_ubyte(frames)
                    
                    # Save video using imageio (same method as ml_logger)
                    try:
                        imageio.v3.imwrite(video_path, frame_stack, fps=1/self.env.dt, format='mp4')
                    except imageio.core.NeedDownloadError:
                        # Download ffmpeg if needed (same as ml_logger)
                        imageio.plugins.ffmpeg.download()
                        imageio.v3.imwrite(video_path, frame_stack, fps=1/self.env.dt, format='mp4')
                    
                    # Check if video file was created successfully
                    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                        # Upload to wandb
                        wandb.log({"eval_video": wandb.Video(video_path, fps=1/self.env.dt, format="mp4")}, step=it)
                        print(f"Eval video saved successfully: {video_path}")
                    else:
                        print(f"Warning: Eval video file was not created or is empty: {video_path}")
                        
                except Exception as e:
                    print(f"Error saving eval video: {e}")
        
    '''
    def log_video(self, it):
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()
            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            
            # Log video with wandb
            import cv2
            import numpy as np
            
            # Convert frames to video format
            video_path = f"{self.training_root}/videos/{it:05d}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 1/self.env.dt, (frames[0].shape[1], frames[0].shape[0]))
            
            for frame in frames:
                # Assume frame is in RGB format, convert to BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            
            # Upload to wandb
            wandb.log({"video": wandb.Video(video_path, fps=1/self.env.dt, format="mp4")}, step=it)

        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()
            if len(frames) > 0:
                self.env.pause_recording_eval()
                print("LOGGING EVAL VIDEO")
                
                # Create video directory
                os.makedirs(f"{self.training_root}/videos", exist_ok=True)
                
                # Log evaluation video with wandb
                video_path = f"{self.training_root}/videos/{it:05d}_eval.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 1/self.env.dt, (frames[0].shape[1], frames[0].shape[0]))
                
                for frame in frames:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()
                
                # Upload to wandb
                wandb.log({"eval_video": wandb.Video(video_path, fps=1/self.env.dt, format="mp4")}, step=it)
    '''

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert