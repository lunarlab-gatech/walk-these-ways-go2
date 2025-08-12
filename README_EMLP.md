# Walk-these-way-go2 with EMLP



## Install requirements

Install MorphoSymm

```
cd MorphoSymm/
pip install -e .
```





## Train an EMLP policy

```
cd scripts/
python -u train_emlp.py
```

If you want to change the command velocity, you need to change `lin_vel_x/lin_vel_y/ang_vel_yaw`, `limit_vel_x/limit_vel_y/limit_vel_yaw`, `bins_vel_x/bins_vel_y/bins_vel_yaw` in `train_emlp.py`.

Issue:
```bash
(wtw) swei303@aellnar08106d:~/Documents/proj/walk-these-ways-go2/scripts$ python -u train_emlp.py 
✓ created a new logging client
Dashboard: http://app.dash.ml/gait-conditioned-agility/2025-08-08/train_emlp/004045.614818
Log_directory: /home/swei303/Documents/proj/walk-these-ways-go2/runs
Importing module 'gym_38' (/home/swei303/Documents/proj/walk-these-ways-go2/isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_38.so)
Setting GYM_USD_PLUG_INFO_PATH to /home/swei303/Documents/proj/walk-these-ways-go2/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugInfo.json
PyTorch version 1.13.1+cu117
Device count 1
/home/swei303/Documents/proj/walk-these-ways-go2/isaacgym/python/isaacgym/_bindings/src/gymtorch
Using /home/swei303/.cache/torch_extensions/py38_cu117 as PyTorch extensions root...
Emitting ninja build file /home/swei303/.cache/torch_extensions/py38_cu117/gymtorch/build.ninja...
Building extension module gymtorch...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module gymtorch...
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
Users of this version of Gym should be able to simply replace 'import gym' with 'import gymnasium as gym' in the vast majority of cases.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
Traceback (most recent call last):
  File "train_emlp.py", line 260, in <module>
    train_go2(headless=True)
  File "train_emlp.py", line 13, in train_go2
    from go2_gym_learn.ppo_cse import PPOEMLPRunner
  File "/home/swei303/Documents/proj/walk-these-ways-go2/go2_gym_learn/ppo_cse/__init__.py", line 328, in <module>
    from .actor_critic_symmetric import ActorCriticSymm
  File "/home/swei303/Documents/proj/walk-these-ways-go2/go2_gym_learn/ppo_cse/actor_critic_symmetric.py", line 38, in <module>
    from morpho_symm.utils.robot_utils import load_symmetric_system
  File "/home/swei303/Documents/proj/walk-these-ways-go2/MorphoSymm/morpho_symm/utils/robot_utils.py", line 18, in <module>
    from morpho_symm.robots.PinBulletWrapper import PinBulletWrapper
  File "/home/swei303/Documents/proj/walk-these-ways-go2/MorphoSymm/morpho_symm/robots/PinBulletWrapper.py", line 16, in <module>
    import pinocchio
  File "/home/swei303/miniconda3/envs/wtw/lib/python3.8/site-packages/cmeel.prefix/lib/python3.8/site-packages/pinocchio/__init__.py", line 72, in <module>
    from .robot_wrapper import RobotWrapper
  File "/home/swei303/miniconda3/envs/wtw/lib/python3.8/site-packages/cmeel.prefix/lib/python3.8/site-packages/pinocchio/robot_wrapper.py", line 7, in <module>
    from .shortcuts import (
  File "/home/swei303/miniconda3/envs/wtw/lib/python3.8/site-packages/cmeel.prefix/lib/python3.8/site-packages/pinocchio/shortcuts.py", line 16, in <module>
    ) -> tuple[pin.Model, pin.GeometryModel, pin.GeometryModel]:
TypeError: 'type' object is not subscriptable
```

Solution: degrade `pin` to 3.3.1
```
pip install pin==3.3.1
```



## Test an EMLP policy

```	
cd scripts/
python -u play_emlp.py
```

Remember to change `label=your/policy/path` at line 134. You can also specify the command velocities at line 145.