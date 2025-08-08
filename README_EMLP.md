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



## Test an EMLP policy

```	
cd scripts/
python -u play_emlp.py
```

Remember to change `label=your/policy/path` at line 134. You can also specify the command velocities at line 145.