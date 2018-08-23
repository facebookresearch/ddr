# ddr_for_tl
Decoupling Dynamics and Reward for Transfer Learning
Paper: https://arxiv.org/abs/1804.10689
Generate data for Dynamics Module: (run twice in different locations for train and test sets)
```
python generate_dynamics_data.py --env-name HalfCheetahEnv --framework rllab --random-start --N 100000 --reset --out <trainout_dir>
python generate_dynamics_data.py --env-name HalfCheetahEnv --framework rllab --random-start --N 10000 --reset --out <testout_dir>
```

Continuous space + MuJoCo/gym:


Example command to train the dynamics module:
```
python main.py --train-dynamics --train-set <traindata_dir> --test-set <testdata_dir> --train-batch 2500  --test-batch 250 --log-interval 10 --dim 200 --batch-size 512 --num-epochs 100 --env-name HalfCheetahEnv --framework rllab
```


Example command to train the rewards module:
```
python main.py --train-reward --env-name HalfCheetahEnv --framework rllab --dynamics-module <model_dir> --dim 200 --num-episodes 10000000
```

Transfer
Dynamics: Include flag "--from-file {xml_file}"
Reward: Include flag "--neg-reward"


## Tensorboard

Make sure you have [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) installed in your current conda installation. You can install it by executing following command:
```
pip install tensorboardX
```

Specify the directory where you want to log the tensorboard summaries (logs) with the ```--log-dir``` flag, eg:

```
python main.py --train-reward --dynamics-module  /private/home/hsatija/ddr_for_tl/data/SwimmerMazeEnvmazeid0length1/_entropy_coef0.0_dec_loss_coef0.1_forward_loss_coef10_rollout3_train_size10000/dynamics_module_epoch10.pt --dim 10 --framework rllab --env-name SwimmerMazeEnv --out ./data/ --log-dir ./runs/
```

The tensorboard logs (summaries/events) will be published in <log-dir>/tb_logs/ directory. Launch the tensorboard server on the devfair machine,
```
tensorboard --logdir <path-to-log-dir>/tb_logs/ --port 6006
```


You can then set up port forwarding to access the tensorboard on the local machine.

## License
Attribution-NonCommercial 4.0 International as found in the LICENSE file.
