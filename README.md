# CS 285: Deep Reinforcement Learning

A minimalist reinforcement learning package for TensorFlow 2.0. Have fun! -Brandon

## Setup

Clone and install with pip.

``` 
git clone git@github.com:brandontrabucco/cs285.git
pip install -e cs285
```

## Assignment 1

To launch the first assignment, first download the expert policies.

```
bash cs285/cs285/scripts/download_experts.sh cs285
```

Evaluate the expert policies after downloading finishes.

```
bash cs285/cs285/hw1/ant/eval.py --policy_ckpt cs285/ant/expert_policy.ckpt
bash cs285/cs285/hw1/humanoid/eval.py --policy_ckpt cs285/humanoid/expert_policy.ckpt
bash cs285/cs285/hw1/hopper/eval.py --policy_ckpt cs285/hopper/expert_policy.ckpt
```

Launch training sessions for behavior cloning and dagger.

```
bash cs285/cs285/hw1/ant/launch.sh cs285
bash cs285/cs285/hw1/humanoid/launch.sh cs285
bash cs285/cs285/hw1/hopper/launch.sh cs285
```

Evaluate the policies after training finishes.

```
bash cs285/cs285/hw1/ant/eval.py --policy_ckpt cs285/ant/behavior_cloning/?/policy.ckpt
bash cs285/cs285/hw1/humanoid/eval.py --policy_ckpt cs285/humanoid/behavior_cloning/?/policy.ckpt
bash cs285/cs285/hw1/hopper/eval.py --policy_ckpt cs285/hopper/behavior_cloning/?/policy.ckpt
```

Perform a small hyperparameter sweep on the number of demonstrations.

```
bash cs285/cs285/hw1/sweep/hopper/launch.sh cs285
```

Evaluate the policies after training finishes.

```
bash cs285/cs285/hw1/sweep/hopper/eval.py --policy_ckpt cs285/sweep/hopper/behavior_cloning/?/policy.ckpt
```

Generate plots of learning curves for behavior cloning and dagger.

```
tensorboard --logdir cs285 --port 9999
```
