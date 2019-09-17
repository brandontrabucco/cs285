# CS 285: Deep RL

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

Launch training sessions for behavior cloning and dagger.

```
bash cs285/cs285/hw1/ant/launch.sh cs285

bash cs285/cs285/hw1/humanoid/launch.sh cs285

bash cs285/cs285/hw1/hopper/launch.sh cs285

bash cs285/cs285/hw1/sweep/hopper/launch.sh cs285
```

Evaluate behavior cloning policies after training finishes.

```
python cs285/cs285/hw1/ant/eval.py \
    --policy_ckpt cs285/ant/behavior_cloning/*/policy.ckpt
    
python cs285/cs285/hw1/humanoid/eval.py \
    --policy_ckpt cs285/humanoid/behavior_cloning/*/policy.ckpt
    
python cs285/cs285/hw1/hopper/eval.py \
    --policy_ckpt cs285/hopper/behavior_cloning/*/policy.ckpt
    
python cs285/cs285/hw1/sweep/hopper/eval.py \
    --policy_ckpt cs285/sweep/hopper/behavior_cloning/*/policy.ckpt
```

Evaluate the expert policies after downloading finishes.

```
python cs285/cs285/hw1/ant/eval.py \
    --policy_ckpt cs285/ant/expert_policy.ckpt
    
python cs285/cs285/hw1/humanoid/eval.py \
    --policy_ckpt cs285/humanoid/expert_policy.ckpt
    
python cs285/cs285/hw1/hopper/eval.py \
    --policy_ckpt cs285/hopper/expert_policy.ckpt
```

Download json data of learning curves for dagger.

```
tensorboard --logdir cs285 --port 9999
```

Render plots using json from tensorboard.

```
python scripts/generate_plot.py \
    --output_file humanoid/dagger/dagger_learning_curve.png \
    --title "Learning Curve For DAgger" \
    --xlabel "Gradient Descent Iterations" \
    --ylabel "Return Mean" \
    --input_patterns ${FILE_PATTERN_TO_JSON_FILES} \
    --input_names dagger \
    --bars ${EXPERT_RETURN_MEAN} ${BEHAVIOR_CLONING_RETURN_MEAN} \
    --bar_names expert behavior_cloning
```
