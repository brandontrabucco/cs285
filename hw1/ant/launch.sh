#!/bin/bash
python $1/scripts/parse_expert_policy.py \
  --expert_policy_file=$1/experts/Ant.pkl \
  --output_policy_file=$1/ant/expert_policy.ckpt
python $1/hw1/ant/dagger.py &
python $1/hw1/ant/behavior_cloning.py &
wait
