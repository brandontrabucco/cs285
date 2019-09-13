#!/bin/bash
python $1/scripts/parse_expert.py \
  --expert_policy_file=$1/experts/Humanoid.pkl \
  --output_policy_file=$1/humanoid/expert_policy.ckpt
python $1/hw1/humanoid/dagger.py &
python $1/hw1/humanoid/behavior_cloning.py &
wait
