#!/bin/bash
python $1/scripts/parse_expert.py \
  --expert_policy_file=$1/experts/Walker2d.pkl \
  --output_policy_file=$1/walker2d/expert_policy.ckpt
python $1/hw1/walker2d/dagger.py &
python $1/hw1/walker2d/behavior_cloning.py &
wait
