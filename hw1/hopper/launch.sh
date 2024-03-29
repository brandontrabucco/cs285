#!/bin/bash
python $1/scripts/parse_expert.py \
  --expert_policy_file=$1/experts/Hopper.pkl \
  --output_policy_file=$1/hopper/expert_policy.ckpt
python $1/hw1/hopper/dagger.py &
python $1/hw1/hopper/sac.py &
wait
