#!/bin/bash
python $1/scripts/parse_expert_policy.py --expert_policy_file=$1/experts/HalfCheetah.pkl \
                                         --output_policy_file=$1/half_cheetah/expert_policy.ckpt
python $1/hw1/dagger.py
python $1/hw1/behavior_cloning.py
