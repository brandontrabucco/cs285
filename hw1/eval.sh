#!/bin/bash

echo "ant:"
echo "behavior_cloning:"
python $1/hw1/ant/eval.py --policy_ckpt=$1/ant/behavior_cloning/?/policy.ckpt
echo "dagger:"
python $1/hw1/ant/eval.py --policy_ckpt=$1/ant/dagger/?/policy.ckpt

echo "half_cheetah:"
echo "behavior_cloning:"
python $1/hw1/half_cheetah/eval.py --policy_ckpt=$1/half_cheetah/behavior_cloning/?/policy.ckpt
echo "dagger:"
python $1/hw1/half_cheetah/eval.py --policy_ckpt=$1/half_cheetah/dagger/?/policy.ckpt

echo "hopper:"
echo "behavior_cloning:"
python $1/hw1/hopper/eval.py --policy_ckpt=$1/hopper/behavior_cloning/?/policy.ckpt
echo "dagger:"
python $1/hw1/hopper/eval.py --policy_ckpt=$1/hopper/dagger/?/policy.ckpt

echo "humanoid:"
echo "behavior_cloning:"
python $1/hw1/humanoid/eval.py --policy_ckpt=$1/humanoid/behavior_cloning/?/policy.ckpt
echo "dagger:"
python $1/hw1/humanoid/eval.py --policy_ckpt=$1/humanoid/dagger/?/policy.ckpt

echo "walker2d:"
echo "behavior_cloning:"
python $1/hw1/walker2d/eval.py --policy_ckpt=$1/walker2d/behavior_cloning/?/policy.ckpt
echo "dagger:"
python $1/hw1/walker2d/eval.py --policy_ckpt=$1/walker2d/dagger/?/policy.ckpt
