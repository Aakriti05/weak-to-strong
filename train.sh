#!/bin/bash
#
source /cmlscratch/agrawal5/bashrc_copy.sh
echo `nvidia-smi`
echo `which python`
conda activate w2s
echo `which python`

python "./ada_train_weak.py" --w2s_generalisation False --weak_model_size gpt2-medium
for Epoch in 1 2
do
    for filename in "./ada_generate_weight.py" "ada_train_weak_weight.py"
    do
        python $filename $Epoch --weak_model_size gpt2-medium
    done
done

python ada_predict.py --weak_model_size gpt2-medium

# python "ada_generate_weight.py 1 --weak_model_size gpt2-medium"
