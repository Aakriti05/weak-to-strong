#!/bin/bash
#
source /cmlscratch/agrawal5/bashrc_copy.sh
echo `nvidia-smi`
echo `which python`
conda activate w2s
echo `which python`

python "./ada_train_weak.py" 
for Epoch in 1 2 3 4
do
    for filename in "./ada_generate_weight.py" "ada_train_weak_weight.py"
    do
        python $filename $Epoch
    done
done
