#!/bin/bash
#
source /cmlscratch/agrawal5/bashrc_copy.sh
echo `nvidia-smi`
echo `which python`
conda activate w2s
echo `which python`

model="gpt2"
ds_name="sciq"
weighted_sampling=False

python "./ada_train_weak.py" --w2s_generalisation False --weighted_sampling $weighted_sampling --weak_model_size $model --ds_name $ds_name
for Epoch in 1 2
do
    for filename in "./ada_generate_weight.py" "ada_train_weak_weight.py"
    do
        echo $filename
        echo $Epoch
        python $filename --E $Epoch --weak_model_size $model --ds_name $ds_name
    done
done

python ada_predict.py --weak_model_size $model --ds_name $ds_name

rm -r ./sciq
