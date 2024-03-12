#!/bin/bash
#
source /cmlscratch/agrawal5/bashrc_copy.sh
echo `nvidia-smi`
echo `which python`
conda activate w2s
echo `which python`

model="gpt2"
ds_name="sciq"
weighted_sampling=True
gt_epochs=2
loss="xent"
w2s_generalisation=True

python "./ada_train_weak.py" --w2s_generalisation $w2s_generalisation --weighted_sampling $weighted_sampling --loss_ $loss --weak_model_size $model --ds_name $ds_name --gt_epochs $gt_epochs
for Epoch in 1 2
do
    for filename in "./ada_generate_weight.py" "ada_train_weak_weight.py"
    do
        echo $filename
        echo $Epoch
        python $filename --E $Epoch --weak_model_size $model --ds_name $ds_name --weighted_sampling $weighted_sampling --gt_epochs $gt_epochs
    done
done

python ada_predict.py --weak_model_size $model --ds_name $ds_name --w2s_generalisation $w2s_generalisation

# rm -r ./sciq
