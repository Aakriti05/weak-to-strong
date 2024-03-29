#!/bin/bash
#
source /cmlscratch/agrawal5/bashrc_copy.sh
echo `nvidia-smi`
echo `which python`
conda activate w2s
echo `which python`

model="gpt2-medium"
strong_model="gpt2-large"
ds_name="sciq"
weighted_sampling=False
gt_epochs=3
loss="xent"
w2s_generalisation=True
rounds=11

#python "./ada_train_weak.py" --split_by_difficulty $w2s_generalisation --loss_ $loss --weak_model_size $model --ds_name $ds_name --weighted_sampling $weighted_sampling --gt_epochs $gt_epochs
#for Epoch in {1..28};
#do
#    for filename in "./ada_generate_weight.py" "ada_train_weak_weight.py"
#    do
#        echo $filename
#        echo $Epoch
#        python $filename --E $Epoch --weak_model_size $model --ds_name $ds_name --weighted_sampling $weighted_sampling --gt_epochs $gt_epochs
#    done
#done

#python ada_predict.py --weak_model_size $model --rounds $rounds --ds_name $ds_name --w2s_generalisation $w2s_generalisation --results_folder "./results"

python ada_train_weak_to_strong.py --weak_model_size $model --adaboost True --strong_model_size $strong_model
