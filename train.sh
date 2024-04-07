#!/bin/bash
#
source /cmlscratch/agrawal5/bashrc_copy.sh
echo `nvidia-smi`
echo `which python`
conda activate w2s
echo `which python`

weak_model="gpt2"
strong_model="gpt2-medium"
ds_name="sciq"
weighted_sampling=False
gt_epochs=2
loss="xent"
w2s_generalisation=True
split_by_random=False
split_by_difficulty=True
rounds=29
adaboost=True

python "./ada_train_weak.py" --split_by_difficulty $split_by_difficulty --split_by_random $split_by_random --loss_ $loss --weak_model_size $weak_model --ds_name $ds_name --weighted_sampling $weighted_sampling --gt_epochs $gt_epochs --results_folder "./results/results_${weak_model}_to_${strong_model}"
for Epoch in {1..28};
do
    for filename in "./ada_generate_weight.py" "ada_train_weak_weight.py"
    do
        echo $filename
        echo $Epoch
        python $filename --E $Epoch --weak_model_size $weak_model --ds_name $ds_name --weighted_sampling $weighted_sampling --gt_epochs $gt_epochs --results_folder "./results/results_${weak_model}_to_${strong_model}"
    done
done

python ada_predict.py --weak_model_size $weak_model --rounds $rounds --ds_name $ds_name --w2s_generalisation $w2s_generalisation --results_folder "./results/results_${weak_model}_to_${strong_model}"

# python ada_train_weak_to_strong.py --weak_model_size $weak_model --adaboost $adaboost --rounds $rounds --strong_model_size $strong_model --results_folder "./results/results_${weak_modelname}_to_${strong_modelname}" --sweep_subfolder "./w2s_${adaboost}_${rounds}"