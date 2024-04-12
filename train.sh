#!/bin/bash
#
source /cmlscratch/agrawal5/bashrc_copy.sh
echo `nvidia-smi`
echo `which python`
conda activate w2s
echo `which python`

weak_model="gpt2-medium"
strong_model="gpt2-large"
ds_name="sciq"
weighted_sampling=False
weak_gt_epochs=3
strong_gt_epochs=5
loss="xent"
w2s_generalisation=True
split_by_random=False
split_by_difficulty=True
start_round=4
rounds=4
adaboost=True

[ -d "./${ds_name}" ] && mv "./${ds_name}" "./${ds_name}_data" || echo "Directory does not exist, skipping move."

directory="./results/results_${weak_model}_to_${strong_model}"

[ -d "$directory" ] && rm -r "$directory" && echo "Directory '$directory' has been deleted." || echo "Directory '$directory' does not exist."


python "./ada_train_weak.py" --split_by_difficulty $split_by_difficulty --split_by_random $split_by_random --loss_ $loss --weak_model_size $weak_model --ds_name $ds_name --weighted_sampling $weighted_sampling --gt_epochs $weak_gt_epochs --results_folder "./results/results_${weak_model}_to_${strong_model}"
for ((Epoch=1; Epoch<=rounds; Epoch++));
do
    for filename in "./ada_generate_weight.py" "ada_train_weak_weight.py"
    do
        echo $filename
        echo $Epoch
        python $filename --E $Epoch --weak_model_size $weak_model --ds_name $ds_name --weighted_sampling $weighted_sampling --gt_epochs $weak_gt_epochs --results_folder "./results/results_${weak_model}_to_${strong_model}"
    done
done
python ada_predict.py --weak_model_size $weak_model --rounds $rounds --ds_name $ds_name --w2s_generalisation $w2s_generalisation --results_folder "./results/results_${weak_model}_to_${strong_model}"
rm -rf ./${ds_name}/${weak_model}/adaboost/weak_model_${rounds}




echo "Training Baseline weak to strong !!!!!!!!!!!!!"
directory="./results/results_${weak_model}_to_${strong_model}/w2s_default"
[ -d "$directory" ] && rm -r "$directory" && echo "Directory '$directory' has been deleted." || echo "Directory '$directory' does not exist."

python ada_train_weak_to_strong.py --gt_epochs strong_gt_epochs --weak_model_size $weak_model --adaboost False --rounds $round --strong_model_size $strong_model --results_folder "./results/results_${weak_model}_to_${strong_model}" --sweep_subfolder "./w2s_default"



echo "Training Different rounds !!!!!!!!!!!!! "
for ((round=start_round; round<=rounds; round++));
do
    directory="./results/results_${weak_model}_to_${strong_model}/w2s_${adaboost}_${round}"
    [ -d "$directory" ] && rm -r "$directory" && echo "Directory '$directory' has been deleted." || echo "Directory '$directory' does not exist."

    python ada_predict.py --weak_model_size $weak_model --rounds $round --ds_name $ds_name --w2s_generalisation $w2s_generalisation --results_folder "./results/results_${weak_model}_to_${strong_model}"
    python ada_train_weak_to_strong.py --weak_gt_epochs weak_gt_epochs --gt_epochs strong_gt_epochs --weak_model_size $weak_model --adaboost $adaboost --rounds $round --strong_model_size $strong_model --results_folder "./results/results_${weak_model}_to_${strong_model}" --sweep_subfolder "./w2s_${adaboost}_${round}"
done