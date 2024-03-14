#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --mem=128gb 
#SBATCH --ntasks=4
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --output=train_gpt2_large.log

source /nfshomes/anirudhs/.bashrc
echo `nvidia-smi`
echo `which python`
conda activate w2s
echo `which python`
model="gpt2-large"
python "./ada_train_weak.py" --weak_model_size $model
for Epoch in 1 2
do
    for filename in "./ada_generate_weight.py" "ada_train_weak_weight.py"
    do
        echo $filename
        echo $Epoch
        python $filename --E $Epoch --weak_model_size $model
    done
done

python ada_predict.py --weak_model_size $model
