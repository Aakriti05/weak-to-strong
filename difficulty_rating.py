import json
import os
from typing import Dict, List, Optional, Sequence, Union
import fire
import shutil
from tqdm.auto import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import torch
import weak_to_strong.logger as logger
from weak_to_strong.common import get_tokenizer
from datasets import concatenate_datasets
from weak_to_strong.datasets import (VALID_DATASETS, load_dataset, 
                                     tokenize_dataset)
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss
from weak_to_strong.train import ModelConfig, train_and_save_model


results_folder = "./results/"
loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
}

def train_model(
    model_config: ModelConfig,
    train_ds: torch.utils.data.Dataset,
    test_ds: torch.utils.data.Dataset,
    *,
    loss_type: str,
    label: str,
    subpath,
    lr,
    eval_batch_size,
    epochs=1,
    inference_ds: Optional[torch.utils.data.Dataset] = None,
    linear_probe: bool = False,
    optimizer_name: str = "adam",
):
    save_path = os.path.join(results_folder, subpath)
    linprobe_str = "_linprobe" if linear_probe else ""
    logger.configure(
        name="{log_prefix}{label}_{base_model_name}_{ds_name}_{loss_type}_{optimizer_name}_{lr}_{lr_schedule}{linprobe_str}_{datetime_now}",
        label=label,
        ds_name=ds_name,
        truncation_max_len=n_docs or "none",
        loss_type=loss_type,
        lr=lr,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        minibatch_size_per_device=minibatch_size_per_device,
        save_path=save_path,
        base_model_name=model_config.name,
        epochs=epochs,
        linprobe_str=linprobe_str,
        lr_schedule=lr_schedule,
        log_prefix=log_prefix,
        optimizer_name=optimizer_name,
    )
    # Tokenize datasets
    tokenizer = get_tokenizer(model_config.name)
    train_ds = tokenize_dataset(train_ds, tokenizer, max_ctx)
    test_ds = tokenize_dataset(test_ds, tokenizer, max_ctx)
    if inference_ds:
        inference_ds = tokenize_dataset(inference_ds, tokenizer, max_ctx)

    loss_fn = loss_dict[loss_type]
    return train_and_save_model(
        model_config,
        train_ds,
        test_ds,
        inference_ds=inference_ds,
        batch_size=batch_size,
        save_path=save_path,
        loss_fn=loss_fn,
        lr=lr,
        epochs=epochs,
        force_retrain=force_retrain,
        eval_batch_size=eval_batch_size,
        minibatch_size_per_device=minibatch_size_per_device,
        train_with_dropout=train_with_dropout,
        linear_probe=linear_probe,
        lr_schedule=lr_schedule,
        optimizer_name=optimizer_name,
        eval_every=eval_every,
    )


batch_size = 32
minibatch_size_per_device = 32
lr_schedule = "cosine_anneal"
force_retrain = False
max_ctx = 512
train_with_dropout = False
eval_every = 100000000
weak_model_size = "gpt2"
log_prefix = ""
weak_model_config = ModelConfig(
    name="gpt2",
    default_lr=5e-5,
    eval_batch_size=32,
    custom_kwargs={
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    },
)
gt_epochs = 2
seed = 42

ds_name = "amazon_polarity"
# ds_name = "sciq"
n_docs = 3600000 #11679 #
n_test_docs = 400000 #1000 #
dataset = load_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs)) #, validation=n_test_docs))
train_dataset, test_ds = dataset["train"], dataset["test"] #, dataset["validation"] , val_ds
print(len(train_dataset), len(test_ds)) #, len(val_ds))
train_dataset = concatenate_datasets([train_dataset, test_ds]) #, val_ds
print(len(train_dataset))
n_docs = 3600000 + n_test_docs #+ n_test_docs
n_folds = 10

difficulties = np.zeros(n_docs)
for i in trange(n_folds):
    print(f"Fold {i+1}/{n_folds}")
    # shutil.rmtree("./results/weak_model_gt")
    train_ds = train_dataset.select(range(i*(n_docs//n_folds), (i+1)*(n_docs//n_folds)))
    val_ds = train_dataset.filter(lambda example, indice: indice < i*(n_docs//n_folds) or indice >= (i+1)*(n_docs//n_folds), 
                                  with_indices=True)
    weak_test_results, weak_ds = train_model(
        weak_model_config,
        train_dataset,
        train_ds,
        loss_type="xent",
        label="weak",
        subpath=os.path.join("difficulty_rating", weak_model_size.replace("/", "_")),
        lr=weak_model_config.default_lr,
        eval_batch_size=weak_model_config.eval_batch_size,
        inference_ds=val_ds,
        epochs=gt_epochs,
        linear_probe=False,
        optimizer_name=weak_model_config.default_optimizer,
    )
    partial_difficulties = 1 - np.array(weak_ds["soft_label"])[np.arange(len(weak_ds)), weak_ds["gt_label"]]
    print(partial_difficulties[:10])
    difficulties[:i*(n_docs//n_folds)] += partial_difficulties[:i*(n_docs//n_folds)] / (n_folds-1)
    difficulties[(i+1)*(n_docs//n_folds):] += partial_difficulties[i*(n_docs//n_folds):] / (n_folds-1)



# Draw histogram
plt.hist(difficulties, bins=10, alpha=0.75, color='blue')
plt.title(f'{ds_name.upper()}: Histogram of Difficulties')
plt.xlabel('Value')
plt.ylabel('Frequency')
#matplotlib save file 
plt.savefig(f'./results/difficulties_{ds_name}_{n_docs}_{seed}.png')

# Find indices of 3 smallest and 3 largest numbers
indices_smallest = np.argsort(difficulties)[:3]
indices_largest = np.argsort(difficulties)[-3:]

print("3 easiest questions:")
for idx in indices_smallest:
    print('\t' + train_dataset["question" if ds_name == "sciq" else "title"][idx])
print("3 hardest questions:")
for idx in indices_largest:
    print('\t' + train_dataset["question" if ds_name == "sciq" else "title"][idx])


np.savetxt(f"./results/difficulties_{ds_name}_{n_docs}_{seed}.txt", difficulties, fmt='%f')