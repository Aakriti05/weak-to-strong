import json
import os
import sys
import random


from typing import Dict, List, Optional, Sequence, Union

import fire
import numpy as np
import torch
from datasets import load_from_disk

import weak_to_strong.logger as logger
from weak_to_strong.common import get_tokenizer
from weak_to_strong.datasets import (VALID_DATASETS, load_dataset, tokenize_dataset)
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss
from weak_to_strong.train import ModelConfig, train_and_save_model

# NOTE learning rates are not particularly tuned, work somewhat reasonably at train batch size 32
MODEL_CONFIGS = [
    ModelConfig(
        name="gpt2",
        default_lr=5e-5,
        eval_batch_size=32,
        custom_kwargs={
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
    ),
    ModelConfig(
        name="gpt2-medium",
        default_lr=5e-5,
        eval_batch_size=32,
        custom_kwargs={
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
    ),
    ModelConfig(
        name="gpt2-large",
        default_lr=1e-5,
        eval_batch_size=32,
        custom_kwargs={
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
    ),
    ModelConfig(
        name="gpt2-xl",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        custom_kwargs={
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-1_8B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        custom_kwargs={
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-7B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this without many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-14B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this without bf16 support and many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-72B",
        default_lr=1e-5,
        eval_batch_size=1,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this without bf16 support and many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
        # This model is really big, save space by using adafactor.
        # Note that even then it will take up ~60GB per GPU on an 8-GPU machine.
        default_optimizer="adafactor",
    ),
]
MODELS_DICT: Dict[str, ModelConfig] = {
    model_config.name: model_config for model_config in MODEL_CONFIGS
}


loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
}

VALID_LOSSES: List[str] = list(loss_dict.keys())

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

# E = int(sys.argv[1])
seed_torch(1029)


def main(
    batch_size: int = 32,
    max_ctx: int = 1024,
    ds_name: str = "sciq",
    split_by_difficulty: bool = True,
    adaboost:bool = True,
    rounds: int = 26,
    transfer_loss: Union[str, Sequence[str]] = "xent", #logconf",
    n_docs: int = 20000, #10000,
    n_test_docs: int = 2000, #200,
    weak_model_size: str = "gpt2",
    weak_lr: Optional[float] = None,
    strong_model_size: str = "gpt2-medium",
    strong_lr: Optional[float] = None,
    # Defaults to strong_lr
    transfer_lr: Optional[float] = None,
    # Optims default to default_optimizer in the model definitions
    weak_optim: Optional[str] = None,
    strong_optim: Optional[str] = None,
    transfer_optim: Optional[str] = None,
    weak_gt_epochs: int = 3,
    gt_epochs: int = 3,
    # defaults to gt_epochs
    transfer_epochs: Optional[int] = None,
    force_retrain: bool = False,
    seed: int = 42,
    minibatch_size_per_device: Optional[int] = 32,
    train_with_dropout: bool = False,
    results_folder: str = "./results",
    sweep_subfolder: str = "./w2s_default",
    linear_probe: bool = False,
    lr_schedule: str = "cosine_anneal",
    log_prefix: str = "",
    # Set to an absurdly high value so we don't do intermediate evals by default.
    eval_every: int = 100,
):
    seed_torch(1029)
    # this is per device!
    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1
    assert ds_name in VALID_DATASETS, f"Unknown dataset {ds_name} not in {VALID_DATASETS}"
    if isinstance(transfer_loss, str):
        transfer_losses = transfer_loss.split(",")
    else:
        transfer_losses = transfer_loss
    del transfer_loss
    for tloss in transfer_losses:
        assert tloss in VALID_LOSSES, f"Unknown loss {tloss} not in {VALID_LOSSES}"
    assert (
        weak_model_size in MODELS_DICT
    ), f"Unknown model size {weak_model_size} not in {MODELS_DICT}"
    weak_model_config = MODELS_DICT[weak_model_size]
    assert (
        strong_model_size in MODELS_DICT
    ), f"Unknown model size {strong_model_size} not in {MODELS_DICT}"
    strong_model_config = MODELS_DICT[strong_model_size]

    if weak_lr is None:
        assert batch_size == 32
        weak_lr = weak_model_config.default_lr
    if strong_lr is None:
        assert batch_size == 32
        strong_lr = strong_model_config.default_lr
    if transfer_lr is None:
        transfer_lr = strong_lr
    if transfer_epochs is None:
        transfer_epochs = gt_epochs

    if weak_optim is None:
        weak_optim = weak_model_config.default_optimizer
    if strong_optim is None:
        strong_optim = strong_model_config.default_optimizer
    if transfer_optim is None:
        transfer_optim = strong_optim

    weak_eval_batch_size = weak_model_config.eval_batch_size
    strong_eval_batch_size = strong_model_config.eval_batch_size


    if split_by_difficulty:
        train1_ds = load_from_disk(ds_name + "_data" + "/" + weak_model_size + "/adaboost/train1_10000_0")
        train2_ds = load_from_disk(ds_name + "_data" + "/" + weak_model_size + "/train2") #"./dataset/" + 
        test_ds = load_from_disk(ds_name + "_data" + "/" + weak_model_size + "/test") #"./dataset/" +
        val_ds = load_from_disk(ds_name + "_data" + "/" + weak_model_size + "/val") #"./dataset/" +

    else:
        # Load dataset
        dataset = load_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs, validation=n_test_docs)) #n_test_docs
        # Split the training dataset in half
        train_dataset, test_ds, val_ds = dataset["train"], dataset["test"], dataset["validation"]
        split_data = train_dataset.train_test_split(test_size=0.5, seed=seed)
        train1_ds, train2_ds = split_data["train"], split_data["test"]

    print("len(train1):", len(train1_ds), "len(train2):", len(train2_ds), "len(test):", len(test_ds), "len(val):", len(val_ds))
    # train1_ds = train1_ds.shuffle(seed=7)
    # train2_ds = train2_ds.shuffle(seed=7)
    # test_ds = test_ds.shuffle(seed=7)

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

    # Train the weak model on the first half of the training data
    print(f"Training weak model, size {weak_model_size}")

    
    if adaboost:
        weak_ds = load_from_disk(ds_name + "_data" + "/" + weak_model_size + "/adaboost/weak_data_" + str(rounds) + "/".format()) #"./dataset/" + 

        weak_accuracy = 1
    
    else:
        weak_test_results, weak_ds = train_model(
            weak_model_config,
            train1_ds,
            val_ds,
            loss_type="xent",
            label="weak",
            subpath=os.path.join(sweep_subfolder, "weak_model_gt", weak_model_size.replace("/", "_")),
            lr=weak_lr,
            eval_batch_size=weak_eval_batch_size,
            inference_ds=train2_ds,
            epochs= weak_gt_epochs,
            linear_probe=linear_probe,
            optimizer_name=weak_optim,
        )

    print("weak_ds:", weak_ds[0])
    print("weak_ds:", weak_ds[1])

    def small_process2(i):
        with torch.no_grad():
            i["acc"] = (i["hard_label"] == i["gt_label"])
            return i
    weak_ds = weak_ds.map(small_process2)

    ac = weak_ds["acc"]
    accuracy = np.mean(ac)
    print("weak accuracy:", accuracy)

    # print("train1_ds:", train1_ds[0])
    # print("train1_ds:", train1_ds[1])


    # print("train2_ds:", train2_ds[0])
    # print("train2_ds:", train2_ds[1])


    if adaboost:
        pass
    else:
        # Train the strong model on the second half of the training data
        print(f"Training strong model, size {strong_model_size}")
        strong_test_results, _ = train_model(
            strong_model_config,
            train2_ds,
            val_ds,
            loss_type="xent",
            label="strong",
            subpath=os.path.join(sweep_subfolder, "strong_model_gt", strong_model_size.replace("/", "_")),
            lr=strong_lr,
            eval_batch_size=strong_eval_batch_size,
            epochs=gt_epochs,
            linear_probe=linear_probe,
            optimizer_name=strong_optim,
        )

    # Train the strong model on the second half of the training data with labels generated by the weak model
    all_transfer_test_results = {}
    for tloss in transfer_losses:
        print(
            f"Training transfer model, size {strong_model_size} on labels from {weak_model_size}, with loss {tloss}"
        )
        transfer_test_results, _ = train_model(
            strong_model_config,
            weak_ds,
            val_ds,
            loss_type=tloss,
            label="weak2strong",
            subpath=os.path.join(sweep_subfolder,
                "strong_model_transfer",
                f"{weak_model_size.replace('/', '_')}_{strong_model_size.replace('/', '_')}_{tloss}",
            ),
            lr=transfer_lr,
            eval_batch_size=strong_eval_batch_size,
            epochs=gt_epochs,
            linear_probe=linear_probe,
            optimizer_name=transfer_optim,
        )
        all_transfer_test_results[tloss] = transfer_test_results
        del transfer_test_results

    if adaboost:
        weak_acc = weak_accuracy
        strong_acc = 1
    else:
        weak_acc = np.mean([x["acc"] for x in weak_test_results])
        strong_acc = np.mean([x["acc"] for x in strong_test_results])
    
    res_dict = {
        "weak_acc": weak_acc,
        "strong_acc": strong_acc,
    }
    print("weak acc:", weak_acc)
    print("strong acc:", strong_acc)
    for tloss, transfer_test_results in all_transfer_test_results.items():
        transfer_acc = np.mean([x["acc"] for x in transfer_test_results])
        res_dict[f"transfer_acc_{tloss}"] = transfer_acc
        print(f"transfer acc ({tloss}):", transfer_acc)

    with open(
        os.path.join(
            results_folder, sweep_subfolder,
            f"{weak_model_size.replace('/', '_')}_{strong_model_size.replace('/', '_')}.results_summary.json",
        ),
        "w",
    ) as f:
        json.dump(
            res_dict,
            f,
        )


# python train_weak_to_strong.py --batch_size 32 --max_ctx 512 --ds_name "sciq" --transfer_loss "logconf" --n_docs 1000 --n_test_docs 100 --weak_model_size "gpt2-medium" --strong_model_size "gpt2-large" --seed 42
if __name__ == "__main__":
    fire.Fire(main)
