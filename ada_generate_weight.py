import json
import sys
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import pickle
import time
import datasets
from typing import Callable, Optional

from weak_to_strong.eval import eval_model_acc
from weak_to_strong.model import TransformerWithHead

from typing import Dict, List, Optional, Sequence, Union
from transformers.modeling_utils import load_sharded_checkpoint

import fire
import numpy as np
import torch
# import tiktoken
import weak_to_strong.logger as logger
from weak_to_strong.common import clear_mem, get_tokenizer
from weak_to_strong.datasets import tokenize_dataset
from datasets import load_dataset,load_from_disk
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss, weight_xent_loss
from weak_to_strong.train import ModelConfig, train_and_save_model
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--E", type=int, default=1)
parser.add_argument("--weak_model_size", type=str, default="gpt2-medium")
parser.add_argument("--ds_name", type=str, default="sciq")
parser.add_argument("--weighted_sampling", type=bool, default=False)
parser.add_argument("--gt_epochs", type=int, default=2)
parser.add_argument("--results_folder", type=str, default="./results")

args = parser.parse_args()


MODEL_CONFIGS = [
    ModelConfig(
        name="gpt2",
        default_lr=5e-5,
        eval_batch_size=32,
         default_optimizer="adamw",
        custom_kwargs={
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
    ),
    ModelConfig(
        name="gpt2-medium",
        default_lr=5e-5,
        eval_batch_size=32,
         default_optimizer="adamw",
        custom_kwargs={
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
    ),
    ModelConfig(
        name="gpt2-large",
        default_lr=1e-4,
        eval_batch_size=32,
         default_optimizer="adamw",
        custom_kwargs={
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
    ),
    ModelConfig(
        name="gpt2-xl",
        default_lr=1e-5,
        eval_batch_size=2,
        # gradient_checkpointing=True,
        # model_parallel=True,
        custom_kwargs={
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
    ),
    ModelConfig(
        name="Qwen/Qwen1.5-0.5B",
        default_lr=2e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        default_optimizer="adamw",
        custom_kwargs={
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
    ),
    ModelConfig(
        name="Qwen/Qwen1.5-1.8B",
        default_lr=2e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        default_optimizer="adamw",
        custom_kwargs={
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-1_8B",
        default_lr=2e-5,
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
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
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
    "weight_xent":weight_xent_loss(),
}

VALID_LOSSES: List[str] = list(loss_dict.keys())

E = args.E

def main(
    batch_size: int = 32,
    max_ctx: int = 1024,
    ds_name: str = "sciq",
    weighted_sampling: bool = args.weighted_sampling,
    split_by_random: bool = False,
    train1_name: str = "/adaboost/train1_10000_{}/".format(E-1),
    train2_name: str = "./sciq/validation/",
    test_name: str = "/data2/kongchao/superalignment/myweak2strong2/myweak2strong/sciq/test",
    transfer_loss: Union[str, Sequence[str]] = "xent,logconf",
    n_docs: int = 10000,
    n_test_docs: int = 200,
    weak_model_size: str = args.weak_model_size,
    weak_lr: Optional[float] = None,
    strong_model_size: str = "gpt2-medium",
    strong_lr: Optional[float] = None,
    # Defaults to strong_lr
    transfer_lr: Optional[float] = None,
    # Optims default to default_optimizer in the model definitions
    weak_optim: Optional[str] = None,
    strong_optim: Optional[str] = None,
    transfer_optim: Optional[str] = None,
    gt_epochs: int = args.gt_epochs,
    # defaults to gt_epochs
    transfer_epochs: Optional[int] = None,
    force_retrain: bool = False,
    seed: int = 42,
    minibatch_size_per_device: Optional[int] = 32,
    train_with_dropout: bool = False,
    results_folder: str = args.results_folder, # "./results",
    linear_probe: bool = False,
    lr_schedule: str = "cosine_anneal",
    log_prefix: str = "",
    # Set to an absurdly high value so we don't do intermediate evals by default.
    eval_every: int = 100000000,
):
    print("batch size:", batch_size, "E: ", E)
    # this is per device!
    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1
 
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



    # Load dataset
    if split_by_random:
        f = "_data/random/"
    else:
        f = "_data/"
    
    train1_ds = load_from_disk("./" + ds_name + f + weak_model_size + train1_name)

    tokenizer = get_tokenizer(weak_model_config.name)
    train_ds = tokenize_dataset(train1_ds, tokenizer, max_ctx, weight = True)


    ### model prepare
    subpath=os.path.join("weak_model_gt/10000", weak_model_size.replace("/", "_")) + str(E - 1)
    save_path = os.path.join(results_folder, subpath)
    custom_kwargs = weak_model_config.custom_kwargs or {}
    def maybe_load_model(model):
        if os.path.exists(os.path.join(save_path, "results.pkl")) and not force_retrain:
            print("loading from", save_path)
            checkpoint_path = os.path.join(save_path, "pytorch_model.bin")
            # checkpoint_path = os.path.join(save_path, "model.safetensors")
            if not os.path.exists(checkpoint_path):
                # Assume this means we have a sharded checkpoint, and load it appropriately
                load_sharded_checkpoint(model, checkpoint_path)
            else:
                state_dict = torch.load(os.path.join(save_path, "pytorch_model.bin"))
                # state_dict = load_file(os.path.join(save_path, "pytorch_model.bin"))
                # state_dict = {
                #     k.replace("transformer.module", "transformer"): v
                #     for (k, v) in state_dict.items()
                # }
                custom_kwargs["state_dict"] = state_dict
            return True
        return False
    model = TransformerWithHead.from_pretrained(
    weak_model_config.name, num_labels=2, linear_probe=linear_probe, **custom_kwargs
    ).to("cuda")
    already_trained = maybe_load_model(model)
    if already_trained:
        model.load_state_dict(torch.load(os.path.join(save_path, "pytorch_model.bin")))
    # data parallel:  currently not supported with model parallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, output_device=0)
        minibatch_size = min(minibatch_size_per_device * torch.cuda.device_count(), batch_size)
        print(
            "Using",
            torch.cuda.device_count(),
            "GPUs, setting minibatch_size to",
            minibatch_size,
        )
    

    ### predict
    io_device = model.device if hasattr(model, "device") else 0
    model.eval()
    result = pickle.load(open(os.path.join(save_path, "results.pkl"), "rb"))
    e = result["weighted_error_inference"] 
    alpha = 0.5 * np.log((1 - e) / e)
    z = 2 * np.sqrt(e * (1 - e))
    print("Weighted Error: ", e, "alpha: ", alpha, "z: ", z)
    def small_process(i):
        with torch.no_grad():
            input_ids = torch.tensor(i["input_ids"]).unsqueeze(0).to(io_device)
            labels = torch.tensor(i["soft_label"]).unsqueeze(0)
            logits = model(input_ids)
            probs = torch.nn.functional.softmax(logits, dim = -1).to("cpu")
            
            preds = torch.argmax(probs)
            labels = torch.argmax(labels)
            if preds == labels:
                i["weight"] = (i["weight"]*np.exp(-alpha)) #/ (2 * (1 - e)))
            else:
                i["weight"] = (i["weight"]*np.exp(alpha)) # / (2*e))
            
            return i
    train_ds = train_ds.map(small_process)
    weight_sum = sum(train_ds["weight"])
    print("weight_sum: ", weight_sum)

    def small_process2(i):
        with torch.no_grad():
            i["weight"] = i["weight"]/weight_sum
            return i
    train_ds = train_ds.map(small_process2)
    train_ds.save_to_disk("./" + ds_name + f + weak_model_size + "/adaboost/train1_10000_{}/".format(E))



if __name__ == "__main__":
    # fire.Fire(main)
    main()
