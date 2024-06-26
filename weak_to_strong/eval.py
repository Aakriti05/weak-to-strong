import datasets
import numpy as np
import torch
from torch import nn


def to_batch(x, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size]


def unpack(x):
    assert isinstance(x, torch.Tensor), type(x)
    return x.detach().float().cpu().numpy().tolist()


def eval_model_acc(model: nn.Module, ds: datasets.Dataset, eval_batch_size: int = 16) -> None:
    """
    This function evaluates the accuracy of a given model on a given dataset.

    Parameters:
    model (nn.Module): The model to be evaluated.
    ds (datasets.Dataset): The dataset on which the model is to be evaluated.

    Returns:
    results (list): A list of dictionaries containing the input_ids, ground truth label, predicted label,
                    accuracy of prediction, logits and soft label for each example in the dataset.
    """

    model.eval()
    

    with torch.no_grad():
        results = []
        # for ex in ds:
        for batch in to_batch(ds, eval_batch_size):
            # pad input_ids to common length
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch["input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            labels = batch["soft_label"]
            if "weight" in batch:
                weights = batch["weight"]
            else:
                weights = [1] * len(batch["input_ids"])
            # run forward pass
            raw_logits = model(input_ids)

            probs = unpack(torch.nn.functional.softmax(raw_logits, dim=-1))
            logits = unpack(raw_logits)

            preds = np.argmax(probs, axis=-1)
            labels = np.argmax(labels, axis=-1)

            results.extend(
                [
                    dict(
                        txt=txt,
                        input_ids=input_id,
                        gt_label=label,
                        hard_label=pred,
                        acc=label == pred,
                        weight_err = weight * (label != pred),
                        logits=logit,
                        soft_label=(prob/np.sum(prob, axis=-1) >= .5)*1.0,
                    )
                    for input_id, txt, label, pred, prob, logit, weight in zip(
                        batch["input_ids"], batch["txt"], labels, preds, probs, logits, weights
                    )
                ]
            )
        accs = [r["acc"] for r in results]
        print("Accuracy:", np.mean(accs), "+/-", np.std(accs) / np.sqrt(len(accs)))
        accs = [r["weight_err"] for r in results]
        print("Weights Error:", np.sum(accs), "+/-", np.std(accs) / np.sqrt(len(accs)))


        return datasets.Dataset.from_list(results)


def eval_model_logits(model: nn.Module, ds: datasets.Dataset, eval_batch_size: int = 16):
    model.eval()

    with torch.no_grad():
        results = []
        # for ex in ds:
        for batch in to_batch(ds, eval_batch_size):
            # pad input_ids to common length
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch["input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")
            labels = batch["soft_label"]
            if "weight" in batch:
                weights = batch["weight"]
            else:
                weights = [1] * len(batch["input_ids"])
            # run forward pass
            raw_logits = model(input_ids)
            raw_logits = torch.clamp(raw_logits, 0, 1)
            other_logits = 1 - raw_logits
            logits = torch.concat([other_logits, raw_logits], dim=-1)
            
            # probs = unpack(torch.nn.functional.softmax(raw_logits, dim=-1))
            # logits = unpack(raw_logits)

            probs = unpack(logits)
            preds = np.argmax(probs, axis=-1)
            labels = np.argmax(labels, axis=-1)
            results.extend(
                [
                    dict(
                        txt=txt,
                        input_ids=input_id,
                        gt_label=label,
                        hard_label=pred,
                        acc=label == pred,
                        weight_err = weight * (label != pred),
                        logits=logit,
                        soft_label=(prob/np.sum(prob, axis=-1) >= .5)*1.0,
                    )
                    for input_id, txt, label, pred, prob, logit, weight in zip(
                        batch["input_ids"], batch["txt"], labels, preds, probs, logits, weights
                    )
                ]
            )
        accs = [r["acc"] for r in results]
        print("Accuracy:", np.mean(accs), "+/-", np.std(accs) / np.sqrt(len(accs)))
        accs = [r["weight_err"] for r in results]
        print("Weights Error:", np.sum(accs), "+/-", np.std(accs) / np.sqrt(len(accs)))

        return datasets.Dataset.from_list(results)