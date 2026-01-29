import argparse
import math
import os
import time
from functools import partial
from pathlib import Path

import datasets
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# import evaluate
import utils
from mlx.nn.utils import average_gradients
from mlx.utils import tree_map, tree_reduce


def load_data(tokenizer, data_path="allenai/dolma3_mix-150B-1025", valid_size=1000):
    group = mx.distributed.init()
    size = group.size()
    rank = group.rank()
    ds = datasets.load_dataset(
        data_path,
        split="train",
        streaming=True,
    )
    ds = ds.shard(num_shards=size, index=rank)
    ds = ds.shuffle(buffer_size=10000)

    def tokenize(d):
        tokens = tokenizer.encode(d["text"], add_special_tokens=False)
        tokens.append(tokenizer.eos_token_id)
        return {"data": tokens}

    ds = ds.map(tokenize)
    local_valid_size = valid_size // size
    valid_ds = ds.take(local_valid_size)
    train_ds = ds.skip(local_valid_size)
    return train_ds, valid_ds


def iterate_batches(dataset, context_size, batch_size, max_batches=None):
    """
    Simply concatenate documents until the batch is full.
    """
    dataset = iter(dataset)
    seq_len = context_size + 1
    batch = np.empty((batch_size * seq_len), np.int32)
    max_batches = max_batches or float("inf")
    d_next = []
    batch_num = 0
    while batch_num < max_batches:
        i = 0
        while i < len(batch):
            if len(d_next) > 0:
                d = d_next
                d_next = []
            else:
                d = next(dataset, None)
            if d is None:
                break
            d = d["data"]
            e = i + len(d)
            if e > len(batch):
                trim = e - len(batch)
                d_next = d[-trim:]
                d = d[:-trim]
                e = len(batch)
            batch[i:e] = d
            i += len(d)
        # Iterator ended
        if i < len(batch):
            break
        batch_num += 1
        yield batch.reshape(batch_size, seq_len)


def main(config, save_dir):

    np.random.seed(config.seed)
    mx.random.seed(config.seed)

    rank, world_size = utils.init_distributed()
    batch_size = config.batch_size
    context_size = config.context_size

    if rank == 0:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        utils.save_config(save_dir, config)

    # data.download_eval_bundle()

    optimizer = utils.load_optimizer(config)
    tokenizer = utils.load_tokenizer()
    train_set, valid_set = load_data(tokenizer)

    model = utils.load_model(config.model)
    dtype = getattr(mx, config.data_type)

    @mx.compile
    def loss_fn(params, sample):
        model.update(tree_map(lambda x: x.astype(dtype), params))
        inputs = sample[:, :-1]
        targets = sample[:, 1:]

        logits = model(inputs).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        return losses.sum() / targets.size

    state = [optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(sample, params):
        loss, grads = mx.value_and_grad(loss_fn)(params, sample)
        grads = average_gradients(grads, all_reduce_size=4e9)
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=config.max_grad_norm)
        params = optimizer.apply_gradients(grads, params)
        return loss, grad_norm, params

    def eval_fn(params, dataset):
        data_it = iterate_batches(
            dataset,
            context_size=context_size,
            batch_size=batch_size,
            max_batches=config.num_valid_batches,
        )
        losses = 0
        ntoks = 0
        toks_per_batch = context_size * batch_size
        for sample in data_it:  # data.prefetch(data_it):
            loss = loss_fn(params, mx.array(sample))
            losses += loss * toks_per_batch
            mx.eval(losses)
            ntoks += toks_per_batch
        return losses / ntoks

    params = model.trainable_parameters()
    nparams = tree_reduce(lambda acc, p: acc + p.size, params, 0)
    if rank == 0:
        print(f"Model has {nparams} parameters.")
    mx.eval(params)

    train_iterator = iterate_batches(
        train_set,
        context_size=config.context_size,
        batch_size=config.batch_size,
    )

    metrics = utils.Metrics()
    tokens = 0
    tic = time.perf_counter()
    for it, sample in zip(
        range(0, config.num_steps), train_iterator
    ):  # data.prefetch(train_iterator)):
        loss, grad_norm, params = step(mx.array(sample), params)
        loss = mx.distributed.all_sum(loss) / world_size
        grad_norm = mx.distributed.all_sum(grad_norm) / world_size
        mx.eval(loss, grad_norm, params, state)
        metrics.train_loss.append(loss.item())
        metrics.grad_norm.append(grad_norm.item())
        tokens += config.context_size * config.batch_size * world_size

        if (it + 1) % config.steps_per_report == 0:
            toc = time.perf_counter()
            metrics.step = it + 1
            metrics.tokens += tokens
            metrics.its_per_sec = config.steps_per_report * world_size / (toc - tic)
            metrics.toks_per_sec = tokens / (toc - tic)
            tokens = 0

            if (it + 1) % config.steps_per_eval == 0:
                # Do the evaluation in the final precision,
                # but only cast the model once
                eval_params = tree_map(lambda x: x.astype(dtype), params)
                loss = eval_fn(eval_params, valid_set)
                loss = mx.distributed.all_sum(loss) / world_size
                metrics.valid_loss = loss.item()
                metrics.valid_ppl = math.exp(metrics.valid_loss)
                model.update(params)

            if rank == 0:
                if (it + 1) % config.steps_per_checkpoint == 0:
                    utils.save_checkpoint(save_dir, it, params, optimizer)
                    utils.save_checkpoint(save_dir, None, params, optimizer)
                utils.log_metrics(metrics)
            metrics = utils.Metrics(tokens=metrics.tokens)
            tic = time.perf_counter()

    utils.save_checkpoint(save_dir, None, params, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LM.")
    parser.add_argument(
        "--config",
        default="configs/tiny.yaml",
        type=str,
        help="Experiment config",
    )
    parser.add_argument(
        "--save-dir",
        default="checkpoints",
        type=str,
        help="Location to save the model and checkpoints",
    )
    args = parser.parse_args()
    config = utils.load_config(args.config)
    main(config, args.save_dir)
