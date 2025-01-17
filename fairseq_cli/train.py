#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import logging
import math
import os
import random
import sys

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from fairseq import (
    checkpoint_utils, distributed_utils, metrics, options, progress_bar, tasks, utils
)
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import StopwatchMeter

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.train')


def main(args, init_distributed=False):
    utils.import_user_module(args)
    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    logger.info(args)

    bsz_per_gpu = args.max_tokens // args.tokens_per_sample
    n_gpus = args.distributed_world_size
    grad_acc = args.update_freq[0]
    logger.info('Batch size per GPU: {} tokens ({} batches)'.format(bsz_per_gpu * args.tokens_per_sample, bsz_per_gpu))
    logger.info('Effective global batch size: {} tokens ({} batches)'.format(
                bsz_per_gpu * n_gpus * grad_acc * args.tokens_per_sample, bsz_per_gpu * n_gpus * grad_acc))

    if args.criterion == 'trime_loss':
        logger.info('Train with Trime loss')
    elif args.criterion == 'trime_long_loss':
        logger.info('Train with Trime_long loss')
        if args.train_mem_size is not None:
            assert args.train_mem_size % args.max_tokens == 0, "Please make sure args.train_mem_size %% args.max_tokens == 0"
            args.num_comb_shards = args.train_mem_size // args.max_tokens
        logger.info('num. consecutive segments: {}'.format(args.num_comb_shards * bsz_per_gpu))
    elif args.criterion == 'trime_long_loss_same_device':
        logger.info('Train with Trime_long loss')
        logger.info('Using "trime_long_loss_same_device", assuming all memories are in the same gpu device')
        logger.info('num. consecutive segments: {}'.format(bsz_per_gpu))
    elif args.criterion == 'trime_ext_loss':
        logger.info('Train with Trime_ext loss')

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)  # self.datasets['valid'] = MonolingualDataset(...)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    # logger.info(model)
    logger.info('model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    logger.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    logger.info('training on {} GPUs'.format(args.distributed_world_size))
    logger.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,  # 9000
        args.max_sentences,  # None
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    print('calling load_checkpoint')
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
    print('done load_checkpoint')
    # epoch_itr:  EpochBatchIterator
    #print(epoch_itr.dataset.dataset.dataset)  # MonolingualDataset.TokenBlockDataset.MMapIndexedDataset

    if args.output_segments_to_file is not None:
        segments = []
        itr = epoch_itr.next_epoch_itr(shuffle=False)
        for samples in itr:
            for j in range(len(samples['id'])):
                s = task.dictionary.string(samples['net_input']['src_tokens'][j])
                segments.append({"id": samples['id'][j].item(), 'contents': s})
        logger.info('num. total segments: {}'.format(len(segments)))
        with open(args.output_segments_to_file, 'w') as f:
            import json
            json.dump(segments, f)
        exit(0)

    if args.data_debug is not None:
        # Doing exactly what training does, only turn shuffle off
        import pickle
        itr = epoch_itr.next_epoch_itr(shuffle=False)
        update_freq = (    # update_freq=2: args.update_freq=[2], epoch_itr.epoch=1
            args.update_freq[epoch_itr.epoch - 1]
            if epoch_itr.epoch <= len(args.update_freq)
            else args.update_freq[-1]
        )
        itr = iterators.GroupedIterator(itr, update_freq)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch, no_progress_bar='simple',
        )
        task.begin_epoch(epoch_itr.epoch, trainer.get_model())
        batches = list(progress)
        data = (len(batches), batches[0], batches[-1])

        with open(args.data_debug, 'wb') as f:
            pickle.dump(data, f)

        print(data[0], 'batches')
        print('First batch')
        print(data[1])
        print('Last batch')
        print(data[2])
        exit()

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')
    while (
            lr > args.min_lr  # lr=1e-07, min_lr=-1
            and (
                epoch_itr.epoch < max_epoch
                # allow resuming training from the final checkpoint
            or epoch_itr._next_epoch_itr is not None
            )
            and trainer.get_num_updates() < max_update
    ):
        # train for one epoch
        if not args.skip_training:
            train(args, trainer, task, epoch_itr)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0 and distributed_utils.is_master(args):
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        # early stop
        if should_stop_early(args, valid_losses[0]):
            logger.info('early stop since valid performance hasn\'t improved for last {} runs'.format(args.patience))
            break

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.epoch,
            # sharded data: get train iterator for next epoch
            load_dataset=(os.pathsep in getattr(args, 'data', '')),
        )
    valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
    train_meter.stop()
    # torch.distributed.barrier()
    logger.info('done training in {:.1f} seconds'.format(train_meter.sum))

def should_stop_early(args, valid_loss):
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, 'best', None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        return should_stop_early.num_runs > args.patience


@metrics.aggregate('train')
def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Initialize data iterator
    print('calling next_epoch_itr in train(..)')
    itr = epoch_itr.next_epoch_itr(  # curriculum = 0
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=((epoch_itr.epoch >= args.curriculum) and (not args.keep_order)),
    )
    print('done next_epoch_itr in train(..)')

    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    if epoch_itr.epoch <= args.ce_warmup_epoch:
        logger.info('Train with CE loss at epoch %d (ce-warmup-epoch: %d)'%(epoch_itr.epoch, args.ce_warmup_epoch))
        trainer.criterion.return_ce_loss = True

    # task specific setup per epoch
    task.begin_epoch(epoch_itr.epoch, trainer.get_model())

    valid_subsets = args.valid_subset.split(',')  # ['valid']
    max_update = args.max_update or math.inf


    def verbalize_samples(samples):
        print('-' * 80)
        print(f'len(samples)={len(samples)}')
        print(f'samples[0].keys()={samples[0].keys()}')
        print(f'samples[0][id]={samples[0]["id"].tolist()}')
        print(f'samples[0][ntokens]={samples[0]["ntokens"]}')
        print(f'samples[0][net_input][src_lengths]=',samples[0]['net_input']['src_lengths'])
        print(f'samples[0][net_input][src_tokens]')
        #print(task.dictionary.string(samples[0]['net_input']['src_tokens']))   # Not relying on this just in case
        [B, T] = list(samples[0]['net_input']['src_tokens'].size())
        for i in range(B):
            for t in range(T):
                s = task.dictionary[samples[0]['net_input']['src_tokens'][i, t].item()]
                print(s, end=' ')
            print()
        print(samples[0]['net_input']['src_tokens'])

        print()
        print(f'samples[0][target]')
        #print(task.dictionary.string(samples[0]['target']))
        [B, T] = list(samples[0]['target'].size())
        for i in range(B):
            for t in range(T):
                s = task.dictionary[samples[0]['target'][i, t].item()]
                print(s, end=' ')
            print()
        print(samples[0]['target'])



    #print([x[0]['id'].tolist() for x in list(progress)])
    for idx_progress, samples in enumerate(progress):
        print(idx_progress + 1, '/', len(progress))
        verbalize_samples(samples)

        log_output = trainer.train_step(samples)            # THE STEP
        num_updates = trainer.get_num_updates()
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(metrics.get_smoothed_values('train'))
        progress.log(stats, tag='train', step=num_updates)

        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            if distributed_utils.is_master(args):
                checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    trainer.criterion.return_ce_loss = False

    # log end-of-epoch stats
    stats = get_training_stats(metrics.get_smoothed_values('train'))
    progress.print(stats, tag='train', step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters('train')


def get_training_stats(stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:  # ['valid']
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),  # MonolingualDataset
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        if distributed_utils.is_master(args):
            stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
            progress.print(stats, tag=subset, step=trainer.get_num_updates())
            valid_losses.append(stats[args.best_checkpoint_metric])
        else:
            valid_losses.append(None)
    return valid_losses


def get_valid_stats(args, trainer, stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            logger.info('NOTE: you may get faster training with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        print('single GPU training')
        main(args)


if __name__ == '__main__':
    cli_main()
