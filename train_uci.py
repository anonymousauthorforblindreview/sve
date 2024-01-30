# Copyright (c) 2021 Cognizant Digital Business, Cognizant AI Labs
# Issued under this Academic Public License: github.com/cognizant-ai-labs/tom-release/LICENSE.

"""
Script for training concentric hyperspheres (ch) universe
"""

import argparse
import os
import random

import torch

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from tqdm import tqdm

from classical_model import ClassicalModel
from soft_order_model import SoftOrderModel
from traveling_observer_model import TravelingObserverModel
from utils import compute_loss_and_accuracy
from utils import load_dataset
from utils import squared_hinge

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import barrier, init_process_group, destroy_process_group

def ddp_setup(rank: int, world_size: int):
  seed = random.randint(0, 100)
  master_port = str(12355 + seed)
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = master_port
  init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=3600))
  torch.cuda.set_device(rank)

N_TASKS = 90


def main(rank: int, world_size: int):
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results/ch_test')
    parser.add_argument('--context_size', type=int, default=128)
    #TODO: restore context_std
    #parser.add_argument('--context_std', type=float, default=0.001)
    parser.add_argument('--context_std', type=float, default=1.0)
    parser.add_argument('--entmax_bisect_alpha', type=float, default=1.05)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--basis_elements', type=int, default=128)
    parser.add_argument('--core_layers', type=int, default=10)
    parser.add_argument('--decoder_layers', type=int, default=10)
    parser.add_argument('--encoder_layers', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--max_steps', type=int, default=10000000)
    parser.add_argument('--steps_per_eval', type=int, default=100)
    parser.add_argument('--epochs_no_improvement', type=int, default=20)
    parser.add_argument('--lr_decrease_stop', type=int, default=5)
    parser.add_argument('--use_classic', action='store_true')
    parser.add_argument('--use_soft_order', action='store_true')
    parser.add_argument('--datasets_per_step', type=int, default=32)
    parser.add_argument('--min_batch_size', type=int, default=200)
    parser.add_argument('--max_batch_size', type=int, default=5000)
    parser.add_argument('--resume_from_checkpoint', type=str)  # results/ch_test{timestamp}/checkpoint_{step}.pt
    args = parser.parse_args()

    ddp_setup(rank, world_size)

    # Set up logging
    if rank == 0:
        exp_dir = args.results_dir + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        os.makedirs(exp_dir)
        results_path = exp_dir + '/results_per_task.csv'
        metrics_path = exp_dir + '/metrics.csv'
        with open(metrics_path, 'w') as my_file:
            my_file.write('step,train_acc,val_acc,best_val_acc,test_acc\n')
        checkpoint_template = exp_dir + '/checkpoint_{}.pt'

    # Load datasets
    datasets = []
    _, _, file_names = [item for item in os.walk("sve/uci")][0]
    dataset_names = [file_name.split(".")[0] for file_name in file_names]

    for dataset_name in dataset_names:
        dataset_path = f'sve/uci/{dataset_name}.csv'
        dataset = load_dataset(dataset_path)
        datasets.append(dataset)

    num_variables_per_dataset = []
    num_variables = 0
    idx_bounds_per_dataset = []
    for dataset in datasets:
        idx_bounds_per_dataset.append([num_variables, num_variables + dataset['train'].shape[1]])
        num_variables_per_dataset.append(dataset['train'].shape[1])
        num_variables += dataset['train'].shape[1]
        for split in ['train', 'val', 'test']:
            data_set_tensor = dataset[split].to(rank)
            dataset[split] = data_set_tensor

    # Add task index for soft order
    for d, dataset in enumerate(datasets):
        dataset['dataset_idx'] = d

    # Build model
    if args.use_classic:
        print("Model type: Classic")
        assert args.context_size == args.hidden_size # Required for classical model
        model = ClassicalModel(args.context_size,
                               args.hidden_size,
                               args.core_layers,
                               dropout=args.dropout)
    elif args.use_soft_order:
        print("Model type: Soft Order")
        assert args.context_size == args.hidden_size # Required for soft order model
        model = SoftOrderModel(args.hidden_size,
                               args.core_layers,
                               len(datasets),
                               dropout=args.dropout)
    else:
        print("Model type: TOM")
        model = TravelingObserverModel(args.context_size,
                                       args.context_std,
                                       args.entmax_bisect_alpha,
                                       num_variables,
                                       args.basis_elements,
                                       args.hidden_size,
                                       args.encoder_layers,
                                       args.core_layers,
                                       args.decoder_layers,
                                       dropout=args.dropout)

    if args.resume_from_checkpoint:
        barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=map_location)
        model.load_state_dict(checkpoint['model_state'])

    model.to(rank)
    model = DDP(model, device_ids=[rank])
    model.train()

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    if args.resume_from_checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    # Train
    if args.resume_from_checkpoint and rank == 0:
        best_mean_val_acc = checkpoint['best_mean_val_acc']
        epochs_without_improvement = checkpoint['epochs_without_improvement']
        lr_decreases = checkpoint['lr_decreases']
        best_val_accs = checkpoint['best_val_accs']
        test_accs = checkpoint['test_accs']
    elif rank == 0:
        best_mean_val_acc = 0.
        epochs_without_improvement = 0
        lr_decreases = 0
        best_val_accs = [0. for dataset in datasets]
        test_accs = [0. for dataset in datasets]

    if args.resume_from_checkpoint:
        first_step = checkpoint['step'] + 1
    else:
        first_step = 0


    for step in tqdm(range(first_step, args.max_steps)):

        # Perform training update
        for d in range(args.datasets_per_step):
            dataset = np.random.choice(datasets)
            data_set_tensor = dataset['train']
            num_samples = data_set_tensor.shape[0]
            batch_idxs = np.random.choice(np.arange(num_samples),
                                          size=max(args.min_batch_size, args.max_batch_size),
                                          replace=True)
            
            batch = data_set_tensor[batch_idxs]

            input_var_indices = dataset['true_input_variable_indices']
            output_var_indices = dataset['true_output_variable_indices']
            batch_input = batch[:,input_var_indices]

            lower_var_idx_bound, upper_var_idx_bound = idx_bounds_per_dataset[dataset['dataset_idx']]
            var_indices = torch.arange(lower_var_idx_bound, upper_var_idx_bound).to(rank)

            pred = model(batch_input, var_indices, input_var_indices, output_var_indices)

            target = batch[:,output_var_indices]

            loss = squared_hinge(pred, target)

            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_accs = []
        val_accs = []
        train_losses = []
        val_losses = []
        # Only evaluate for one process.
        # Sync processes.
        barrier() 
        if step % args.steps_per_eval == 0 and rank == 0:
            print("Evaluating...")
            model.eval()
            print("entmax bisect alpha: ", model.module.entmax_bisect_alpha.data.item())
            for i, dataset in tqdm(enumerate(datasets)):
                train_loss, train_acc = compute_loss_and_accuracy(model, dataset, 'train',
                                                                  dataset['train'].shape[0],
                                                                  idx_bounds_per_dataset,
                                                                  rank,
                                                                  args.use_soft_order)
                train_accs.append(train_acc)
                train_losses.append(train_loss)
                val_loss, val_acc = compute_loss_and_accuracy(model, dataset, 'val',
                                                              dataset['val'].shape[0],
                                                              idx_bounds_per_dataset,
                                                              rank,
                                                              args.use_soft_order)
                val_accs.append(val_acc)
                val_losses.append(val_loss)
                if val_acc >= best_val_accs[i]:
                    best_val_accs[i] = val_acc
                    test_loss, test_acc = compute_loss_and_accuracy(model, dataset, 'test',
                                                                    dataset['test'].shape[0],
                                                                    idx_bounds_per_dataset,
                                                                    rank,
                                                                    args.use_soft_order)
                    test_accs[i] = test_acc
            mean_train_acc = np.mean(train_accs)
            mean_train_loss = np.mean(train_losses)
            mean_val_acc = np.mean(val_accs)
            mean_val_loss = np.mean(val_losses)
            mean_best_val_acc = np.mean(best_val_accs)
            mean_test_acc = np.mean(test_accs)

            df = pd.DataFrame({
                'test_acc': test_accs
            })
            df.to_csv(results_path)

            if mean_val_acc > best_mean_val_acc:
                epochs_without_improvement = 0
                best_mean_val_acc = mean_val_acc
            else:
                epochs_without_improvement += 1

            print('Step:', step)
            print('Mean Train Acc/Loss:', mean_train_acc, mean_train_loss)
            print('Mean Val Acc/Loss:', mean_val_acc, mean_val_loss)
            print('Best Mean Val Acc:', best_mean_val_acc)
            print('Mean Best Val Acc:', mean_best_val_acc)
            print('Mean Test Acc:', mean_test_acc)
            print('Epochs w/o Improvement', epochs_without_improvement)

            with open(metrics_path, 'a') as my_file:
                my_file.write(f'{step},{mean_train_acc},{mean_val_acc},{mean_best_val_acc},{mean_test_acc}\n')
            print("Wrote to", metrics_path)

            step_fmt = str(step).zfill(8)  # formatting depends on max steps
            checkpoint_path = checkpoint_template.format(step_fmt)

            torch.save({
                'step': step,
                'model_state': model.module.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': loss,
                'best_mean_val_acc': best_mean_val_acc,
                'epochs_without_improvement': epochs_without_improvement,
                'lr_decreases': lr_decreases,
                'best_val_accs': best_val_accs,
                'test_accs': test_accs,
            }, checkpoint_path)

            model.train()

            if epochs_without_improvement > args.epochs_no_improvement:
                lr_decreases += 1
                for g in optimizer.param_groups:
                    g['lr'] /= 2

            if lr_decreases > args.lr_decrease_stop:
                break

        # Sync processes.
        barrier()

    destroy_process_group()
    print('Done. Thank You.')


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f'Running training on {world_size} GPU(s).')
    mp.spawn(main, args=(world_size,), nprocs=world_size)
