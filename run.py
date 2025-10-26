"""
This code was adapted from https://github.com/twitter-research/neural-sheaf-diffusion.git
"""

import argparse
import sys
import os
from models import ComplexWeightsDiffusion
import torch
import numpy as np
from tqdm import tqdm
import random
from datasets import get_dataset,get_fixed_splits,to_upper_triangle
import wandb
from distutils.util import strtobool
import torch.nn.functional as F


# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Utility: convert string -> bool (for argparse arguments like True/False)
def str2bool(x):
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')

# Argument parser: defines command-line options for training & model configuration    
def get_parser():
    parser = argparse.ArgumentParser()
    
    # Optimisation params
    parser.add_argument('--epochs', type=int, default=2000,help="Number of epochs")
    parser.add_argument('--lr', type=float, default=0.01,help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--complex_weight_decay', type=float, default=None,help="Weight decay for complex weights learning")

    parser.add_argument('--entity', type=str, default=None,help="Wandb entity name")

    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--min_acc', type=float, default=0.0,
                        help="Minimum test acc on the first fold to continue training.")
    parser.add_argument('--stop_strategy', type=str, choices=['loss', 'acc'], default='loss')
    
    # Model configuration
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=20)
    parser.add_argument('--input_dropout', type=float, default=0.0,help="Dropout rate in the input layer")
    parser.add_argument('--dropout', type=float, default=0.0,help="Dropout rate during diffusion")
    parser.add_argument('--left_weights', dest='left_weights', type=str2bool, default=True,
                        help="Applies left linear layer")
    parser.add_argument('--right_weights', dest='right_weights', type=str2bool, default=True,
                        help="Applies right linear layer")
    parser.add_argument('--use_act', dest='use_act', type=str2bool, default=True,help="Use activation function")
    parser.add_argument('--complex_weights_act', type=str, default="tanh", help="Activation to use in complex weights learner.")
    parser.add_argument('--batch_norm', type=str2bool, default=True,help="Apply batch normalisation after the input layer")
    
    # Experiment parameters
    parser.add_argument('--dataset', default='texas')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--folds', type=int, default=10)
    
    
    return parser

# Single training step: forward pass + loss + backward pass + optimizer step
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[data.train_mask]#data.x (size 183) nodes with features (size 1703)
    nll = F.nll_loss(out, data.y[data.train_mask])
    loss = nll
    loss.backward()

    optimizer.step()
    del out

# Evaluation step: compute accuracy and loss for train/val/test splits
def test(model, data):
    model.eval()
    with torch.no_grad():
        logits, accs, losses, preds = model(data.x), [], [], []
        # Loop over train/val/test masks
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses
    
# Run one experiment for a given dataset split (fold)    
def run_exp(args, dataset, model_cls, fold):
    data = dataset[0]
    # Ensure undirected graph (upper triangular form of adjacency)
    data.edge_index=to_upper_triangle(data.edge_index)
    # Assign train/val/test masks for this fold: data contains fields data.train_mask,data.val_mask,data.test_mask
    data = get_fixed_splits(data, args['dataset'], fold)
    data = data.to(args['device'])
    
    # Initialize model
    model = model_cls(data.edge_index, args)
    model = model.to(args['device'])
    
    # Different weight decay for complex-weighted matrix parameters and rest of parameters
    weights_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {'params': weights_learner_params, 'weight_decay': args['complex_weight_decay']},
        {'params': other_params, 'weight_decay': args['weight_decay']}
    ], lr=args['lr'])
    
    epoch = 0
    
    # Tracking variables
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')

    best_epoch = 0
    bad_counter = 0
    
    # Training loop
    for epoch in range(args['epochs']):
        train(model, optimizer, data)
        
        # Evaluate performance
        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)
        
        # Log only fold 0 results (for wandb visualization)
        if fold == 0:
            res_dict = {
                f'fold{fold}_train_acc': train_acc,
                f'fold{fold}_train_loss': train_loss,
                f'fold{fold}_val_acc': val_acc,
                f'fold{fold}_val_loss': val_loss,
                f'fold{fold}_tmp_test_acc': tmp_test_acc,
                f'fold{fold}_tmp_test_loss': tmp_test_loss,
            }
            wandb.log(res_dict, step=epoch)
            
        # Check if validation improved (based on chosen stop_strategy)
        new_best_trigger = val_acc > best_val_acc if args['stop_strategy'] == 'acc' else val_loss < best_val_loss
        if new_best_trigger:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1
            
        # Early stopping
        if bad_counter == args['early_stopping']:
            break
        
    # Print summary for this fold        
    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}")
    # Log best results to wandb
    wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})
    # Decide if training should continue (based on min_acc threshold)
    keep_running = False if test_acc < args['min_acc'] else True

   
    return test_acc, best_val_acc, keep_running

if __name__ == '__main__':
    
    parser = get_parser()
    args = parser.parse_args()
    model_cls=ComplexWeightsDiffusion
    dataset = get_dataset(args.dataset)
    
    # Set dataset/model dimensions    
    args.graph_size = dataset[0].x.size(0)
    args.input_dim = dataset.num_features
    args.output_dim = dataset.num_classes
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    
    # If not set, use same weight decay for complex weights
    if args.complex_weight_decay is None:
        args.complex_weight_decay = args.weight_decay
    
    # Set random seeds 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    results = []
    print(f"Running with wandb account: {args.entity}")
    print(args)
    # Initialize wandb run
    wandb.init(project="complex-weights-sweep", config=vars(args), entity=args.entity)
    
    # Run experiments for all folds
    for fold in tqdm(range(args.folds)):
        test_acc, best_val_acc, keep_running = run_exp(wandb.config, dataset, model_cls, fold)

        results.append([test_acc, best_val_acc])

        if not keep_running:
            break
    # Compute mean/std results across fold        
    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    # Log final results to wandb
    wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std}
    wandb.log(wandb_results)
    wandb.finish()
    
    # Print final summary
    print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')
