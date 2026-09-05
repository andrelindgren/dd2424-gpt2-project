#!/usr/bin/env python3
'''
Trains only the linear head on features from extract_features.py.

uv run python train_head.py --train_features sst_train_features.pt --dev_features sst_dev_features.pt --lr 1e-3 --epochs 10 --use_gpu --run_name sst-head-lr1e-3

uv run python train_head.py --train_features sst_train_features.pt --dev_features sst_dev_features.pt --lr 3.78e-4 --batch_size 1 --grad_accum_steps 1024 --epochs 10000  --hidden_dropout_prob 0.1 --use_gpu --run_name sst-head-grad-test

'''
import argparse
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

from optimizer import AdamW
from classifier import seed_everything


class LinearHead(torch.nn.Module):

  def __init__(self, hidden_size, num_labels, hidden_dropout_prob):
    super().__init__()
    self.dropout = torch.nn.Dropout(hidden_dropout_prob)
    self.classifier_head = torch.nn.Linear(hidden_size, num_labels)

  def forward(self, feats):
    feats = self.dropout(feats)
    return self.classifier_head(feats)


# Eval but using features directly
def model_eval(feats, labels, model, batch_size):
  model.eval()
  y_pred = []

  for i in range(0, feats.shape[0], batch_size):
    logits = model(feats[i:i + batch_size])
    y_pred.append(logits.argmax(dim=1).cpu())
  y_pred = torch.cat(y_pred).numpy()
  y_true = labels.cpu().numpy()

  f1 = f1_score(y_true, y_pred, average='macro')
  acc = accuracy_score(y_true, y_pred)

  return acc, f1, y_pred


def save_model(model, optimizer, args, config, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'model_config': config,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  train_data = torch.load(args.train_features, weights_only=False)
  dev_data = torch.load(args.dev_features, weights_only=False)

  train_feats = train_data['features'].to(device)
  train_labels = train_data['labels'].to(device)
  dev_feats = dev_data['features'].to(device)
  dev_labels = dev_data['labels'].to(device)

  num_labels = int(train_labels.max().item()) + 1
  hidden_size = train_feats.shape[1]

  config = SimpleNamespace(
    hidden_dropout_prob=args.hidden_dropout_prob,
    num_labels=num_labels,
    hidden_size=hidden_size,
    data_dir='.',
    fine_tune_mode='last-linear-layer',
    num_unfrozen_layers=0,
  )

  head = LinearHead(hidden_size, num_labels, args.hidden_dropout_prob).to(device)
  optimizer = AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  best_dev_acc = 0

  # For my timekeeping / memory stats
  total_train_time = 0
  if args.run_name:
    os.makedirs('results', exist_ok=True)
    results_path = f'results/{args.run_name}.csv'
    with open(results_path, 'w') as f:
      f.write('epoch,train_loss,dev_acc,epoch_seconds,peak_mem_bytes\n')

  n = train_feats.shape[0]

  for epoch in range(args.epochs):
    head.train()
    train_loss = 0
    num_batches = 0

    # For my timekeeping / memory stats
    if args.use_gpu:
      torch.cuda.reset_peak_memory_stats()
    epoch_start = time.perf_counter()

    # Randomizer
    perm = torch.randperm(n, device=device)

    # For my extension: gradient accumulation
    optimizer.zero_grad()
    for i, start in enumerate(range(0, n, args.batch_size)):
      idx = perm[start:start + args.batch_size]

      logits = head(train_feats[idx])
      loss = F.cross_entropy(logits, train_labels[idx].view(-1), reduction='sum') / args.batch_size

      # For my extension: gradient accumulation
      loss = loss / args.grad_accum_steps

      loss.backward()

      # For my extension: gradient accumulation
      if (i + 1) % args.grad_accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

      # For my extension: gradient accumulation
      train_loss += loss.item() * args.grad_accum_steps
      num_batches += 1

    train_loss = train_loss / num_batches

    # For my timekeeping / memory stats
    epoch_seconds = time.perf_counter() - epoch_start
    total_train_time += epoch_seconds
    peak_mem_bytes = torch.cuda.max_memory_allocated() if args.use_gpu else 0

    # TODO Commented out to save time
    #train_acc, train_f1, _ = model_eval(train_feats, train_labels, head, args.batch_size)
    dev_acc, dev_f1, dev_pred = model_eval(dev_feats, dev_labels, head, args.batch_size)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      if not args.no_save:
        save_model(head, optimizer, args, config, args.filepath)

    # TODO Commented out to save time
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")

    # For my timekeeping / memory stats
    if args.run_name:
      with open(results_path, 'a') as f:
        f.write(f'{epoch},{train_loss},{dev_acc},{epoch_seconds},{peak_mem_bytes}\n')

  print(f'best dev acc :: {best_dev_acc:.3f}')
  print(f'total train time :: {total_train_time:.3f}s, peak mem :: {peak_mem_bytes / 1e9:.3f} GB')


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--seed', type=int, default=11711)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--use_gpu', action='store_true')
  parser.add_argument('--batch_size', type=int, default=8)
  parser.add_argument('--hidden_dropout_prob', type=float, default=0.3)
  parser.add_argument('--weight_decay', type=float, default=0.0)
  parser.add_argument('--lr', type=float, default=1e-5)
  parser.add_argument('--train_features')
  parser.add_argument('--dev_features')
  parser.add_argument('--grad_accum_steps', type=int, default=1)
  parser.add_argument('--run_name', type=str, default=None)
  parser.add_argument('--no_save', action='store_true')
  parser.add_argument('--filepath', type=str, default='head-classifier.pt')
  parser.add_argument('--dev_out', type=str, default=None)
  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  seed_everything(args.seed)
  train(args)