#!/usr/bin/env python3

#
# LLM disclosure: Generated boilerplate/schafolding file that does the following I asked:
#
# 1. Do a hyperparameter sweep for each type of sst model: head, full, top 2/4/8 layers
# 2. Hyperparameters are learning rate, dropout and weight decay.
# 3. Do seeded runs (3 with best full) (5 with head) for the best model. This is for determiningnthe spread between models.
# 4. Take the best hyperparameters from sst and use for cfimdb.
#
# Extended with the following:
# 4. Run an extended search on head with random selection. Do 3 seeds per config.
# Taking the mean of the 3 seeds is to try to avoid a lucky dev value from misrepresenting how good a config is.
#
# I myself picked the range of hyperparameters


'''


uv run python sweep.py --stage all --use_gpu
uv run python sweep.py --stage lr_full --use_gpu

Every run appends one row to results/runs.csv. Runs whose name is already in that file are
skipped, so the sweep can be stopped and restarted across sessions.
'''
import argparse
import csv
import math
import os
import random
import time
from types import SimpleNamespace

import classifier
import extract_features
import train_head

RUNS_CSV = 'results/runs.csv'

FIELDS = ['dataset', 'arm', 'stage', 'run_name', 'fine_tune_mode', 'num_unfrozen_layers',
          'lr', 'weight_decay', 'dropout', 'batch_size', 'epochs', 'seed', 'n_train',
          'opt_steps', 'best_dev_acc', 'best_epoch', 'wall_seconds', 'peak_mem_bytes']

DATA = {
  'sst': dict(train='data/ids-sst-train.csv', dev='data/ids-sst-dev.csv',
              batch_size=8, max_length=None),
  'cfimdb': dict(train='data/ids-cfimdb-train.csv', dev='data/ids-cfimdb-dev.csv',
                 batch_size=2, max_length=None),
}

EPOCHS = 5
SEED = 11711

HEAD_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
FULL_LRS = [5e-6, 1e-5, 3e-5, 1e-4]
LADDER_LRS = [1e-5, 3e-5, 1e-4]
LADDER_LAYERS = [2, 4, 8]
DROPOUTS = [0.0, 0.1, 0.3]
WEIGHT_DECAYS = [0.01, 0.1]

# matched-budget head search: spend the full arm's search time on the head arm instead
BUDGET_STAGES = {'lr', 'ladder', 'reg'}   # what counts as search effort
BUDGET_SEEDS = 3                          # seeds per config; search on the mean
#BUDGET_LR_RANGE = (-4.5, -1.5)            # log10
#BUDGET_LR_RANGE = (-7.0, -1.0)            # log10
BUDGET_LR_RANGE = (-7.0, -4.0)            # log10

BUDGET_DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.5]
BUDGET_WDS = [0.0, 1e-3, 1e-2, 1e-1]
#BUDGET_EPOCHS = [5, 10, 20, 40]
#BUDGET_EPOCHS = [5, 10, 20, 40, 80, 160]
BUDGET_EPOCHS = [80]



# ---------- bookkeeping ----------

def read_runs():
  if not os.path.exists(RUNS_CSV):
    return []
  with open(RUNS_CSV) as f:
    return list(csv.DictReader(f))


def already_done(run_name):
  return any(r['run_name'] == run_name for r in read_runs())


def append_row(row):
  os.makedirs('results', exist_ok=True)
  new = not os.path.exists(RUNS_CSV)
  with open(RUNS_CSV, 'a', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
      w.writeheader()
    w.writerow(row)


def read_epoch_csv(run_name):
  '''Pull best dev acc, its epoch and the peak memory out of the per-epoch log.'''
  path = f'results/{run_name}.csv'
  best_acc, best_epoch, peak_mem = 0.0, -1, 0
  with open(path) as f:
    for r in csv.DictReader(f):
      acc = float(r['dev_acc'])
      if acc > best_acc:
        best_acc, best_epoch = acc, int(r['epoch'])
      peak_mem = max(peak_mem, int(float(r['peak_mem_bytes'])))
  return best_acc, best_epoch, peak_mem


_n_train_cache = {}


def n_train(dataset):
  if dataset not in _n_train_cache:
    data, _ = classifier.load_data(DATA[dataset]['train'], 'train')
    _n_train_cache[dataset] = len(data)
  return _n_train_cache[dataset]


def best_row(dataset, arm, stages):
  '''Best logged run for an arm, restricted to the given stages.'''
  rows = [r for r in read_runs()
          if r['dataset'] == dataset and r['arm'] == arm and r['stage'] in stages
          and r['best_dev_acc']]
  if not rows:
    raise SystemExit(f'no runs logged yet for {dataset}/{arm} in {stages} - run those stages first')
  return max(rows, key=lambda r: float(r['best_dev_acc']))


# ---------- runners ----------

def features_path(dataset, split):
  return f'features/{dataset}_{split}.pt'


def run_extract(dataset, split, use_gpu):
  name = f'extract-{dataset}-{split}'
  if already_done(name):
    print(f'skip {name}')
    return
  os.makedirs('features', exist_ok=True)
  d = DATA[dataset]
  cfg = SimpleNamespace(
    csv=d['train'] if split == 'train' else d['dev'],
    flag='train' if split == 'train' else 'valid',
    filepath=features_path(dataset, split),
    batch_size=d['batch_size'],
    max_length=d['max_length'],
    use_gpu=use_gpu,
    run_name=None,
  )
  classifier.seed_everything(SEED)
  t0 = time.perf_counter()
  extract_features.extract(cfg)
  wall = time.perf_counter() - t0

  row = {k: '' for k in FIELDS}
  row.update(dataset=dataset, arm='head', stage='extract', run_name=name,
             batch_size=d['batch_size'], wall_seconds=f'{wall:.1f}')
  append_row(row)
  print(f'{name}: {wall:.1f}s')


def run_head(dataset, stage, name, lr, dropout=0.3, wd=0.0, seed=SEED, epochs=EPOCHS,
             arm='head'):
  if already_done(name):
    print(f'skip {name}')
    return 0.0
  d = DATA[dataset]
  cfg = SimpleNamespace(
    train_features=features_path(dataset, 'train'),
    dev_features=features_path(dataset, 'dev'),
    lr=lr, weight_decay=wd, hidden_dropout_prob=dropout,
    epochs=epochs, batch_size=d['batch_size'], grad_accum_steps=1,
    use_gpu=USE_GPU, run_name=name, no_save=True, filepath='head-classifier.pt',
  )
  classifier.seed_everything(seed)
  t0 = time.perf_counter()
  train_head.train(cfg)
  wall = time.perf_counter() - t0
  acc, epoch, mem = read_epoch_csv(name)

  n = n_train(dataset)
  append_row(dict(
    dataset=dataset, arm=arm, stage=stage, run_name=name,
    fine_tune_mode='last-linear-layer', num_unfrozen_layers=0,
    lr=lr, weight_decay=wd, dropout=dropout, batch_size=d['batch_size'],
    epochs=epochs, seed=seed, n_train=n,
    opt_steps=epochs * math.ceil(n / d['batch_size']),
    best_dev_acc=f'{acc:.4f}', best_epoch=epoch,
    wall_seconds=f'{wall:.1f}', peak_mem_bytes=mem))
  print(f'{name}: dev {acc:.4f} in {wall:.1f}s')
  return wall


def run_gpt(dataset, stage, name, lr, unfrozen, dropout=0.3, wd=0.0, seed=SEED, epochs=EPOCHS):
  '''unfrozen: 2/4/8 -> top-layers, 12 -> full-model.'''
  if already_done(name):
    print(f'skip {name}')
    return
  d = DATA[dataset]
  mode = 'full-model' if unfrozen >= 12 else 'top-layers'
  arm = 'full' if unfrozen >= 12 else f'top{unfrozen}'
  cfg = SimpleNamespace(
    train=d['train'], dev=d['dev'], max_length=d['max_length'],
    fine_tune_mode=mode, num_unfrozen_layers=unfrozen,
    lr=lr, weight_decay=wd, hidden_dropout_prob=dropout,
    epochs=epochs, batch_size=d['batch_size'], grad_accum_steps=1,
    use_gpu=USE_GPU, run_name=name, no_save=True, filepath='sweep-classifier.pt',
  )
  classifier.seed_everything(seed)
  t0 = time.perf_counter()
  classifier.train(cfg)
  wall = time.perf_counter() - t0
  acc, epoch, mem = read_epoch_csv(name)

  n = n_train(dataset)
  append_row(dict(
    dataset=dataset, arm=arm, stage=stage, run_name=name,
    fine_tune_mode=mode, num_unfrozen_layers=unfrozen,
    lr=lr, weight_decay=wd, dropout=dropout, batch_size=d['batch_size'],
    epochs=epochs, seed=seed, n_train=n,
    opt_steps=epochs * math.ceil(n / d['batch_size']),
    best_dev_acc=f'{acc:.4f}', best_epoch=epoch,
    wall_seconds=f'{wall:.1f}', peak_mem_bytes=mem))
  print(f'{name}: dev {acc:.4f} in {wall:.1f}s')


# ---------- stages ----------

def stage_extract():
  for dataset in ('sst', 'cfimdb'):
    for split in ('train', 'dev'):
      run_extract(dataset, split, USE_GPU)


def stage_lr_head():
  for lr in HEAD_LRS:
    run_head('sst', 'lr', f'sst-head-lr{lr:g}', lr)


def stage_lr_full():
  for lr in FULL_LRS:
    run_gpt('sst', 'lr', f'sst-full-lr{lr:g}', lr, unfrozen=12)


def stage_ladder():
  for layers in LADDER_LAYERS:
    for lr in LADDER_LRS:
      run_gpt('sst', 'ladder', f'sst-top{layers}-lr{lr:g}', lr, unfrozen=layers)


def stage_reg():
  # head: all combinations
  lr = float(best_row('sst', 'head', ['lr'])['lr'])
  for dropout in DROPOUTS:
    for wd in [0.0] + WEIGHT_DECAYS:
      run_head('sst', 'reg', f'sst-head-do{dropout}-wd{wd:g}', lr, dropout=dropout, wd=wd)

  # full: one axis at a time, starting from the best lr run
  lr = float(best_row('sst', 'full', ['lr'])['lr'])
  for dropout in DROPOUTS:
    if dropout != 0.3:
      run_gpt('sst', 'reg', f'sst-full-do{dropout}', lr, unfrozen=12, dropout=dropout)
  best_do = float(best_row('sst', 'full', ['lr', 'reg'])['dropout'])
  for wd in WEIGHT_DECAYS:
    run_gpt('sst', 'reg', f'sst-full-wd{wd:g}', lr, unfrozen=12, dropout=best_do, wd=wd)


def stage_seeds():
  b = best_row('sst', 'head', ['lr', 'reg'])
  for seed in [1, 2, 3, 4, 5]:
    run_head('sst', 'seed', f'sst-head-seed{seed}', float(b['lr']),
             dropout=float(b['dropout']), wd=float(b['weight_decay']), seed=seed)

  b = best_row('sst', 'full', ['lr', 'reg'])
  for seed in [1, 2, 3]:
    run_gpt('sst', 'seed', f'sst-full-seed{seed}', float(b['lr']), unfrozen=12,
            dropout=float(b['dropout']), wd=float(b['weight_decay']), seed=seed)


def stage_cfimdb():
  for lr in HEAD_LRS:
    run_head('cfimdb', 'lr', f'cfimdb-head-lr{lr:g}', lr)

  b = best_row('sst', 'full', ['lr', 'reg'])
  lr = float(b['lr'])
  run_gpt('cfimdb', 'transfer', f'cfimdb-full-best', lr, unfrozen=12,
            dropout=float(b['dropout']), wd=float(b['weight_decay']))


def stage_cfimdb_seeds():
  b = best_row('cfimdb', 'full', ['transfer'])
  for seed in [1, 2, 3]:
    run_gpt('cfimdb', 'seed', f'cfimdb-full-seed{seed}', float(b['lr']), unfrozen=12,
            dropout=float(b['dropout']), wd=float(b['weight_decay']), seed=seed)


def spent_seconds(dataset, arm, stages):
  return sum(float(r['wall_seconds']) for r in read_runs()
             if r['dataset'] == dataset and r['arm'] == arm and r['stage'] in stages
             and r['wall_seconds'])


def stage_head_budget():
  dataset = BUDGET_DATASET
  budget = spent_seconds(dataset, 'full', BUDGET_STAGES)
  if budget == 0:
    raise SystemExit('no full-model search runs logged yet - nothing to match the budget to')

  # the head has already spent time on extraction and on its own 5-epoch search
  used = (spent_seconds(dataset, 'head', BUDGET_STAGES | {'extract'})
          + spent_seconds(dataset, 'head_ext', {'budget'}))
  print(f'full arm spent {budget:.0f}s; head arm has used {used:.0f}s')

  rng = random.Random(0)
  i = 0
  while used < budget:
    lr = 10 ** rng.uniform(*BUDGET_LR_RANGE)
    dropout = rng.choice(BUDGET_DROPOUTS)
    wd = rng.choice(BUDGET_WDS)
    epochs = rng.choice(BUDGET_EPOCHS)
    for s in range(BUDGET_SEEDS):
      name = f'{dataset}-headx-{i:03d}--seed{s}'
      used += run_head(dataset, 'budget', name, lr, dropout=dropout, wd=wd,
                       seed=SEED + s, epochs=epochs, arm='head_ext')
    i += 1
  print(f'head arm now at {used:.0f}s of the {budget:.0f}s budget, {i} configs sampled')


STAGES = {
  'extract': stage_extract,
  'lr_head': stage_lr_head,
  'lr_full': stage_lr_full,
  'ladder': stage_ladder,
  'reg': stage_reg,
  'seeds': stage_seeds,
  'cfimdb': stage_cfimdb,
  'cfimdb_seeds': stage_cfimdb_seeds,
  'head_budget': stage_head_budget,
}

ORDER = ['extract', 'lr_head', 'lr_full', 'ladder', 'reg', 'seeds', 'cfimdb', 'cfimdb_seeds', 'head_budget']


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--stage', default='all', choices=['all'] + ORDER)
  parser.add_argument('--use_gpu', action='store_true')
  parser.add_argument('--budget_dataset', default='sst', choices=['sst', 'cfimdb'],
                      help='which dataset the head_budget stage runs on')
  args = parser.parse_args()

  USE_GPU = args.use_gpu
  BUDGET_DATASET = args.budget_dataset
  todo = ORDER if args.stage == 'all' else [args.stage]
  for s in todo:
    print(f'=== stage {s} ===')
    STAGES[s]()

