#!/usr/bin/env python3

#
# LLM disclosure: Generated boilerplate/schafolding file that does the following I asked:
#
# 1. Run: Do a seeded sweep over checkpoints x (temperature, top_p) x seeds
# 2. Analysis: Automatic calculation of the 95% confidence interval and print standard deviation and winners/ties.
# 3. Speed up this process (it added caching)
#
# I myself picked the range of temps and top_p.
# Decided the range from the results of sonnet_sweep and shrinked down because of limited time.
#

"""Seeded sweep over checkpoints x (temperature, top_p) x seeds.

  python sonnet_sweep_seeded.py plan               # print the budget, run nothing
  python sonnet_sweep_seeded.py run                # the overnight job
  python sonnet_sweep_seeded.py analyze            # mean/sem per cell, ties flagged

Resumable: every (checkpoint, temperature, top_p, seed) row already in the CSV
is skipped, so Ctrl-C and restart is safe.

"""

import argparse
import csv
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch

from sonnet_generation import SonnetGPT, seed_everything
from evaluation import test_sonnet
from datasets import SonnetsDataset

# ---------------------------------------------------------------- config ----

CHECKPOINTS = [
  '5-0.003-sonnet.pt',
  '5-0.001-sonnet.pt',
  '5-0.0003-sonnet.pt',
  '5-0.0001-sonnet.pt',
  '5-3e-05-sonnet.pt',
  '5-1e-05-sonnet.pt',
  '5-3e-06-sonnet.pt',
  '10-0.001-sonnet.pt',
  '10-0.0003-sonnet.pt',
  '10-0.0001-sonnet.pt',
]


CONFIGS = [
  # temperature, top_p 0.9
  (0.9, 0.9),
  (1.0, 0.9),
  (1.1, 0.9),
  (1.2, 0.9),
  (1.4, 0.9),
  # top_p, temperature 1.1
  (1.1, 0.85),
  (1.1, 0.95),
  (1.1, 1.0),
  # extra
  (0.9, 0.95),
  (1.2, 0.85),
  # outliers
  (0.6, 0.9),
  (2.0, 0.9),
  (1.0, 0.5),
]

SEEDS = [0, 1, 2, 3, 4]

GOLD_PATH = 'data/TRUE_sonnets_held_out_dev.txt'
HELD_OUT_PATH = 'data/sonnets_held_out_dev.txt'

RESULTS_DIR = 'results_sonnet_seeded'
RESULTS_CSV = f'{RESULTS_DIR}/sonnet_sweep_seeded.csv'
PRED_DIR = 'predictions/seeded'

USE_GPU = True
FIELDS = ['checkpoint', 'temperature', 'top_p', 'seed', 'chrf', 'seconds']


# ------------------------------------------------------------------ io ------

def key(checkpoint, temperature, top_p, seed):
  return (checkpoint, f'{temperature:g}', f'{top_p:g}', str(seed))


def done_keys():
  if not os.path.exists(RESULTS_CSV):
    return set()
  with open(RESULTS_CSV) as f:
    return {(r['checkpoint'], r['temperature'], r['top_p'], r['seed'])
            for r in csv.DictReader(f)}


def append_row(checkpoint, temperature, top_p, seed, chrf, seconds):
  os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
  new = not os.path.exists(RESULTS_CSV)
  with open(RESULTS_CSV, 'a', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
      w.writeheader()
    k = key(checkpoint, temperature, top_p, seed)
    w.writerow(dict(checkpoint=k[0], temperature=k[1], top_p=k[2], seed=k[3],
                    chrf=f'{chrf:.4f}', seconds=f'{seconds:.1f}'))
    f.flush()
    os.fsync(f.fileno())  # survive a hard kill mid-run


def load_rows():
  if not os.path.exists(RESULTS_CSV):
    return []
  with open(RESULTS_CSV) as f:
    return [dict(r, chrf=float(r['chrf']), seed=int(r['seed'])) for r in csv.DictReader(f)]


# -------------------------------------------------------------- generate ----

_model_cache = {'path': None, 'model': None}


def get_model(checkpoint, device):
  """Cache the last-loaded checkpoint."""
  if _model_cache['path'] != checkpoint:
    _model_cache['model'] = None
    if device.type == 'cuda':
      torch.cuda.empty_cache()
    saved = torch.load(checkpoint, weights_only=False)
    model = SonnetGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()
    _model_cache.update(path=checkpoint, model=model)
  return _model_cache['model']


@torch.no_grad()
def generate_to_file(model, dataset, out_path, temperature, top_p):
  """Mirrors generate_submission_sonnets() but takes an already-loaded model."""
  generated = []
  for batch in dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False,
                               truncation=True).to(model.get_device())
    output = model.generate(encoding['input_ids'], temperature=temperature, top_p=top_p)[0][0]
    generated.append((sonnet_id, f'{model.tokenizer.decode(output)}\n\n'))

  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  with open(out_path, 'w+', encoding='utf-8') as f:
    f.write('--Generated Sonnets-- \n\n')
    for sonnet_id, text in generated:
      f.write(f'\n{sonnet_id}\n')
      f.write(text)


# ------------------------------------------------------------------ run ------

def plan():
  total = len(CHECKPOINTS) * len(CONFIGS) * len(SEEDS)
  done = len(done_keys())
  rows = load_rows()
  secs = [float(r['seconds']) for r in rows if r.get('seconds')]
  print(f'{len(CHECKPOINTS)} checkpoints x {len(CONFIGS)} configs x {len(SEEDS)} seeds'
        f' = {total} runs')
  print(f'already in {RESULTS_CSV}: {done}   remaining: {total - done}')
  if secs:
    med = float(np.median(secs))
    print(f'median {med:.1f}s/run  ->  ETA {(total - done) * med / 3600:.1f}h')
  else:
    print('no timing data yet')
  print(f'\none seed pass over everything = {len(CHECKPOINTS) * len(CONFIGS)} runs')


def run():
  device = torch.device('cuda') if USE_GPU else torch.device('cpu')
  dataset = SonnetsDataset(HELD_OUT_PATH)
  done = done_keys()
  started = time.time()
  n_done = 0

  for seed in SEEDS:
    for checkpoint in CHECKPOINTS:
      tag = os.path.splitext(os.path.basename(checkpoint))[0]
      for temperature, top_p in CONFIGS:
        if key(checkpoint, temperature, top_p, seed) in done:
          continue

        t0 = time.time()
        seed_everything(seed)
        model = get_model(checkpoint, device)
        out = f'{PRED_DIR}/{tag}_temp{temperature:g}_topp{top_p:g}_seed{seed}.txt'
        generate_to_file(model, dataset, out, temperature, top_p)
        chrf = test_sonnet(test_path=out, gold_path=GOLD_PATH)
        elapsed = time.time() - t0

        append_row(checkpoint, temperature, top_p, seed, chrf, elapsed)
        n_done += 1
        print(f'seed={seed} {tag} temp={temperature:g} top_p={top_p:g} '
              f'CHRF={chrf:.3f} ({elapsed:.0f}s)', flush=True)

  print(f'\ndone: {n_done} new runs in {(time.time() - started) / 3600:.1f}h')


# -------------------------------------------------------------- analyze -----

def tie_z(mean_a, se_a, mean_b, se_b):
  """(a - b) in units of the sem of the difference. nan if a cell has n=1."""
  if math.isnan(se_a) or math.isnan(se_b):
    return float('nan')
  se_diff = math.sqrt(se_a ** 2 + se_b ** 2)
  if se_diff == 0:
    return 0.0 if mean_a == mean_b else float('inf')
  return (mean_a - mean_b) / se_diff


def analyze():
  rows = load_rows()
  if not rows:
    print(f'nothing in {RESULTS_CSV} yet')
    return

  cells = defaultdict(list)
  for r in rows:
    cells[(r['checkpoint'], r['temperature'], r['top_p'])].append(r['chrf'])

  def stats(v):
    v = np.asarray(v)
    sd = v.std(ddof=1) if len(v) > 1 else float('nan')
    return v.mean(), sd, (sd / math.sqrt(len(v)) if len(v) > 1 else float('nan')), len(v)

  by_ckpt = defaultdict(list)
  for (ckpt, t, p), v in cells.items():
    by_ckpt[ckpt].append(((t, p), stats(v)))

  for ckpt in sorted(by_ckpt):
    print(f'\n=== {ckpt} ===')
    entries = sorted(by_ckpt[ckpt], key=lambda e: -e[1][0])
    best_mean, _, best_se, _ = entries[0][1]
    print(f"{'temp':>6} {'top_p':>6} {'n':>3} {'mean':>7} {'sd':>6} {'sem':>6}  ")
    for (t, p), (mean, sd, se, n) in entries:
      # Two-sided 0.05 check against winner. If inside margin = winner is tied.
      # It's a tie if: (best_mean - mean) < 1.96 * se_diff
      z = tie_z(best_mean, best_se, mean, se)
      tie = 'tied with best' if not math.isnan(z) and z < 1.96 else ''
      print(f'{t:>6} {p:>6} {n:>3} {mean:>7.3f} {sd:>6.3f} {se:>6.3f}  {tie}')

  print('\n=== best per checkpoint ===')
  for ckpt in sorted(by_ckpt):
    (t, p), (mean, sd, se, n) = max(by_ckpt[ckpt], key=lambda e: e[1][0])
    print(f'{ckpt:<24} temp={t:>5} top_p={p:>5}  {mean:.3f} +/- {se:.3f} (n={n})')

  print('\n=== checkpoints ranked by best cell ===')
  best_cell = {c: max(v, key=lambda e: e[1][0]) for c, v in by_ckpt.items()}
  order = sorted(best_cell.items(), key=lambda kv: -kv[1][1][0])
  (_, (_, (top_mean, _, top_se, _))) = order[0]
  print(f"{'checkpoint':<24}{'temp':>6}{'top_p':>7}{'mean':>9}{'sem':>7}"
        f"{'delta':>8}{'z':>6}  ")
  for ckpt, ((t, p), (mean, sd, se, n)) in order:
    z = tie_z(top_mean, top_se, mean, se)
    tie = 'tied with best' if not math.isnan(z) and z < 1.96 else ''
    print(f'{ckpt:<24}{t:>6}{p:>7}{mean:>9.3f}{se:>7.3f}'
          f'{top_mean - mean:>8.3f}{z:>6.2f}  {tie}')



if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('mode', choices=['plan', 'run', 'analyze'])
  a = ap.parse_args()
  if a.mode == 'plan':
    plan()
  elif a.mode == 'run':
    run()
  else:
    analyze()
